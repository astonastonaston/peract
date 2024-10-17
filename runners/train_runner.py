import copy
import logging
import os
import shutil
import signal
import sys
import threading
import time
from typing import Optional, List
from typing import Union

from omegaconf import DictConfig
import gc
import numpy as np
import psutil
import torch
import pandas as pd
from torchvision import transforms
from agents.agent import Agent
from runners.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from runners.log_writer import LogWriter
from runners.stat_accumulator import StatAccumulator


class OfflineTrainRunner():

    def __init__(self,
                 agent: Agent,
                 wrapped_replay_buffer: PyTorchReplayBuffer,
                 train_device: torch.device,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(6e6),
                 logdir: str = '/tmp/runtime/logs',
                 logging_level: int = logging.INFO,
                 log_freq: int = 10,
                 weightsdir: str = '/tmp/runtime/weights',
                 num_weights_to_keep: int = 60,
                 save_freq: int = 100,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False,
                 load_existing_weights: bool = True,
                 rank: int = None,
                 world_size: int = None):
        self._agent = agent
        self._wrapped_buffer = wrapped_replay_buffer
        # self._stat_accumulator = stat_accumulator
        self._iterations = iterations
        self._logdir = logdir
        self._logging_level = logging_level
        self._log_freq = log_freq
        self._weightsdir = weightsdir
        self._num_weights_to_keep = num_weights_to_keep
        self._save_freq = save_freq

        self._wrapped_buffer = wrapped_replay_buffer
        self._train_device = train_device
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        self._load_existing_weights = load_existing_weights
        self._rank = rank
        self._world_size = world_size

        self._writer = None
        if logdir is None:
            logging.info("'logdir' was None. No logging will take place.")
        else:
            self._writer = LogWriter(
                self._logdir, tensorboard_logging, csv_logging)

        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)

    def _save_model(self, i):
        d = os.path.join(self._weightsdir, str(i))
        os.makedirs(d, exist_ok=True)
        self._agent.save_weights(d)

        # remove oldest save
        prev_dir = os.path.join(self._weightsdir, str(
            i - self._save_freq * self._num_weights_to_keep))
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)

    def _step(self, i, sampled_batch):
        # print(f"in step, rgb scale {sampled_batch['rgb'][0][0][0]}")
        # if sampled_batch["demo_number"][0] == 0:
        #     print("in step, sampled batch has pcd rgb")
        #     print(sampled_batch['point_cloud'])
        #     print(sampled_batch['rgb'])
        #     print()
        update_dict = self._agent.update(i, sampled_batch)
        total_losses = update_dict['total_losses'].item()
        # # save voxel img
        # img, demo_num, inp_fr, sup_fr = update_dict['voxel_img'], update_dict['demo_number'], update_dict['input_frame'], update_dict['supervision_frame']
        # img = img.transpose(1, 2, 0)
        # print(f"shape of voxel img {img.shape}")
        # to_pil = transforms.ToPILImage()
        # img = to_pil(img)
        # img.save(os.path.join(self._logdir, f'step_{i}_demo_{demo_num}_inp_{inp_fr}_sup_{sup_fr}_loss_{total_losses:0.5f}.png'))
        return total_losses

    def _get_resume_eval_epoch(self):
        starting_epoch = 0
        eval_csv_file = self._weightsdir.replace('weights', 'eval_data.csv') # TODO(mohit): check if it's supposed be 'env_data.csv'
        if os.path.exists(eval_csv_file):
             eval_dict = pd.read_csv(eval_csv_file).to_dict()
             epochs = list(eval_dict['step'].values())
             return epochs[-1] if len(epochs) > 0 else starting_epoch
        else:
            return starting_epoch

    def start(self):
        logging.getLogger().setLevel(self._logging_level)
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device)

        if self._weightsdir is not None:
            existing_weights = sorted([int(f) for f in os.listdir(self._weightsdir)])
            if (not self._load_existing_weights) or len(existing_weights) == 0:
                self._save_model(0)
                start_iter = 0
            else:
                resume_iteration = existing_weights[-1]
                self._agent.load_weights(os.path.join(self._weightsdir, str(resume_iteration)))
                start_iter = resume_iteration + 1
                if self._rank == 0:
                    logging.info(f"Resuming training from iteration {resume_iteration} ...")

        dataset = self._wrapped_buffer.dataset()
        data_iter = iter(dataset)

        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        for i in range(start_iter, self._iterations):
            log_iteration = i % self._log_freq == 0 and i > 0

            if log_iteration:
                process.cpu_percent(interval=None)

            t = time.time()
            sampled_batch = next(data_iter)
            sample_time = time.time() - t

            batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if type(v) == torch.Tensor}
            t = time.time()
            # print(batch.keys())
            # print(batch["rgb"].shape)
            # print(batch["rgb"][0][0][0])
            
            loss = self._step(i, batch)
            step_time = time.time() - t


            if self._rank == 0:
                if log_iteration and self._writer is not None:
                    agent_summaries = self._agent.update_summaries()
                    # print(agent_summaries)
                    # for j in agent_summaries:
                    #     print(j.name)
                    self._writer.add_summaries(i, agent_summaries)

                    self._writer.add_scalar(
                        i, 'monitoring/memory_gb',
                        process.memory_info().rss * 1e-9)
                    self._writer.add_scalar(
                        i, 'monitoring/cpu_percent',
                        process.cpu_percent(interval=None) / num_cpu)

                    # demo_number = batch['demo_number'].int()
                    # input_frame = batch['input_frame'].int()
                    # supervision_frame = batch['supervision_frame'].int()

                    logging.info(f"Train Step {i:06d} | Loss: {loss:0.5f} | Sample time: {sample_time:0.6f} | Step time: {step_time:0.4f}.")
                    # logging.info(f"Using demo {demo_number} from frame {input_frame} to frame {supervision_frame}")
                self._writer.end_iteration()

                if i % self._save_freq == 0 and self._weightsdir is not None:
                    self._save_model(i)

        if self._rank == 0 and self._writer is not None:
            self._writer.close()
            logging.info('Stopping envs ...')

            self._wrapped_buffer.replay_buffer.shutdown()