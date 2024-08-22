import os
import pickle
import gc
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

# from rlbench import CameraConfig, ObservationConfig
# from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
# from yarr.runners.offline_train_runner import OfflineTrainRunner
# from yarr.utils.stat_accumulator import SimpleAccumulator

# from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

# from agents import c2farm_lingunet_bc
# from agents import arm
# from agents.baselines import bc_lang, vit_bc_lang
from agents import peract_bc


def run_seed(rank,
             cfg: DictConfig,
            #  obs_config: ObservationConfig,
             cams,
             multi_task,
             seed,
             world_size) -> None:
    dist.init_process_group("gloo",
                            rank=rank,
                            world_size=world_size)

    task = cfg.maniskill3.tasks[0]
    tasks = cfg.maniskill3.tasks

    task_folder = task if not multi_task else 'multi'
    replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d' % seed)

    if cfg.method.name == 'ARM':
        raise NotImplementedError("Only PERACT_BC is supported for maniskill yet")

    elif cfg.method.name == 'BC_LANG':
        raise NotImplementedError("Only PERACT_BC is supported for maniskill yet")

    elif cfg.method.name == 'VIT_BC_LANG':
        raise NotImplementedError("Only PERACT_BC is supported for maniskill yet")

    elif cfg.method.name == 'C2FARM_LINGUNET_BC':
        raise NotImplementedError("Only PERACT_BC is supported for maniskill yet")

    elif cfg.method.name == 'PERACT_BC':
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.maniskill3.camera_resolution)

        peract_bc.launch_utils.fill_multi_task_replay(
            cfg, rank,
            # cfg, obs_config, rank, # TODO: add obs config back when it's ready
            replay_buffer, tasks, cfg.maniskill3.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.maniskill3.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = peract_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'PERACT_RL':
        raise NotImplementedError("Only PERACT_BC is supported for maniskill yet")

    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)


    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    logdir = os.path.join(cwd, 'seed%d' % seed)

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size)

    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()