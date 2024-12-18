import numpy as np
import torch
import logging
import os
import copy
import time
from typing import List
from typing import Union
from multiprocessing import Value, Process, Manager

from agents.agent import Agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.record import RecordEpisode
from runners.rollout_generator import RolloutGenerator
from runners.stat_accumulator import StatAccumulator, SimpleAccumulator
from runners.log_writer import LogWriter
from torchvision import transforms
from agents.agent import Summary, ScalarSummary
# from yarr.replay_buffer.replay_buffer import ReplayBuffer
# from helpers.custom_ms_env import CustomManiskillEnv
# from agents.agent import Summary
# from runners.env_runner import EnvRunner

import sapien
import gymnasium as gym

class IndependentEnvRunner(object):

    def __init__(self,
                 train_env: BaseEnv,
                 agent: Agent,
                 train_replay_buffer: None,
                 num_train_envs: int,
                 num_eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 eval_env: Union[BaseEnv, None] = None,
                 eval_replay_buffer: Union[None] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 rollout_generator: RolloutGenerator = None,
                 weightsdir: str = None,
                 logdir: str = None,
                 max_fails: int = 10,
                 num_eval_runs: int = 1,
                 env_device: torch.device = None,
                 multi_task: bool = False, 
                 json_path: str = None,
                 eval_save_voxel_images: bool = False):
            self._train_env = train_env
            self._eval_env = eval_env if eval_env else train_env
            self._agent = agent
            self._train_envs = num_train_envs
            self._eval_envs = num_eval_envs
            self._train_replay_buffer = train_replay_buffer if isinstance(train_replay_buffer, list) else [train_replay_buffer]
            self._timesteps = self._train_replay_buffer[0].timesteps if self._train_replay_buffer[0] is not None else 1

            if eval_replay_buffer is not None:
                eval_replay_buffer = eval_replay_buffer if isinstance(eval_replay_buffer, list) else [eval_replay_buffer]
            self._eval_replay_buffer = eval_replay_buffer
            self._rollout_episodes = rollout_episodes
            self._eval_episodes = eval_episodes
            self._num_eval_runs = num_eval_runs
            self._training_iterations = training_iterations
            self._eval_from_eps_number = eval_from_eps_number
            self._episode_length = episode_length
            self._stat_accumulator = stat_accumulator
            self._rollout_generator = (
                RolloutGenerator() if rollout_generator is None
                else rollout_generator)
            self._rollout_generator._env_device = env_device
            self._weightsdir = weightsdir
            self._logdir = logdir
            self._max_fails = max_fails
            self._env_device = env_device
            self._previous_loaded_weight_folder = ''
            self._p = None
            self._kill_signal = Value('b', 0)
            self._step_signal = Value('i', -1)
            self._num_eval_episodes_signal = Value('i', 0)
            self._eval_epochs_signal = Value('i', 0)
            self._eval_report_signal = Value('b', 0)
            self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
            self._total_transitions = {'train_envs': 0, 'eval_envs': 0}
            self.log_freq = 1000  # Will get overridden later
            self.target_replay_ratio = None  # Will get overridden later
            self.current_replay_ratio = Value('f', -1)
            self._current_task_id = -1
            self._multi_task = multi_task
            self._eval_save_voxel_images = eval_save_voxel_images
            self._json_path = json_path
            self._demo_meta_data = load_json(json_path)
            manager = Manager()
            self.write_lock = manager.Lock()
            self.stored_transitions = manager.list()
            self.agent_summaries = manager.list()

    def _get_task_name(self): # TODO: support multi-task evals
        eval_task_name = self._eval_env.unwrapped.spec.id
        multi_task = self._multi_task # Multi-task eval not supported yet
        return eval_task_name, multi_task
    
    def summaries(self) -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop()) # add task statistics
        summaries.extend(self.agent_summaries) # add agent summaries (like images)
        return summaries

    def _run_eval_independent(self, name: str,
                            stats_accumulator,
                            weight,
                            writer_lock,
                            eval_cfg,
                            train_cfg,
                            env_config, # those configs are for logging purposes only
                            train_config,
                            eval=True,
                            device_idx=0,
                            cinematic_recorder_cfg=None,
                            ):
        save_metrics = eval_cfg.framework.eval_save_metrics
        vis_pose = eval_cfg.maniskill3.vis_pose
        gripper_open_delta = train_cfg.replay.gripper_open_delta

        self._name = name
        self._save_metrics = save_metrics
        self._is_test_set = type(weight) == dict

        self._agent = copy.deepcopy(self._agent)

        device = torch.device('cuda:%d' % device_idx) if torch.cuda.device_count() > 1 else torch.device('cuda:0')
        with writer_lock: # hack to prevent multiple CLIP downloads ... argh should use a separate lock
            self._agent.build(training=False, device=device)
        print(f"agent build complete. Device {device}")

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')
     
        # to save or not to save evaluation metrics (set as False for recording videos)
        if self._save_metrics:
            csv_file = 'eval_data.csv' if not self._is_test_set else 'test_data.csv'
            # add train and eval configs to wandb if used
            use_wandb = eval_cfg.wandb.use     
            if use_wandb:
                writer = LogWriter(self._logdir, True, True, True, 
                                project_name=eval_cfg.wandb.project_name, 
                                exp_name=eval_cfg.wandb.exp_name, 
                                env_csv=csv_file) 
                writer.add_wandb_config(train_config, env_config, evaluation=True)
            else:
                writer = LogWriter(self._logdir, True, True, False,
                                env_csv=csv_file)


        # one weight for all tasks (used for validation). For now, only single-task evaluation is supported
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            print('Evaluating weight %s' % weight)
            weight_path = os.path.join(self._weightsdir, str(weight))
            seed_path = self._weightsdir.replace('/weights', '')
            self._agent.load_weights(weight_path)
            weight_name = str(weight)
            if self._eval_save_voxel_images:
                img_log_dir_name = f"{weight_name}_images"
                img_log_dir = os.path.join(seed_path, img_log_dir_name)
                if not os.path.exists(img_log_dir):
                    os.makedirs(img_log_dir)

        for n_eval in range(self._num_eval_runs):
            # best weight for each task (used for test evaluation)
            if type(weight) == dict:
                task_name = list(weight.keys())[n_eval]
                task_weight = weight[task_name]
                weight_path = os.path.join(self._weightsdir, str(task_weight))
                seed_path = self._weightsdir.replace('/weights', '')
                self._agent.load_weights(weight_path)
                weight_name = str(task_weight)
                if self._eval_save_voxel_images:
                    img_log_dir_name = f"{weight_name}_{task_name}_images"
                    img_log_dir = os.path.join(self._logdir, img_log_dir_name)
                    if not os.path.exists(img_log_dir):
                        os.makedirs(img_log_dir)
                print('Evaluating weight %s for %s' % (weight_name, task_name))

            # evaluate on N tasks * M episodes per task = total eval episodes
            reward_list = []
            success_list = []
            print(f"Evaluating from episode {self._eval_from_eps_number}. In total {self._eval_episodes} episodes to evaluate")
            for ep in range(self._eval_episodes):
                eval_demo_seed = ep + self._eval_from_eps_number
                logging.info('%s: Starting episode %d, seed %d.' % (name, ep, eval_demo_seed))
                
                # if needed, create dir to log voxel images
                if self._eval_save_voxel_images:
                    ep_img_log_dir_name = f"episode_{eval_demo_seed}"
                    ep_img_log_dir = os.path.join(img_log_dir, ep_img_log_dir_name)
                    if not os.path.exists(ep_img_log_dir):
                        os.makedirs(ep_img_log_dir)

                # the current task gets reset after every M episodes
                episode_rollout = []

                # get reset status for the current episode
                reset_kwargs = {"seed": eval_demo_seed} # demo reset seed, which is also the episode number

                # Keeping stepping until the episode finishes
                generator = self._rollout_generator.generator(
                    self._step_signal, env, self._agent,
                    self._episode_length, self._timesteps,
                    eval, self._lang_goal, eval_demo_seed=eval_demo_seed, 
                    reset_kwargs=reset_kwargs, vis_pose=vis_pose,
                    gripper_open_delta=gripper_open_delta)
                
                step_cnt = 0
                for replay_transition in generator:
                    if self._eval_save_voxel_images:
                        ep_img_log_path = os.path.join(ep_img_log_dir, f'episode_{eval_demo_seed}_pose_step_{step_cnt}_image.png')
                        img = replay_transition.observation['voxel_grid_img_0'] # peract only uses depth=0 so here's a hack
                        img = img.transpose(1, 2, 0)
                        to_pil = transforms.ToPILImage()
                        img = to_pil(img)
                        img.save(ep_img_log_path)
                    
                    while True:
                        if self._kill_signal.value:
                            # env.shutdown()
                            return
                        if (eval or self._target_replay_ratio is None or
                                self._step_signal.value <= 0 or (
                                        self._current_replay_ratio.value >
                                        self._target_replay_ratio) or
                                        replay_transition.info["success"]):
                            break
                        time.sleep(1)
                        logging.debug(
                            'Agent. Waiting for replay_ratio %f to be more than %f' %
                            (self._current_replay_ratio.value, self._target_replay_ratio))

                    with self.write_lock:
                        if len(self.agent_summaries) == 0:
                            # Only store new summaries if the previous ones
                            # have been popped by the main env runner.
                            for s in self._agent.act_summaries():
                                self.agent_summaries.append(s)
                    episode_rollout.append(replay_transition)
                    step_cnt += 1

                with self.write_lock:
                    for transition in episode_rollout:
                        self.stored_transitions.append((name, transition, eval))
                        stats_accumulator.step(transition, eval)
                        current_task_id = transition.info['active_task_id']

                self._num_eval_episodes_signal.value += 1

                task_name = env.unwrapped.spec.id
                lang_goal = self._lang_goal
                if episode_rollout != []:
                    reward = episode_rollout[-1].reward
                    reward_list.append(reward)
                    success = episode_rollout[-1].info["success"]
                    success_list.append(success)
                    print(f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Lang Goal: {lang_goal} | Success: {success}")
                else:
                    print(f"Evaluating {task_name} | Episode {ep} | Score: {0} () | Lang Goal: {lang_goal} | Success: {0}")

            # reset at last to save the video for the last episode
            if cinematic_recorder_cfg.enabled:
                _, _ = env.reset(**reset_kwargs)

            summaries = self.summaries() # log stat accumulator summary + agent summary

            # add scalar success rate summary
            mean_success_rate = np.mean(success_list)
            eval_prefix = stats_accumulator.get_prefix()
            summaries.append(ScalarSummary('%s/success_rate' % eval_prefix, 
                                           mean_success_rate)) 

            # convert to multi-task summaries (unused for single-task yet)
            eval_task_name, multi_task = self._get_task_name()
            if len(summaries) > 0:
                if multi_task:
                    task_score = [s.value for s in summaries if f'eval_envs/return/{eval_task_name}' in s.name][0]
                else:
                    task_score = [s.value for s in summaries if f'eval_envs/return' in s.name][0]
            else:
                task_score = "unknown"

            print(f"Finished {eval_task_name} | Final Score: {task_score} | Final Success Rate {mean_success_rate}\n")

            if self._save_metrics:
                with writer_lock:
                    writer.add_summaries(weight_name, summaries)
                    # pass

            self.agent_summaries[:] = []
            self.stored_transitions[:] = []

        if self._save_metrics:
            with writer_lock:
                writer.end_iteration()
                # pass

        logging.info('Finished evaluation.')
        # env.shutdown()

    # serialized evaluator for individual tasks
    def start(self, weight,
              save_load_lock, writer_lock,
              env_config,
              train_config, # those configs are for logging purposes only
              device_idx,
              eval_cfg,
              train_cfg
              ):
        cinematic_recorder_cfg = eval_cfg.cinematic_recorder
        
        multi_task = isinstance(env_config["tasks"], list)

        env_kwargs = {'control_mode': env_config["control_mode"], 
                      "obs_mode": "pointcloud",
                      "num_envs": self._eval_envs,
                      "max_episode_steps": 1000}
        if cinematic_recorder_cfg.enabled:
            env_kwargs["render_mode"] = "rgb_array"
            
        print(f"cuda status {torch.cuda.is_available(), sapien.Device('cuda')}")
        if multi_task:
            raise NotImplementedError("Multi-task evaluation not supported yet")
        else:
            eval_env = gym.make(env_config["tasks"], **env_kwargs)
            if cinematic_recorder_cfg.enabled:
                eval_env = RecordEpisode(eval_env, output_dir=cinematic_recorder_cfg.save_path, save_trajectory=True, trajectory_name="trajectory", save_video=True, video_fps=30)

        self._eval_env = eval_env
        self._lang_goal = env_config["lang_goal"]

        self._run_eval_independent('eval_env',
                                    self._stat_accumulator,
                                    weight,
                                    writer_lock,
                                    eval_cfg,
                                    train_cfg,
                                    env_config,
                                    train_config,
                                    True,
                                    device_idx,
                                    cinematic_recorder_cfg)