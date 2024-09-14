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
from runners.rollout_generator import RolloutGenerator
from runners.stat_accumulator import StatAccumulator, SimpleAccumulator
from runners.log_writer import LogWriter
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
                #  train_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer]],
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
                #  eval_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer], None] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 rollout_generator: RolloutGenerator = None,
                 weightsdir: str = None,
                 logdir: str = None,
                 max_fails: int = 10,
                 num_eval_runs: int = 1,
                 env_device: torch.device = None,
                 multi_task: bool = False, 
                 json_path: str = None):
            # super().__init__(train_env, agent, train_replay_buffer, num_train_envs, num_eval_envs,
            #                 rollout_episodes, eval_episodes, training_iterations, eval_from_eps_number,
            #                 episode_length, eval_env, eval_replay_buffer,
            #                 rollout_generator, weightsdir, logdir, max_fails, num_eval_runs,
            #                 env_device, multi_task)
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
            self._json_path = json_path
            self._demo_meta_data = load_json(json_path)
            manager = Manager()
            self.write_lock = manager.Lock()
            self.stored_transitions = manager.list()
            self.agent_summaries = manager.list()

    def _get_task_name(self): # TODO: support multi-task evals
        # if hasattr(self._eval_env, '_task_class'):
        #     eval_task_name = change_case(self._eval_env._task_class.__name__)
        #     multi_task = False
        # elif hasattr(self._eval_env, '_task_classes'):
        #     if self._eval_env.active_task_id != -1:
        #         task_id = (self._eval_env.active_task_id) % len(self._eval_env._task_classes)
        #         eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
        #     else:
        #         eval_task_name = ''
        #     multi_task = True
        # else:
        #     raise Exception('Neither task_class nor task_classes found in eval env')
        eval_task_name = self._eval_env.unwrapped.spec.id
        multi_task = self._multi_task # Multi-task eval not supported yet
        return eval_task_name, multi_task
    
    def summaries(self) -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop())
        for key, value in self._new_transitions.items():
            summaries.append(ScalarSummary('%s/new_transitions' % key, value))
        for key, value in self._total_transitions.items():
            summaries.append(ScalarSummary('%s/total_transitions' % key, value))
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        summaries.extend(self.agent_summaries)

        # TODO: add current task_name to eval summaries .... argh this should be inside a helper function
        # TODO: Multi-task is not supported yet
        eval_task_name = self._eval_env.unwrapped.spec.id
        # if hasattr(self._eval_env, '_task_class'):
        # elif hasattr(self._eval_env, '_task_classes'): # TODO: multi-task summary not supported yet
        #     if self._current_task_id != -1:
        #         task_id = (self._current_task_id) % len(self._eval_env._task_classes)
        #         eval_task_name = self._eval_env.unwrapped.spec.id
        #     else:
        #         eval_task_name = ''
        # else:
        #     raise Exception('Neither task_class nor task_classes found in eval env')

        # multi-task summaries
        if eval_task_name and self._multi_task:
            for s in summaries:
                if 'eval' in s.name:
                    s.name = '%s/%s' % (s.name, eval_task_name)

        return summaries

    def _run_eval_independent(self, name: str,
                            stats_accumulator,
                            eval_env,
                            weight,
                            writer_lock,
                            eval=True,
                            device_idx=0,
                            save_metrics=True,
                            cinematic_recorder_cfg=None):

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

        # initialize cinematic recorder if specified
        # TODO: support cinematic recorder
        # rec_cfg = cinematic_recorder_cfg
        # if rec_cfg.enabled:
        #     cam_placeholder = Dummy('cam_cinematic_placeholder')
        #     cam = VisionSensor.create(rec_cfg.camera_resolution)
        #     cam.set_pose(cam_placeholder.get_pose())
        #     cam.set_parent(cam_placeholder)

        #     cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), rec_cfg.rotate_speed)
        #     tr = TaskRecorder(env, cam_motion, fps=rec_cfg.fps)

        #     env.env._action_mode.arm_action_mode.set_callable_each_step(tr.take_snap)

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')

        # to save or not to save evaluation metrics (set as False for recording videos)
        if self._save_metrics:
            csv_file = 'eval_data.csv' if not self._is_test_set else 'test_data.csv'
            # TODO: add log writting
            writer = LogWriter(self._logdir, True, True,
                            env_csv=csv_file)
            # pass

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self._weightsdir, str(weight))
            seed_path = self._weightsdir.replace('/weights', '')
            self._agent.load_weights(weight_path)
            weight_name = str(weight)

        new_transitions = {'train_envs': 0, 'eval_envs': 0}
        total_transitions = {'train_envs': 0, 'eval_envs': 0}
        current_task_id = -1

        for n_eval in range(self._num_eval_runs):
            # if rec_cfg.enabled:
            #     tr._cam_motion.save_pose()

            # best weight for each task (used for test evaluation)
            if type(weight) == dict:
                task_name = list(weight.keys())[n_eval]
                task_weight = weight[task_name]
                weight_path = os.path.join(self._weightsdir, str(task_weight))
                seed_path = self._weightsdir.replace('/weights', '')
                self._agent.load_weights(weight_path)
                weight_name = str(task_weight)
                print('Evaluating weight %s for %s' % (weight_name, task_name))

            # evaluate on N tasks * M episodes per task = total eval episodes
            reward_list = []
            success_list = []
            for ep in range(self._eval_episodes):
                print(f"episode num {ep} under total {self._eval_episodes} episodes")
                eval_demo_seed = ep + self._eval_from_eps_number
                logging.info('%s: Starting episode %d, seed %d.' % (name, ep, eval_demo_seed))

                # the current task gets reset after every M episodes
                episode_rollout = []

                # get reset status for the current episode
                episodes = self._demo_meta_data["episodes"]
                episode = episodes[ep]

                reset_kwargs = episode["reset_kwargs"].copy()
                reset_kwargs["seed"] = eval_demo_seed # demo reset seed, which is also the episode number

                # TODO: modify this to keeping stepping till one episode finishes
                generator = self._rollout_generator.generator(
                    self._step_signal, env, self._agent,
                    self._episode_length, self._timesteps,
                    eval, self._lang_goal, eval_demo_seed=eval_demo_seed, reset_kwargs=reset_kwargs)
                    # TODO: enable recording
                    # record_enabled=rec_cfg.enabled)
                    
                for replay_transition in generator:
                    # print("summary of replay tran")
                    # print(replay_transition.summaries)
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

                with self.write_lock:
                    for transition in episode_rollout:
                        self.stored_transitions.append((name, transition, eval))

                        new_transitions['eval_envs'] += 1
                        total_transitions['eval_envs'] += 1
                        # print(f"Transition reward is {transition.reward}")
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
                    print("Final state info:")
                    print(episode_rollout[-1].observation.keys())
                    print(episode_rollout[-1].action.shape)
                    print(episode_rollout[-1].terminal)
                    print(episode_rollout[-1].terminated)
                    print(episode_rollout[-1].truncated)
                else:
                    print(f"Evaluating {task_name} | Episode {ep} | Score: {0} () | Lang Goal: {lang_goal} | Success: {0}")
                # # TODO: save recording at maniskill
                # if rec_cfg.enabled:
                #     success = reward > 0.99
                #     record_file = os.path.join(seed_path, 'videos',
                #                             '%s_w%s_s%s_%s.mp4' % (task_name,
                #                                                     weight_name,
                #                                                     eval_demo_seed,
                #                                                     'succ' if success else 'fail'))

                #     lang_goal = self._eval_env._lang_goal

                #     tr.save(record_file, lang_goal, reward)
                #     tr._cam_motion.restore_pose()

            # report summaries
            summaries = []
            summaries.extend(stats_accumulator.pop())

            eval_task_name, multi_task = self._get_task_name()

            if eval_task_name and multi_task: # Do multi-task summary
                for s in summaries:
                    if 'eval' in s.name:
                        s.name = '%s/%s' % (s.name, eval_task_name)

            # print("summaries")
            # print(summaries)
            if len(summaries) > 0:
                if multi_task:
                    task_score = [s.value for s in summaries if f'eval_envs/return/{eval_task_name}' in s.name][0]
                else:
                    task_score = [s.value for s in summaries if f'eval_envs/return' in s.name][0]
            else:
                task_score = "unknown"

            # TODO: Do eval summary reports
            print(f"Finished {eval_task_name} | Final Score: {task_score} | Final Success Rate {np.mean(success_list)}\n")
            # print(f"Finished {task_name} | Final Score: {np.mean(reward_list)} | Final Success Rate {np.mean(success_list)}\n")

            if self._save_metrics:
                # TODO: Do summary savings
                with writer_lock:
                    writer.add_summaries(weight_name, summaries)
                    # pass

            self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
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
              device_idx,
              save_metrics,
              cinematic_recorder_cfg):
        multi_task = isinstance(env_config[0], list)

        # env_config = (tasks,
        #               control_mode,
        #               eval_cfg.maniskill3.traj_path,
        #               eval_cfg.maniskill3.json_path,
        #               eval_cfg.maniskill3.desc_pkl_path,
        #               eval_cfg.maniskill3.episode_length,
        #               eval_cfg.framework.eval_episodes,
        #               train_cfg.maniskill3.include_lang_goal_in_obs,
        #               eval_cfg.maniskill3.time_in_state,
        #               eval_cfg.framework.record_every_n)
        env_kwargs = {'control_mode': env_config[1], 
                      "obs_mode": "pointcloud",
                      "num_envs": self._eval_envs,
                      "max_episode_steps": 1000}
        # env_kwargs = {'control_mode': "pd_joint_pos", 
        #               "obs_mode": "pointcloud",
        #               "num_envs": self._eval_envs}
        print(f"cuda status {torch.cuda.is_available(), sapien.Device('cuda')}")
        if multi_task:
            # TODO: support multi-task env eval
            # eval_env = CustomMultiTaskRLBenchEnv(
            #     task_classes=env_config[0],
            #     observation_config=env_config[1],
            #     action_mode=env_config[2],
            #     dataset_root=env_config[3],
            #     episode_length=env_config[4],
            #     headless=env_config[5],
            #     swap_task_every=env_config[6],
            #     include_lang_goal_in_obs=env_config[7],
            #     time_in_state=env_config[8],
            #     record_every_n=env_config[9])
            # eval_env = gym.make()
            raise NotImplementedError("Multi-task evaluation not supported yet")
        else:
            eval_env = gym.make(env_config[0], **env_kwargs)

        # self._internal_env_runner = _IndependentEnvRunner(
        #     self._train_env, eval_env, self._agent, self._timesteps, self._train_envs,
        #     self._eval_envs, self._rollout_episodes, self._eval_episodes,
        #     self._training_iterations, self._eval_from_eps_number, self._episode_length, self._kill_signal,
        #     self._step_signal, self._num_eval_episodes_signal,
        #     self._eval_epochs_signal, self._eval_report_signal,
        #     self.log_freq, self._rollout_generator, None,
        #     self.current_replay_ratio, self.target_replay_ratio,
        #     self._weightsdir, self._logdir,
        #     self._env_device, self._previous_loaded_weight_folder,
        #     num_eval_runs=self._num_eval_runs)
        self._eval_env = eval_env
        self._lang_goal = env_config[2]

        self._run_eval_independent('eval_env',
                                    self._stat_accumulator,
                                    eval_env,
                                    weight,
                                    writer_lock,
                                    True,
                                    device_idx,
                                    save_metrics,
                                    cinematic_recorder_cfg)