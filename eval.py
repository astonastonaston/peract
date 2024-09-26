import gc
import logging
import os
import sys
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from typing import List

import hydra
import numpy as np
import torch
import sapien
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig
# from rlbench.action_modes.action_mode import MoveArmThenGripper
# from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
# from rlbench.action_modes.gripper_action_modes import Discrete
# from rlbench.backend.utils import task_file_to_task_class
from runners.ms_env_runner import IndependentEnvRunner
from runners.log_writer import LogWriter
from runners.stat_accumulator import SimpleAccumulator

from agents import peract_bc
# from agents import c2farm_lingunet_bc
# from agents import arm
# from agents.baselines import bc_lang, vit_bc_lang

# from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
from helpers import utils

from runners.rollout_generator import RolloutGenerator
from torch.multiprocessing import Process, Manager

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from tasks import push_cube


def eval_seed(train_cfg,
              eval_cfg,
              logdir,
              cams,
              env_device,
              multi_task,
              seed,
              env_config) -> None:

    tasks = eval_cfg.maniskill3.tasks
    rg = RolloutGenerator()

    # print(train_cfg)
    if train_cfg.method.name == 'ARM':
        raise NotImplementedError('ARM not yet supported for eval.py')

    elif train_cfg.method.name == 'BC_LANG':
        raise NotImplementedError("Only PERACT_BC is supported for maniskill yet")
        
    elif train_cfg.method.name == 'VIT_BC_LANG':
        raise NotImplementedError("Only PERACT_BC is supported for maniskill yet")
        
    elif train_cfg.method.name == 'C2FARM_LINGUNET_BC':
        raise NotImplementedError("Only PERACT_BC is supported for maniskill yet")
        
    elif train_cfg.method.name == 'PERACT_BC':
        agent = peract_bc.launch_utils.create_agent(train_cfg)

    elif train_cfg.method.name == 'PERACT_RL':
        raise NotImplementedError("PERACT_RL not yet supported for eval.py")

    else:
        raise ValueError('Method %s does not exists.' % train_cfg.method.name)

    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(logdir, 'weights')
    # print(f"weight dir {weightsdir}")

    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent,
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=train_cfg.framework.training_iterations,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.maniskill3.episode_length, # max episode length
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=rg,
        num_eval_runs=len(tasks),
        multi_task=multi_task,
        json_path=eval_cfg.maniskill3.json_path)

    manager = Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()

    # evaluate all checkpoints (0, 1000, ...) which don't have results, i.e. validation phase
    if eval_cfg.framework.eval_type == 'missing':
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))

        env_data_csv_file = os.path.join(logdir, 'eval_data.csv')
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            evaluated_weights = sorted(map(int, list(env_dict['step'].values())))
            weight_folders = [w for w in weight_folders if w not in evaluated_weights]

        print('Missing weights: ', weight_folders)

    # pick the best checkpoint from validation and evaluate, i.e. test phase
    elif eval_cfg.framework.eval_type == 'best':
        env_data_csv_file = os.path.join(logdir, 'eval_data.csv')
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            existing_weights = list(map(int, sorted(os.listdir(os.path.join(logdir, 'weights')))))
            task_weights = {}
            for task in tasks:
                weights = list(env_dict['step'].values())

                if len(tasks) > 1:
                    task_score = list(env_dict['eval_envs/return/%s' % task].values())
                else:
                    task_score = list(env_dict['eval_envs/return'].values())

                avail_weights, avail_task_scores = [], []
                for step_idx, step in enumerate(weights):
                    if step in existing_weights:
                        avail_weights.append(step)
                        avail_task_scores.append(task_score[step_idx])

                assert(len(avail_weights) == len(avail_task_scores))
                best_weight = avail_weights[np.argwhere(avail_task_scores == np.amax(avail_task_scores)).flatten().tolist()[-1]]
                task_weights[task] = best_weight

            weight_folders = [task_weights]
            print("Best weights:", weight_folders)
        else:
            raise Exception('No existing eval_data.csv file found in %s' % logdir)

    # evaluate only the last checkpoint
    elif eval_cfg.framework.eval_type == 'last':
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))
        weight_folders = [weight_folders[-1]]
        print("Last weight:", weight_folders)

    # evaluate a specific checkpoint
    elif type(eval_cfg.framework.eval_type) == int:
        weight_folders = [int(eval_cfg.framework.eval_type)]
        print("Weight:", weight_folders)

    else:
        raise Exception('Unknown eval type')

    num_weights_to_eval = np.arange(len(weight_folders))
    if len(num_weights_to_eval) == 0:
        logging.info("No weights to evaluate. Results are already available in eval_data.csv")
        sys.exit(0)

    # evaluate several checkpoints in parallel
    # NOTE: in multi-task settings, each task is evaluated serially, which makes everything slow!
    split_n = utils.split_list(num_weights_to_eval, eval_cfg.framework.eval_envs)
    # print(f"Num of splits {len(split_n)}")
    for split in split_n:
        processes = []
        print(f"Num of processes {len(split), sapien.Device('cuda')}")
        for e_idx, weight_idx in enumerate(split):
            weight = weight_folders[weight_idx]
            # TODO: the maniskill gym env is already parallalized, so don't need to rewrite torch multi-processing again
            p = Process(target=env_runner.start,
                        args=(weight,
                              save_load_lock,
                              writer_lock,
                              env_config,
                              e_idx % torch.cuda.device_count(),
                              eval_cfg.framework.eval_save_metrics,
                              eval_cfg.cinematic_recorder,
                              eval_cfg.maniskill3.vis_pose))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_name='eval', config_path='conf')
def main(eval_cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(eval_cfg.framework.logdir,
                                eval_cfg.maniskill3.task_name,
                                eval_cfg.method.name,
                                'seed%d' % start_seed)

    train_config_path = os.path.join(logdir, 'config.yaml')
    # print(f"loading {train_config_path, os.path.exists(train_config_path)}")
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            train_cfg = OmegaConf.load(f)
    else:
        raise Exception("Missing seed%d/config.yaml" % start_seed)
    
    print("getting the device")
    print(f"cuda available status: {torch.cuda.is_available(), sapien.Device('cuda')}")
    env_device = utils.get_device(eval_cfg.framework.gpu)
    logging.info('Using env device %s.' % str(env_device))

    # gripper_mode = Discrete()
    # arm_action_mode = EndEffectorPoseViaPlanning()
    control_mode = 'pd_joint_pos'

    # TODO: add task existance check for ms3
    # task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
    #               if t != '__init__.py' and t.endswith('.py')]
    eval_cfg.maniskill3.cameras = eval_cfg.maniskill3.cameras if isinstance(
        eval_cfg.maniskill3.cameras, ListConfig) else [eval_cfg.maniskill3.cameras]
    if os.path.exists(eval_cfg.maniskill3.desc_pkl_path):
        with open(eval_cfg.maniskill3.desc_pkl_path, "rb") as f:
            lang_goal_tokens = pickle.load(f) # TODO: only single-task lang goal supported yet
    else:
        raise Exception("Missing task desc file {}" % eval_cfg.maniskill3.desc_pkl_path)

    # single-task or multi-task
    if len(eval_cfg.maniskill3.tasks) > 1:
        tasks = eval_cfg.maniskill3.tasks
        multi_task = True

        for task in tasks:
            # TODO: add task existance check for ms3
            # if task not in task_files:
            #     raise ValueError('Task %s not recognised!.' % task)
            # task_classes.append(task_file_to_task_class(task))
            pass

        env_config = (tasks,
                      control_mode,
                      lang_goal_tokens,
                      eval_cfg.maniskill3.episode_length,
                      eval_cfg.framework.eval_episodes,
                      train_cfg.maniskill3.include_lang_goal_in_obs,
                      eval_cfg.maniskill3.time_in_state,
                      eval_cfg.framework.record_every_n)
    else:
        task = eval_cfg.maniskill3.tasks[0]
        multi_task = False

        # TODO: add task existance check for ms3
        # if task not in task_files:
        #     raise ValueError('Task %s not recognised!.' % task)

        env_config = (task,
                      control_mode,
                      lang_goal_tokens,
                      eval_cfg.maniskill3.episode_length,
                      eval_cfg.framework.eval_episodes,
                      train_cfg.maniskill3.include_lang_goal_in_obs,
                      eval_cfg.maniskill3.time_in_state,
                      eval_cfg.framework.record_every_n)

    logging.info('Evaluating seed %d.' % start_seed)
    eval_seed(train_cfg,
              eval_cfg,
              logdir,
              eval_cfg.maniskill3.cameras,
              env_device,
              multi_task, start_seed,
              env_config)

# if __name__ == '__main__':
#     main()
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn') # multiprocessing with cuda re-init
    main()