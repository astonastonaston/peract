# Adapted from https://github.com/MohitShridhar/YARR/blob/peract/yarr/utils/rollout_generator.py
from multiprocessing import Value

import numpy as np
import torch
import time
import copy
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from agents.agent import Agent, VideoSummary, TextSummary
from helpers.transition import ReplayTransition
from helpers.ms3_utils import add_low_dim_states, reduce_obs
from runners.motion_planner import PandaArmMotionPlanningSolver
from clip import tokenize

class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: BaseEnv, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, lang_goal: list[str], eval_demo_seed: int = 0, 
                  reset_kwargs: dict = None, vis_pose=False,
                  gripper_open_delta: float = 1e-3):
                #   record_enabled: bool = False):

        # reset env and agent 
        if eval:
            obs, _ = env.reset(**reset_kwargs)
        else:
            obs, _ = env.reset()

        # move to cube back to simplify the task
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            # vis=False, # visualization of next pose mode
            vis=vis_pose, # visualization of next pose mode
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=True,
            print_env_info=False,
        )

        tokens = tokenize(lang_goal[0]).numpy()
        token_tensor = torch.from_numpy(tokens).to("cuda")
        lang_goal_tokens = token_tensor  # assume only one desc for each task only
        obs["lang_goal_tokens"] = token_tensor # all data arrays in obs should be torch.Tensor
        obs = add_low_dim_states(obs, 0, episode_length, gripper_open_delta)
        obs = reduce_obs(obs) # flatten obs to 2 levels of dicts only for easier

        agent.reset()
        obs_history = {k: [v] * timesteps for k, v in obs.items()} # timestep = 1 or so
        urdf_path = env.agent.urdf_path
        time_step = env.unwrapped.control_timestep
        
        # start episode generation (episode_length: the number of pose-based control steps)
        for step in range(episode_length):
            prepped_data = {k: v[-1] for k, v in obs_history.items()} # use the latest obs as input

            act_result = agent.act(step, prepped_data,
                                   deterministic=eval)
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            agent_obs_elems["lang_goal_tokens"] = lang_goal_tokens
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}
            
            # plan the path to the target pose
            trans_coordinate, rot, gripper_open, _ = act_result.action[:3], act_result.action[3:7], act_result.action[7], act_result.action[8]
            planner = PandaArmMotionPlanningSolver(
                env,
                debug=False,
                # vis=False, # visualization of next pose mode
                vis=vis_pose, # visualization of next pose mode
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=True,
                print_env_info=False,
            )

            # do gripper action
            if gripper_open:
                planner.open_gripper()
            else:
                planner.close_gripper()

            # set and reach the target pose
            reach_pose = sapien.Pose(p=trans_coordinate, q=rot)
            obs, reward, terminated, truncated, info = planner.move_to_pose_with_screw(reach_pose)
            if step == episode_length - 1: # manually truncate if max pose-based control episodic steps is reached
                truncated = True
            if info["plan_failed"]: # if planning failed, truncate this episode
                print("Planning failed! Restarting another episode")
                truncated = True
                
            obs["lang_goal_tokens"] = lang_goal_tokens # all data arrays in obs should be torch.Tensor
            obs = add_low_dim_states(obs, step+1, episode_length, gripper_open_delta)
            obs = reduce_obs(obs)
            transition = {"observation": obs, 
                          "info": info, 
                          "reward": reward, 
                          "terminal": terminated or truncated, 
                          "truncated": truncated and not terminated, 
                          "terminated": terminated}
            timeout = truncated

            # Reset when terminated
            if transition["terminal"]: 
                if "needs_reset" in transition["info"]:
                    transition["info"]["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition["observation"][k])
                obs_history[k].pop(0)

            transition["info"]["active_task_id"] = env.unwrapped.spec.id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition["reward"],
                transition["terminal"], transition["terminated"], transition["truncated"], timeout, 
                info=transition["info"])

            if transition["terminal"]:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                obs_tp1 = copy.deepcopy(obs) 
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: v[-1] for k, v in obs_history.items()} # use the latest obs as input
                    act_result = agent.act(step, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            obs = transition["observation"]
            yield replay_transition

            if transition["info"].get("needs_reset", transition["terminal"]) or terminated: # truncated or terminated
                return
