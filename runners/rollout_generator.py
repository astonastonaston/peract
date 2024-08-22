from multiprocessing import Value

import numpy as np
import torch
import copy
from mani_skill.envs.sapien_env import BaseEnv
from agents.agent import Agent
from helpers.transition import ReplayTransition
from helpers.ms3_utils import add_low_dim_states, extract_obs

from clip import tokenize

class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: BaseEnv, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, lang_goal: list[str], eval_demo_seed: int = 0, reset_kwargs: dict = None):
                #   record_enabled: bool = False):

        if eval:
            obs, _ = env.reset(**reset_kwargs)
        else:
            obs, _ = env.reset()
        # reset env and agent 
        # print(obs)
        # print(type(obs))
        # print(obs.keys())
        # print(type(obs["agent"]))
        # print(obs["agent"])
        # print(obs["agent"].keys())
        # obs = extract_obs(obs)
        lang_goal_tokens = tokenize([lang_goal[0]])[0]  # assume only one desc for each task only
        obs["lang_goal_tokens"] = lang_goal_tokens # all data arrays in obs should be torch.Tensor
        obs = add_low_dim_states(obs, 0, episode_length)
        obs = extract_obs(obs) # flatten obs to 2 levels of dicts only for easier
        agent.reset()
        # obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        obs_history = {k: [v] * timesteps for k, v in obs.items()}
        urdf_path = env.agent.urdf_path
        time_step = env.unwrapped.control_timestep
        
        # start episode generation
        for step in range(episode_length):

            prepped_data = {k: v[-1] for k, v in obs_history.items()} # use the latest obs as input
            # prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)
            print(f"resutl action {act_result.action}")
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            agent_obs_elems["lang_goal_tokens"] = lang_goal_tokens
            # agent_obs_elems = add_low_dim_states(agent_obs_elems)
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            # TODO: convert act results to a seq of poses using mplib, and then plan the traj
            # rotation_resolution = agent.get_rotation_resolution()
            act_joint_pos = act_result.convert_to_joint_pos(obs["qpos"], urdf_path, time_step)
            for i in range(len(act_joint_pos)):
                action = act_joint_pos[i]
                if (i == 0):
                    obs, reward, terminated, truncated, info = env.step(action)
                elif (not (terminated or truncated)):
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(i, terminated, truncated, info)
                    # TODO: add env rendering to see what happens
                    # obs["lang_goal_tokens"] = lang_goal_tokens # all data arrays in obs should be torch.Tensor
            obs["lang_goal_tokens"] = lang_goal_tokens # all data arrays in obs should be torch.Tensor
            obs = add_low_dim_states(obs, step, episode_length)
            obs = extract_obs(obs)
            transition = {"observation": obs, 
                          "info": info, 
                          "reward": reward, 
                          "terminal": terminated or truncated, 
                          "truncated": truncated and not terminated, 
                          "terminated": terminated}
            # obs_tp1 = copy.deepcopy(obs) 
            timeout = truncated # TODO: timeout should be determined w.r.t. the episode length under pose-based control, instead of joint-pos-based control

            if transition["terminal"]: # Reset when terminated
                if "needs_reset" in transition["info"]:
                    print("Reset needed! transition info keys:")
                    print(transition["info"])
                    transition["info"]["needs_reset"] = True


            # if step == episode_length - 1: # Pose-control episode length reaching limit: reset
            #     # If last transition, and not terminal, then we timed out
            #     timeout = not terminated
            #     if timeout:
            #         transition["terminal"] = True # task not finishing at the final timestep. Like truncated
            #         if "needs_reset" in transition["info"]:
            #             print("Reset needed! transition info keys:")
            #             print(transition["info"])
            #             transition["info"]["needs_reset"] = True

            # TODO: add truncated and succeed terminal states

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            print(f"History keys {obs_history.keys()}")
            print(f"Transition obs {transition['observation'].keys()}")
            for k in obs_history.keys():
                obs_history[k].append(transition["observation"][k])
                obs_history[k].pop(0)

            transition["info"]["active_task_id"] = env.unwrapped.spec.id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition["reward"],
                transition["terminal"], transition["terminated"], transition["truncated"], timeout, 
                # summaries=transition.summaries,
                info=transition["info"])

            if transition["terminal"]:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                obs_tp1 = copy.deepcopy(obs) 
                if len(act_result.observation_elements) > 0:
                    # prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    prepped_data = {k: v[-1] for k, v in obs_history.items()} # use the latest obs as input
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            # TODO: enable recording
            # if record_enabled and transition.terminal or timeout or step == episode_length - 1:
            #     env.env._action_mode.arm_action_mode.record_end(env.env._scene,
            #                                                     steps=60, step_scene=True)

            obs = transition["observation"]
            yield replay_transition

            if transition["info"].get("needs_reset", transition["terminal"]) or terminated: # truncated or terminated
                return
