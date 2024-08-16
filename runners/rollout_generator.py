from multiprocessing import Value

import numpy as np
import torch
import copy
from mani_skill.envs.sapien_env import BaseEnv
from agents.agent import Agent
from helpers.transition import ReplayTransition
from helpers.ms3_utils import add_low_dim_states, extract_obs

class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: BaseEnv, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, lang_goal_tokens: list[str], eval_demo_seed: int = 0, reset_kwargs: dict = None):
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
        obs["lang_goal_tokens"] = lang_goal_tokens
        obs = add_low_dim_states(obs)
        agent.reset()
        # obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        obs_history = {k: [v] * timesteps for k, v in obs.items()}
        
        # start episode generation
        for step in range(episode_length):

            prepped_data = {k: v[-1] for k, v in obs_history.items()} # use the latest obs as input
            # prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)

            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            agent_obs_elems["lang_goal_tokens"] = lang_goal_tokens
            # agent_obs_elems = add_low_dim_states(agent_obs_elems)
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            # TODO: convert act results to a seq of poses using mplib, and then plan the traj
            rot_resolution = agent._rotation_resolution
            act_joint_pos = act_result.convert_to_joint_pos(obs["agent"]["qpos"], rot_resolution)
            for i in range(len(act_joint_pos)):
                action = act_joint_pos[i]
                if ((i > 0) and not (terminated or truncated)):
                    obs, reward, terminated, truncated, info = env.step(action)
                    # obs = extract_obs(obs)
            transition = {"observation": obs, "info": info, "reward": reward, "terminal": terminated}
            # obs_tp1 = copy.deepcopy(obs) 
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not terminated
                if timeout:
                    transition["terminal"] = True # task not finishing at the final timestep
                    if "needs_reset" in transition.info:
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
                transition["terminal"], timeout, 
                # summaries=transition.summaries,
                info=transition["info"])

            if transition["terminal"] or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                obs_tp1 = copy.deepcopy(obs) 
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
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

            if transition.info.get("needs_reset", transition["terminal"]):
                return
