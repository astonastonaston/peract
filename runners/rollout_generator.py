from multiprocessing import Value

import numpy as np
import torch
import time
import copy
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from agents.agent import Agent, VideoSummary, TextSummary
from helpers.transition import ReplayTransition
from helpers.ms3_utils import add_low_dim_states, extract_obs
from runners.motion_planner import PandaArmMotionPlanningSolver

from clip import tokenize

class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: BaseEnv, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, lang_goal: list[str], eval_demo_seed: int = 0, reset_kwargs: dict = None, vis_pose=False):
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
        obs_history = {k: [v] * timesteps for k, v in obs.items()} # timestep = 1 or so
        urdf_path = env.agent.urdf_path
        time_step = env.unwrapped.control_timestep
        
        # start episode generation (episode_length: the number of pose-based control steps)
        for step in range(episode_length):
            print(f"step {step} in episode with len {episode_length}")
            prepped_data = {k: v[-1] for k, v in obs_history.items()} # use the latest obs as input
            # prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)
            # print(f"act obs kys {act_result.observation_elements.keys()}")
            print(f"result action {act_result.action}")
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            agent_obs_elems["lang_goal_tokens"] = lang_goal_tokens
            # agent_obs_elems = add_low_dim_states(agent_obs_elems)
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            # plan the path to the target pose
            trans_coordinate, rot, gripper_open, _ = act_result.action[:3], act_result.action[3:7], act_result.action[7], act_result.action[8]
            # rot = np.concatenate([[rot[3]], rot[:3]]) # convert to wxyz
            # trans_coordinate = env.agent.tcp.pose.sp.p + np.array([-0.01, 0, 0]) # debugging mode
            # rot = env.agent.tcp.pose.sp.q
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
                # break
            obs["lang_goal_tokens"] = lang_goal_tokens # all data arrays in obs should be torch.Tensor
            obs = add_low_dim_states(obs, step+1, episode_length)
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

            # # TODO: add video and error summaries
            # summaries = []
            # self._i += 1
            # success = transition["info"]["success"]
            # if ((transition["terminal"] or self._i == self._episode_length)):
            #         # and self._record_current_episode): # TODO: add video summary
            #     # self._append_final_frame(success)
            #     # vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            #     # summaries.append(VideoSummary(
            #     #     'episode_rollout_' + ('success' if success else 'fail'),
            #     #     vid, fps=30))

            #     # error summary
            #     error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
            #                 f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
            #                 f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            #     if not success and self._last_exception is not None:
            #         error_str += f"\n Last Exception: {self._last_exception}"
            #         self._last_exception = None

            #     summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))


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

            # print(f"History keys {obs_history.keys()}")
            # print(f"Transition obs {transition['observation'].keys()}")
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
