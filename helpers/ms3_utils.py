import h5py
import torch
import numpy as np
from mani_skill.utils.io_utils import load_json
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.sapien_env import BaseEnv
from transforms3d import quaternions
import sapien

def extract_obs(obs):
    # extract maniskill obs so that only one level of keys are kept
    new_obs = {}
    for k, v in obs.items():
        if (type(v) == dict):
            for i, j in v.items():
                new_obs[i] = j
        else:
            new_obs[k] = v
    return new_obs

def get_ms_demos(traj_path, json_path):
    # Load associated h5 file
    h5_data = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)
    return h5_data, json_data

def _get_gripper_joint_positions_ms(obs):
    # get the left and right finger joints' positions of the gripper
    return obs["agent"]["qpos"][..., -2:]

def _check_gripper_open_ms(obs, delta=0):
    # check if the gripper is open at the i-th step
    return obs["agent"]["qpos"][..., -1] >= delta

def _get_timestep_encoding(t, episode_length):
    # get the timestep's encoding
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    return np.array([time])

def add_low_dim_states(obs, t, episode_length):    
    gripper_joint_positions = _get_gripper_joint_positions_ms(obs)
    if gripper_joint_positions is not None:
        gripper_joint_positions = np.clip(gripper_joint_positions, 0., 0.04)
    robot_state = np.concatenate([
        _check_gripper_open_ms(obs)[:, None],
        gripper_joint_positions, 
        _get_timestep_encoding(t, episode_length)[:, None]], axis=-1) # left and right finger joint positions
    obs['low_dim_state'] = torch.tensor(robot_state, dtype=torch.float)
    return obs


def build_panda_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual
