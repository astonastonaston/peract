import h5py
import numpy as np
from mani_skill.utils.io_utils import load_json

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

def add_low_dim_states(obs):    
    gripper_joint_positions = _get_gripper_joint_positions_ms(obs)
    if gripper_joint_positions is not None:
        gripper_joint_positions = np.clip(gripper_joint_positions, 0., 0.04)
    robot_state = np.array([
        _check_gripper_open_ms(obs),
        *gripper_joint_positions]) # left and right finger joint positions
    obs['low_dim_state'] = np.array(robot_state, dtype=np.float32)
    return obs
