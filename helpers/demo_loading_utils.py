import logging
from typing import List

import numpy as np

def _check_gripper_open(demo, i, delta=1e-3):
    # check if the gripper is open at the i-th step
    return demo["obs"]["agent"]["qpos"][i, -1] > delta

def _get_demo_len(demo):
    # get the length of the given demo episode
    return demo["obs"]["agent"]["qpos"].shape[0]

def _get_rgb_from_pcd_obs(demo, i):
    # get rgb at step i from pointcloud observations
    return demo["obs"]["pointcloud"]["rgb"][i]

def _get_rgb_range_from_pcd_obs(demo, i):
    # get the range of rgb at step i from pointcloud observations
    # print(demo["obs"]["pointcloud"]["rgb"][i].shape)
    return [np.max(demo["obs"]["pointcloud"]["rgb"][i], axis=0), np.min(demo["obs"]["pointcloud"]["rgb"][i], axis=0)]

def _get_ignore_collision(demo, i):
    # get the collision avoidance bit (indicating whether or not to do collision avodiance planning) at step i
    # since current tasks are simple, we set it False by default
    return False

def _get_seg_from_pcd_obs(demo, i):
    # get pointcloud segmentation at step i from pointcloud observations
    return demo["obs"]["pointcloud"]["segmentation"][i]

def _get_pcd_from_pcd_obs(demo, i):
    # get pointcloud at step i from pointcloud observations
    # note that the pcds are in homogeneous coordinate format: w=0 for infinitely far points and w=1 for the rest
    return demo["obs"]["pointcloud"]["xyzw"][i]

def _get_joint_velocities(demo, i):
    # get the velocities of the arm joints at the i-th step (all values are positive)
    return demo["obs"]["agent"]["qvel"][i, :-2]

def _get_gripper_pose(demo, i):
    # get the tool center point (tcp)'s pose as the gripper pose
    return demo["obs"]["extra"]["tcp_pose"][i, :]

def _get_gripper_joint_positions(demo, i):
    # get the left and right finger joints' positions of the gripper
    return demo["obs"]["agent"]["qpos"][i, -2:]

def _get_camera_extrinsics_intrinsics(demo, i, camera_name):
    # get the extrinsics and intrinsics of the camera with a given name. Note that the extrinsic is cam2world_gl (with shape 4*4) instead of extrinsic_cv (with shape 3*4)
    assert camera_name in demo["obs"]["sensor_param"].keys(), f"No such camera sensor with name {camera_name} in observations"
    return demo["obs"]["sensor_param"][camera_name]["cam2world_gl"][i], demo["obs"]["sensor_param"][camera_name]["intrinsic_cv"][i]

def _is_stopped(demo, demo_len, i, stopped_buffer, stopping_delta=0.1, gripper_open_delta=0.025):
    next_is_not_final = i == (demo_len - 2)
    gripper_state_no_change = (
            i < (demo_len - 2) and i >= 2 and
            (_check_gripper_open(demo, i, gripper_open_delta) == _check_gripper_open(demo, i+1, gripper_open_delta) and
             _check_gripper_open(demo, i, gripper_open_delta) == _check_gripper_open(demo, i-1, gripper_open_delta) and
             _check_gripper_open(demo, i-2, gripper_open_delta) == _check_gripper_open(demo, i-1, gripper_open_delta)))
    small_delta = np.allclose(_get_joint_velocities(demo, i), 0, atol=stopping_delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def keypoint_discovery(d_idx, h5_file, json_data, stopped_buffer_init_val=16,
                       stopping_delta=0.1, # Those stopping parameters are overridden at conf/config.yaml
                       method='heuristic',
                       skip_stopped_steps=10,
                       gripper_open_delta=0.025) -> List[int]:
    # Note: gripper_open_delta is overriden by replay.gripper_open_delta
    # stopping_delta is overriden by replay.stopping_delta
    # stopped_buffer_init_val is overriden by replay.stopped_buffer_init_val
    # skip_stopped_steps is overriden by replay.skip_stopped_steps
    episode_keypoints = []
    demo = h5_file[f"traj_{d_idx}"]
    demo_len = _get_demo_len(demo)
    # print(f"Demo len {demo_len}")
    
    if method == 'heuristic':
        # Heuristically select keypoints by robot-stop checking and gripper-open checking.
        prev_gripper_open = _check_gripper_open(demo, 0, gripper_open_delta)
        stopped_buffer = skip_stopped_steps # Not counting stopping for the first few frames
        for i in range(demo_len):
            stopped = _is_stopped(demo, demo_len, i, stopped_buffer, stopping_delta, gripper_open_delta)
            stopped_buffer = stopped_buffer_init_val if stopped else stopped_buffer - 1
            last = i == (demo_len - 1)
            curr_gripper_open = _check_gripper_open(demo, i, gripper_open_delta)

            # If change in gripper, stop, or at end of episode, mark the keypoint
            if i != 0 and (curr_gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)

            prev_gripper_open = _check_gripper_open(demo, i, gripper_open_delta)
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]: # pop a repetitive final keypoint
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(demo_len),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = demo_len // 20
        for i in range(0, demo_len, segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum