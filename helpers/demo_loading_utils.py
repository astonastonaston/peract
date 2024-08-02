import logging
from typing import List

import numpy as np

def _check_gripper_open(demo, i, delta=0):
    # check if the gripper is open at the i-th step
    return demo["obs"]["agent"]["qpos"][i, -1] >= delta

def _get_rgb_from_pcd_obs(demo, i):
    # get rgb at step i from pointcloud observations
    return demo["obs"]["pointcloud"]["rgb"][i]

def _get_ignore_collision(demo, i):
    # get the collision avoidance bit (indicating whether or not to do collision avodiance planning) at step i
    # since current tasks are simple, we set it False by default
    return False

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

def _is_stopped(demo, i, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (_check_gripper_open(demo, i) == _check_gripper_open(demo, i+1).gripper_open and
             _check_gripper_open(demo, i) == _check_gripper_open(demo, i-1) and
             _check_gripper_open(demo, i-2) == _check_gripper_open(demo, i-1)))
    small_delta = np.allclose(_get_joint_velocities(demo, i), 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def keypoint_discovery(episode_idx, h5_file, json_data, 
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    demo_len = len(h5_file)
    if method == 'heuristic':
        demo = h5_file[f"traj_{episode_idx}"]
        prev_gripper_open = _check_gripper_open(demo, 0)
        stopped_buffer = 0
        for i in range(demo_len):
            stopped = _is_stopped(demo, i, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (demo_len - 1)
            if i != 0 and (_check_gripper_open(demo, i) != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = _check_gripper_open(demo, i)
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


# keyframe discovery following my original implementation for PegInsertionSide-v1
# def keypoint_discovery(episode_idx, h5_file, json_data, 
#                        stopping_delta=0.1,
#                        method='heuristic') -> List[int]:
#     episode_keypoints = []
#     traj_len = len(h5_file)
#     if method == 'heuristic':
#         traj = h5_file[f"traj_{episode_idx}"]
#         # prev_gripper_open = demo[0].gripper_open
#         # stopped_buffer = 0
#         # for i, obs in enumerate(demo):
#         #     stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
#         #     stopped_buffer = 4 if stopped else stopped_buffer - 1
#         #     # If change in gripper, or end of episode.
#         #     last = i == (len(demo) - 1)
#         #     if i != 0 and (obs.gripper_open != prev_gripper_open or
#         #                    last or stopped):
#         #         episode_keypoints.append(i)
#         #     prev_gripper_open = obs.gripper_open
#         # if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
#         #         episode_keypoints[-2]:
#         #     episode_keypoints.pop(-2)
#         # logging.debug('Found %d keypoints.' % len(episode_keypoints),
#         #               episode_keypoints)

#         # keep track of robot static transitions
#         episode_keypoints = []
#         for i in (range(traj_len)):
#             curr_qpos = traj['obs']['agent']['qpos'][i]
#             curr_success = traj['success'][i]

#             if (i != 0):
#                 # compute the differences in joint and gribber qposes
#                 diff_qpos = curr_qpos - prev_qpos
#                 diff_joint, diff_gribber = diff_qpos[:7], diff_qpos[7:]
#                 # diff_qpos_norm = np.linalg.norm(diff_qpos)
#                 diff_joint_norm, diff_gribber_norm = np.linalg.norm(diff_joint), np.linalg.norm(diff_gribber)

#                 # checking keyframe criteria:
#                 # the robot joints doesn't rotate much (the joints are static)
#                 is_key_frame = np.isclose(diff_joint_norm, 0, atol=5e-3)

#                 # remove duplicate temporally-closed static-joint keyframe
#                 if (len(episode_keypoints) > 0):
#                     is_key_frame &= (i-episode_keypoints[-1] > 5)

#                 # if task succeeds, record a keyframe
#                 is_key_frame |= ((curr_success != prev_success) and (prev_success == False))

#                 # if gribber suddenly closes, record a keyframe
#                 is_key_frame |= (diff_gribber_norm >= 1e-2)

#                 # store keyframes
#                 if (is_key_frame):
#                     episode_keypoints.append(i)

#             prev_qpos = curr_qpos
#             prev_success = curr_success

#         return episode_keypoints

#     elif method == 'random':
#         # Randomly select keypoints.
#         episode_keypoints = np.random.choice(
#             range(traj_len),
#             size=20,
#             replace=False)
#         episode_keypoints.sort()
#         return episode_keypoints

#     elif method == 'fixed_interval':
#         # Fixed interval.
#         episode_keypoints = []
#         segment_length = traj_len // 20
#         for i in range(0, traj_len, segment_length):
#             episode_keypoints.append(i)
#         return episode_keypoints

#     else:
#         raise NotImplementedError


# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum