from abc import ABC, abstractmethod
from typing import Any, List

import mplib
import numpy as np
import sapien
import torch
from sapien import Pose
from helpers.utils import discrete_euler_to_quaternion, voxel_index_to_point

OPEN = 1
CLOSED = -1

class Summary(object):
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


class ScalarSummary(Summary):
    pass


class HistogramSummary(Summary):
    pass


class ImageSummary(Summary):
    pass


class TextSummary(Summary):
    pass


class VideoSummary(Summary):
    def __init__(self, name: str, value: Any, fps: int = 30):
        super(VideoSummary, self).__init__(name, value)
        self.fps = fps


class ActResult(object):

    def __init__(self, action: Any,
                 observation_elements: dict = None,
                 replay_elements: dict = None,
                 info: dict = None):
        self.action = action
        self.observation_elements = observation_elements or {}
        self.replay_elements = replay_elements or {}
        self.info = info or {}

    def setup_planner(self, **kwargs):
        """
        Create an mplib planner using the default robot.
        See planner.py for more details on the arguments.
        """
        self.planner = mplib.Planner(
            urdf=kwargs.get("urdf_path", "./data/panda/panda.urdf"),
            srdf=kwargs.get("srdf_path", "./data/panda/panda.srdf"),
            move_group=kwargs.get("move_group", "panda_hand"),
        )
        return self.planner

    def plan_pose_with_RRTConnect(self, pose: np.array, start_qpos):
        """
        Plan and follow a path to a pose using RRTConnect

        Args:
            pose: mplib.Pose
        """
        # result is a dictionary with keys 'status', 'time', 'position', 'velocity',
        # 'acceleration', 'duration'
        # plan_pose ankor
        # print("plan_pose")
        # result = self.planner.plan_pose(pose, start_qpos, time_step=1 / 250)
        result = self.planner.plan_qpos_to_pose(pose, start_qpos, time_step=1 / 250)
        # plan_pose ankor end
        print("RRT Connect planning result:")
        print(result)
        if result["status"] == "Success":
            self.planned_joint_pos = result
            print(result["position"].shape)
            return result
        # do nothing if the planning fails; follow the path if the planning succeeds
        # self.follow_path(result)
        else:
            raise ValueError(f"Planning to reach pose {pose} from {start_qpos} but failed")
            return 0
    
    def plan_pose_with_screw(self, pose, start_qpos):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        mimic_pose = start_qpos[:7].copy()
        mimic_pose[1] = mimic_pose[1] - 0.02
        pose = mimic_pose
        result = self.planner.plan_screw(
            pose,
            start_qpos,
            time_step=1 / 250,
        )
        print("Screw planning results")
        print(result)
        if result["status"] == "Success":
            self.planned_joint_pos = result
            print(result["position"].shape)
            return result
        else:
            # fall back to RRTConnect if the screw motion fails (say contains collision)
            print(f"Planning from {start_qpos} to {pose} failed. Falling back to RRT connect method")
            return self.plan_pose_with_RRTConnect(pose, start_qpos)

    def convert_to_joint_pos(self, start_qpos, urdf_path):
        # convert end pose action (self.action) to pd_joint_pos actions for planning
        self.setup_planner(urdf_path=urdf_path, srdf_path=urdf_path.replace(".urdf", ".srdf"))
        # print("self.action")
        # print(self.action)
        print(start_qpos)
        trans_coordinate, rot, gripper_open, _ = self.action[:3], self.action[3:7], self.action[7], self.action[8]
        if gripper_open:
            gripper_open = OPEN
        else:
            gripper_open = CLOSED
        print("attention coord")
        print(trans_coordinate)
        # trans_coordinate = self.observation_elements["attention_coordinate_layer_0"]
        # tg_pose = sapien.Pose(p=trans_coordinate, q=rot)
        tg_pose = np.concatenate([trans_coordinate, rot])
        # TODO: use ignore_collision for planning. Current tasks assume there's no collision 
        # print(type(start_qpos))
        res = self.plan_pose_with_screw(tg_pose, start_qpos.cpu().numpy()[0])
        res_joint_pos = res["position"] # n*7 joint positions, where n -> the number of timesteps
        num_steps = res_joint_pos.shape[0]
        gripper_open_states = np.zeros((num_steps, 1))
        gripper_open_states[:, 0] = gripper_open
        res_joint_pos = np.concatenate(
                [res_joint_pos,
                 gripper_open_states], -1) # assume gripper open state changes at the beginning
        return res_joint_pos


class Agent(ABC):

    @abstractmethod
    def build(self, training: bool, device=None) -> None:
        pass

    @abstractmethod
    def update(self, step: int, replay_sample: dict) -> dict:
        pass

    @abstractmethod
    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:
        # returns dict of values that get put in the replay.
        # One of these must be 'action'.
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def update_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def act_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def load_weights(self, savedir: str) -> None:
        pass

    @abstractmethod
    def save_weights(self, savedir: str) -> None:
        pass
