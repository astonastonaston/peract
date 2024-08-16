from abc import ABC, abstractmethod
from typing import Any, List

import mplib
import sapien
import torch
from sapien import Pose
from helpers.utils import discrete_euler_to_quaternion, voxel_index_to_point

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
        self.delta_poses = []

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

    def plan_pose_with_RRTConnect(self, pose: Pose, start_qpos):
        """
        Plan and follow a path to a pose using RRTConnect

        Args:
            pose: mplib.Pose
        """
        # result is a dictionary with keys 'status', 'time', 'position', 'velocity',
        # 'acceleration', 'duration'
        # plan_pose ankor
        print("plan_pose")
        result = self.planner.plan_pose(pose, start_qpos, time_step=1 / 250)
        # plan_pose ankor end
        if result["status"] == "Success":
            self.planned_joint_pos = result
            return result
        # do nothing if the planning fails; follow the path if the planning succeeds
        # self.follow_path(result)
        else:
            return 0
    
    def plan_pose_with_screw(self, pose, start_qpos):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            pose,
            start_qpos,
            time_step=1 / 250,
        )
        if result["status"] == "Success":
            self.planned_joint_pos = result
            return result
        else:
            # fall back to RRTConnect if the screw motion fails (say contains collision)
            return self.plan_pose_with_RRTConnect(pose, start_qpos)

    def convert_to_joint_pos(self, start_qpos, rot_resolution):
        # convert end pose (self.action) to delta poses for planning
        self.setup_planner()
        _, rot_grip_action, ignore_collisions_action = self.action[0], self.action[1], self.action[2]
        trans_coordinate = self.observation_elements["attention_coordinate"]
        tg_pose = sapien.Pose(p=trans_coordinate, q=discrete_euler_to_quaternion(rot_grip_action[:, :-1], 
                                                             rot_resolution))
        # TODO: use ignore_collision for planning. Current tasks assume there's no collision 
        res = self.plan_pose_with_screw(tg_pose, start_qpos)
        res_joint_pos = res["position"] # n*7 joint positions, where n -> the number of timesteps
        res_joint_pos = torch.cat(
                [res_joint_pos,
                 rot_grip_action[:, -1].unsqueeze(-1) * 0.4], -1) # assume gripper open state changes at the beginning
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
