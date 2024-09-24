# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

import logging
import copy
import os
from typing import List, Dict


import numpy as np
import pickle
from runners.observation_type import ObservationElement
from runners.task_uniform_replay_buffer import TaskUniformReplayBuffer
from runners.replay_buffer import ReplayElement, ReplayBuffer
from runners.uniform_replay_buffer import UniformReplayBuffer

from helpers import demo_loading_utils, utils
from helpers.utils import save_dict_to_json
from helpers.ms3_utils import get_ms_demos
from helpers.preprocess_agent import PreprocessAgent
from helpers.clip.core.clip import tokenize
from agents.peract_bc.perceiver_lang_io import PerceiverVoxelLangEncoder
from agents.peract_bc.qattention_peract_bc_agent import QAttentionPerActBCAgent
from agents.peract_bc.qattention_stack_agent import QAttentionStackAgent

import torch
import torch.nn as nn
import multiprocessing as mp
from torch.multiprocessing import Process, Value, Manager
from helpers.clip.core.clip import build_model, load_clip, tokenize
from omegaconf import DictConfig

REWARD_SCALE = 100.0
LOW_DIM_SIZE = 4


def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  voxel_sizes,
                  image_size=[128, 128],
                  replay_size=3e5):
    num_cam = len(cameras)
    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))

    # rgb, point cloud, intrinsics, extrinsics
    observation_elements.append(
        ObservationElement('segmentation', (np.prod(image_size) * num_cam, 1), np.float32)) # point cloud segmentation
    observation_elements.append( 
        ObservationElement('rgb', (np.prod(image_size) * num_cam, 3), np.float32))
    observation_elements.append(
        ObservationElement('point_cloud', (np.prod(image_size) * num_cam, 4),
                            np.float32)) # in homogeneous coordinates
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.int32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.int32),
        ReplayElement('ignore_collisions', (ignore_collisions_size,),
                      np.int32),
        ReplayElement('gripper_pose', (gripper_pose_size,),
                      np.float32),
        ReplayElement('lang_goal_emb', (lang_feat_dim,),
                      np.float32),
        ReplayElement('lang_token_embs', (max_token_seq_len, lang_emb_dim,),
                      np.float32), # extracted from CLIP's language encoder
        ReplayElement('task', (),
                      str),
        ReplayElement('lang_goal', (1,),
                      object),  # language goal string for debugging and visualization
        ReplayElement('demo_number', (),
                      np.int32),
        ReplayElement('input_frame', (),
                      np.int32),
        ReplayElement('supervision_frame', (),
                      np.int32),
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool_),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(
        demo: Dict,
        tpl_index: int, tml_index: int,
        scene_bounds: List[float], # metric 3D bounds of the scene
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool):
    # TODO: change the way to get gripper pose from obs, using ms3 style
    # obs_tp1 = demo[keypoint]
    # obs_tm1 = demo[max(0, keypoint - 1)]
    # tpl_index, tml_index = keypoint, max(0, keypoint-1)
    tpl_gripper_pose = demo_loading_utils._get_gripper_pose(demo, tpl_index)
    quat = utils.normalize_quaternion(tpl_gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

    attention_coordinate = tpl_gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(scene_bounds)
    ignore_collisions = int(demo_loading_utils._get_ignore_collision(demo, tml_index))
    # ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                     attention_coordinate + bounds_offset[depth - 1]])
        index = utils.point_to_voxel_index( # convert the next gripper pose to voxel indices, used as the translational target index
            tpl_gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(demo_loading_utils._check_gripper_open(demo, tpl_index))
    rot_and_grip_indicies.extend([int(demo_loading_utils._check_gripper_open(demo, tpl_index))])
    # rot_and_grip_indicies -> gripper rotation discrete eular angles (indices) + gripper open state index
    # trans_indicies -> gripper translational voxel index
    # attention_coordinates -> gripper translational coordinates
    # only gripper pose and gripper open state are used as actions eventually
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [tpl_gripper_pose, np.array([grip])]), attention_coordinates


def _add_keypoints_to_replay(
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        demo: Dict,
        i: int, # current timestep w.r.t pd_joint_pos-based control
        demo_meta_data: Dict, 
        episode_keypoints: List[int],
        cameras: List[str],
        scene_bounds: List[float],
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        description: str = '',
        clip_model = None,
        device = 'cpu',
        demo_number=None):
    prev_action = None
    episode_length = cfg.maniskill3.episode_length # for single-task training, it should be closed to demo_meta_data["env_info"]["max_episode_steps"]
    # print(f"Desc is {description}")
    for k, keypoint in enumerate(episode_keypoints):
        # obs_tp1 = demo[keypoint]
        # obs_tm1 = demo[max(0, keypoint - 1)]
        tpl_index, tml_index = keypoint, max(0, keypoint-1)
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
            demo, tpl_index, tml_index, scene_bounds, voxel_sizes, bounds_offset,
            rotation_resolution, crop_augmentation) # action -> next kf gripper pose

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        print(f"obs from ind {i} and gripper pose from ind {tpl_index}")
        obs_dict = utils.extract_obs(demo, step=i, t=k, prev_action=prev_action,
                                     cameras=cameras, episode_length=episode_length)
        tokens = tokenize(description).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        obs_dict['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
        obs_dict['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
            'gripper_pose': demo_loading_utils._get_gripper_pose(demo, tpl_index),
            'task': task,
            'lang_goal': np.array(description, dtype=object), # TODO: should be token embeddings ???
            'demo_number': demo_number,
            'input_frame': i,
            'supervision_frame': tpl_index
        }

        others.update(final_obs) # update with gripper pose and expert action
        others.update(obs_dict) # update with language goal and embeddings

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)

        # add labelled keypoints' data
        i = copy.deepcopy(tpl_index)

    # final step staying static without moving
    obs_dict_tp1 = utils.extract_obs(demo, tpl_index, t=k + 1, prev_action=prev_action, 
                                     cameras=cameras, episode_length=episode_length)
    obs_dict_tp1['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)


def fill_replay(cfg: DictConfig,
                # obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                clip_model = None,
                device = 'cpu',
                keypoint_method = 'heuristic'):
    logging.getLogger().setLevel(cfg.framework.logging_level)

    if clip_model is None:
        model, _ = load_clip('RN50', jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug('Filling %s replay ...' % task)
    # load demo rgbd and meta data
    demo, demo_meta_data = get_ms_demos(cfg.maniskill3.traj_path, cfg.maniskill3.json_path)
    # print(f"Num of demos: {num_demos}")
    keypts = {}
    for d_idx in range(num_demos):
        # load language descs
        with open(cfg.maniskill3.desc_pkl_path, 'rb') as f:
            desc = pickle.load(f)

        # extract keypoints (a.k.a keyframes)
        episode_keypoints = demo_loading_utils.keypoint_discovery(d_idx, demo, demo_meta_data, stopped_buffer_init_val=cfg.replay.stop_buffer_init_val, method=keypoint_method)
        if cfg.replay.save_keypoints:
            keypts[d_idx] = episode_keypoints

        # print(f"Keypoints for episode {d_idx}: {episode_keypoints}")

        if rank == 0:
            logging.info(f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints: {episode_keypoints} - {task}")

        # for the episode, add keyframes
        demo_ep = demo[f"traj_{d_idx}"]
        # print(f"demo position-ctl epi length {len(demo_ep)}")
        demo_len = demo_loading_utils._get_demo_len(demo_ep)
        for i in range(demo_len - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0: # augment demo every n steps
                continue

            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            print(f"adding demo and frame index {d_idx, i}")
            _add_keypoints_to_replay(
                cfg, task, replay, demo_ep, i, demo_meta_data, episode_keypoints, cameras,
                scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, description=desc,
                clip_model=clip_model, device=device, demo_number=d_idx)
    if cfg.replay.save_keypoints:
        keypt_dir = cfg.replay.save_keypoints_dir
        logging.info(f"Saving keypoints to {keypt_dir}")
        if not os.path.exists(keypt_dir):
            os.makedirs(keypt_dir)
        save_dict_to_json(keypts, os.path.join(keypt_dir, "keypts.json"))

    logging.debug('Replay %s filled with demos.' % task)


def fill_multi_task_replay(cfg: DictConfig,
                        #    obs_config: ObservationConfig, # TODO: add obs config back when it's ready
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str],
                           scene_bounds: List[float],
                           voxel_sizes: List[int],
                           bounds_offset: List[float],
                           rotation_resolution: int,
                           crop_augmentation: bool,
                           clip_model = None,
                           keypoint_method = 'heuristic'):
    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
                                        if torch.cuda.is_available() else 'cpu')
            p = Process(target=fill_replay, args=(cfg,
                                                #   obs_config,
                                                  rank,
                                                  replay,
                                                  task,
                                                  num_demos,
                                                  demo_augmentation,
                                                  demo_augmentation_every_n,
                                                  cameras,
                                                  scene_bounds,
                                                  voxel_sizes,
                                                  bounds_offset,
                                                  rotation_resolution,
                                                  crop_augmentation,
                                                  clip_model,
                                                  model_device,
                                                  keypoint_method))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def create_agent(cfg: DictConfig):
    LATENT_SIZE = 64
    depth_0bounds = cfg.maniskill3.scene_bounds
    cam_resolution = cfg.maniskill3.camera_resolution

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        last = depth == len(cfg.method.voxel_sizes) - 1
        perceiver_encoder = PerceiverVoxelLangEncoder(
            depth=cfg.method.transformer_depth,
            iterations=cfg.method.transformer_iterations,
            voxel_size=vox_size,
            initial_dim = 3 + 3 + 1 + 3,
            low_dim_size=4,
            layer=depth,
            num_rotation_classes=num_rotation_classes if last else 0,
            num_grip_classes=2 if last else 0,
            num_collision_classes=2 if last else 0,
            input_axis=3,
            num_latents = cfg.method.num_latents,
            latent_dim = cfg.method.latent_dim,
            cross_heads = cfg.method.cross_heads,
            latent_heads = cfg.method.latent_heads,
            cross_dim_head = cfg.method.cross_dim_head,
            latent_dim_head = cfg.method.latent_dim_head,
            weight_tie_layers = False,
            activation = cfg.method.activation,
            pos_encoding_with_lang=cfg.method.pos_encoding_with_lang,
            input_dropout=cfg.method.input_dropout,
            attn_dropout=cfg.method.attn_dropout,
            decoder_dropout=cfg.method.decoder_dropout,
            lang_fusion_type=cfg.method.lang_fusion_type,
            voxel_patch_size=cfg.method.voxel_patch_size,
            voxel_patch_stride=cfg.method.voxel_patch_stride,
            no_skip_connection=cfg.method.no_skip_connection,
            no_perceiver=cfg.method.no_perceiver,
            no_language=cfg.method.no_language,
            final_dim=cfg.method.final_dim,
        )
        # print(f"cam in cfg {cfg.maniskill3.cameras}")

        qattention_agent = QAttentionPerActBCAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            perceiver_encoder=perceiver_encoder,
            camera_names=cfg.maniskill3.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            lr=cfg.method.lr,
            training_iterations=cfg.framework.training_iterations,
            lr_scheduler=cfg.method.lr_scheduler,
            num_warmup_steps=cfg.method.num_warmup_steps,
            trans_loss_weight=cfg.method.trans_loss_weight,
            rot_loss_weight=cfg.method.rot_loss_weight,
            grip_loss_weight=cfg.method.grip_loss_weight,
            collision_loss_weight=cfg.method.collision_loss_weight,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            transform_augmentation=cfg.method.transform_augmentation.apply_se3,
            transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
            transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
            transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
            optimizer_type=cfg.method.optimizer,
            num_devices=cfg.ddp.num_devices,
        )
        qattention_agents.append(qattention_agent)

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.maniskill3.cameras,
    )
    preprocess_agent = PreprocessAgent(
        pose_agent=rotation_agent
    )
    return preprocess_agent
