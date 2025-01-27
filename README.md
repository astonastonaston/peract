# Perceiver-Actor



Perceiver-Actor (PerAct) ([Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation](https://arxiv.org/abs/2209.05451)) is an end-to-end, language-conditioned behavior cloning agent that learns policies for diverse robotic manipulation tasks using a small number of demonstrations per task. 
This repo reproduces PerAct on Maniskill. Codes are adapted from https://github.com/peract/peract


![](media/sim_tasks.gif)



## Guides

- Getting Started: [Installation](#installation)
- Data Generation: [Data Generation](#data-generation)
- Training & Evaluation: [Training and Evaluation](#training-and-evaluation)
- Acknowledgements: [Acknowledgements](#acknowledgements), [Citations](#citations)

## Installation

### Prerequisites

#### 1. Environment

```bash
# create conda virtual env
conda create -n "peract" "python==3.9" --yes
conda activate peract

# clone peract repo
git clone https://github.com/astonastonaston/peract.git && cd peract && git checkout release
pip install --upgrade pip
```

#### 2. Maniskill Installation
Install Maniskill from the stable release:

```bash
# Install Maniskill
pip install --upgrade mani_skill
```

#### 3. Install necessary python libraries

Install necessary python libraries:

```bash
# Install peract package requirements
pip install -r requirements.txt
```
#### 4. Pytorch3d Installation
You need [Pytorch3d](https://github.com/facebookresearch/pytorch3d) to convert rotations and translations as well. Here's how to install it from pre-built wheel:

```bash
# Install pytorch3d from pre-built wheel. This is faster than installing from source
# Note: the python version in the link should match that on your env
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html
```


## Data Generation

### Replay Demonstration

Download demonstrations for a desired task e.g. **PushCube-v1**. "-o" specifies the output directory of your downloaded demo.

```bash
mkdir demos
python -m mani_skill.utils.download_demo "PushCube-v1" -o "demos"
```

Then, we need to process the demonstrations (for PushCube-v1) in preparation for the learning workflow. Note that we replayed 60 demos though not all frames in them are useful: We only extract and use the keyframes. Hence, the replay buffer size during training is not that big.

```bash 
python -m replay_tools.replay_trajectory \
--traj-path demos/PushCube-v1/motionplanning/trajectory.h5 \
--save-traj \
--obs-mode pointcloud \
--num-procs 1 \
--count 60
```

After running this, you can obtain the task trajectories with pointclouds needed for voxelization and training.

### Language Goal Generation
To generate language goal for a given task (e.g. PushCube-v1), you can run:

```bash 
python desc_generator.py --task "PushCube-v1" --save_dir "demos/PushCube-v1/motionplanning"
```

This will generate a .pkl file with a language goal description of the task under the specified directory.



## Training and Evaluation


### Training

#### Config preparations

Make sure you have `config.yaml` under the directory `conf`. `conf` has fine-tuned configs for some tabletop tasks, so if you want to train on those tasks you can directly copy them. For example, if you want to train on **PushCube-v1**, you can prepare configs simply by

```bash
cp conf/config_pushcube.yaml conf/config.yaml
```

Note that you need to **change the following paths** in `config.yaml` for your runtime environment: 

* `maniskill3.tasks`: The task to train on. We only support **StackCube-v1** (the model trained on 50 demos and evaluated over 100 episodes reaches success rate 0.55 at maximum) and **PushCube-v1** (the model trained on 50 demos and evaluated over 100 episodes reaches success rate 1 at maximum) for now. We only support single-task training for now, so only 1 task can be in the list
* `framework.logdir`: The directory to save your training results (weights, csv file with rotation and translation losses if enabled, tensorboard events, etc)
* `maniskill3.traj_path`: The path to your Maniskill demo trajectory (the h5 file)
* `maniskill3.json_path` : The path to your Maniskill demos trajectory metadata (the json) file. 
* `maniskill3.desc_pkl_path`: The path to your language goal file (should be a .pkl file generated from the previous section) 
* `replay.save_keypoints_dir`: The directory to save your detected keypoints in the replays (a json file)

In addition, you can change the following hyperparameters for ablation study. Here is an incomplete lits of them 
*(if you just want a quickstart, you can skip the following readings and go to [Training](#training-1) directly)*:

* `maniskill3.episode_length`: The maximal number of steps (in terms of next-best pose) to reach the goal
* `maniskill3.demos`: The number of demo trajectories to train on
* `maniskill3.scene_bounds`: Scene bounds for voxelization from point clouds

Here is another incomplete list of hyperparameters for **replay keypoint detection**:

* `replay.skip_stopped_steps`: The number of frames skipped for robot stop checking at the beginning of the episode. This is to prevent detecting the first few frames, where the robot remains static, to be the keyframes
* `replay.stop_buffer_init_val`: Initial value of the stop buffer. This is the number of next frames skipped for robot stop checking in the middle of the episode when a stop frame is detected. See `helpers/demo_loading_utils.py`
* `replay.stopping_delta`: The delta of joint velocities for stop checking: A stop frame is marked if all joint velocities are closed to 0 in the range of this delta. See `helpers/demo_loading_utils.py`
* `replay.gripper_open_delta`: The delta for gripper open checking: A gripper is detected open if the gripper position (`qpos[..., -1]`) is greater than this delta. See `helpers/demo_loading_utils.py`

Here are some hyperparameters for the agent **training framework**:

* `framework.training_iterations`: The number of training steps
* `framework.save_freq`: The frequency for saving the weights: The number of intermediate steps between two weight saves
* `framework.tensorboard_logging`: Whether or not to use `tensorboard` to monitor training progress and save the results of losses, weights, etc. The logs are inside `framework.logdir`
* `framework.csv_logging`: Whether or not to save a `csv` file with logging results of losses, weights, etc. The logs are inside `framework.logdir`

#### Training
After preparing the above configs, you can simply run training with:
```bash
python train.py
```

Or, you can feed command line arguments to override the configs:

```bash
export PERACT_ROOT=$(pwd)
python train.py \
    maniskill3.tasks=["PushCube-v1"] \
    maniskill3.traj_path=$PERACT_ROOT/demos/PushCube-v1/motionplanning/trajectory.pointcloud.pd_joint_pos.cpu.h5 \
    maniskill3.json_path=$PERACT_ROOT/demos/PushCube-v1/motionplanning/trajectory.pointcloud.pd_joint_pos.cpu.json \
    maniskill3.desc_pkl_path=$PERACT_ROOT/demos/PushCube-v1/motionplanning/desc.pkl \
    maniskill3.episode_length=6 \
    maniskill3.demos=50 \
    method.voxel_sizes=[100] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=2048 \
    method.transform_augmentation.apply_se3=False \
    method.pos_encoding_with_lang=True \
    replay.save_keypoints=True \
    replay.save_keypoints_dir=/tmp/arm \
    replay.stop_buffer_init_val=8 \
    replay.stopping_delta=0.15 \
    replay.skip_stopped_steps=16 \
    replay.gripper_open_delta=0.03 \
    framework.log_freq=100 \
    framework.save_freq=100 \
    framework.num_weights_to_keep=60 \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.training_iterations=60000 \
    framework.csv_logging=True \
    framework.start_seed=0 \
    framework.tensorboard_logging=True \
    ddp.num_devices=1
```

The results will be saved in `train_data.csv` under `framework.logdir`. 
You may view tensorboard logging events of training in `framework.logdir` as well via

```
tensorboard --logdir={framework.logdir}
```





### Evaluation
#### Config preparations

Similar to training, make sure you have `eval.yaml` under the directory `conf`. To evaluate on **PushCube-v1**, you can simply prepare configs by 

```bash
cp conf/config_pushcube.yaml conf/config.yaml
cp conf/eval_pushcube.yaml conf/eval.yaml
```

Note that you need to **change the following paths** in `eval.yaml` for your runtime environment: 

* `maniskill3.tasks`: The task to evaluate on. We only support **StackCube-v1** and **PushCube-v1** for now. We only support single-task evaluation for now, so only 1 task can be in the list
* `maniskill3.traj_path`: The path to your Maniskill evaluation demo trajectory (the h5 file), though it's not actually used and we only use the ids to generate evaluation trajectories
* `maniskill3.json_path` : The path to your Maniskill evaluation demos trajectory metadata (the json file)
* `maniskill3.desc_pkl_path`: The path to your language goal file 
* `framework.logdir`: The directory to find weights and save your evaluation results
* `framework.train_cfg_path`: The path to your training config. This is used to build a PerAct agent consistent with the agent during training

To **save evaluation videos**, you can change those hyperparameters *(if you just want a quickstart, you can skip the following readings and go to [Evaluation](#evaluation-1) directly)*:

* `cinematic_recorder.enabled`: Enable evaluation video saving
* `cinematic_recorder.save_path`: The directory to save your evaluation videos


In addition, you can change the following hyperparameters for ablation study. Here is an incomplete lits of them:

* `maniskill3.episode_length`: The maximal number of pose steps to reach the goal. This should be consistent with those used in training usually
* `maniskill3.eval_from_eps_number`: Starting episode index for evaluation
* `maniskill3.eval_episodes`: The number of episodes to evaluate on

For visualization, you can **save voxel images at each pose step** under your logging directory as well by toggling the following hyperparameter:

* `framework.eval_save_voxel_images`: A Boolean indicating whether or not to save voxel images at each pose step during evaluation



#### Evaluation
After preparing evaluation configs, you can simply run evaluations via

```bash
python eval.py 
```

Or, you can feed command line arguments to override the configs:

```bash
export PERACT_ROOT=$(pwd)
python eval.py \
    maniskill3.tasks=["PushCube-v1"] \
    maniskill3.traj_path=$PERACT_ROOT/demos/PushCube-v1/motionplanning/trajectory.pointcloud.pd_joint_pos.cpu.h5 \
    maniskill3.json_path=$PERACT_ROOT/demos/PushCube-v1/motionplanning/trajectory.pointcloud.pd_joint_pos.cpu.json \
    maniskill3.desc_pkl_path=$PERACT_ROOT/demos/PushCube-v1/motionplanning/desc.pkl \
    maniskill3.episode_length=6 \
    framework.eval_from_eps_number=50 \
    framework.eval_episodes=10 \
    framework.logdir=$PERACT_ROOT/ckpts/ \
    framework.train_cfg_path=$PERACT_ROOT/conf/config.yaml \
    framework.eval_save_voxel_images=True \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    cinematic_recorder.enabled=True \
    cinematic_recorder.save_path=$PERACT_ROOT/videos/
```

The final results will be saved in `eval_data.csv` under `framework.logdir`. 
You may view tensorboard logging events of evaluation in `framework.logdir` as well via

```
tensorboard --logdir={framework.logdir}
```


<!-- 
## Hardware Requirements

Here the single-task PerAct agent was trained with 1 RTX 2080 card with batch_size=1 and 16GB of memory, 
and it's sufficient to solve basic tasks like PushCube-v1 and StackCube-v1.

Tested with:
- **GPU** - NVIDIA GTX 1660
- **CPU** - Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- **RAM** - 16GB
- **OS** - Ubuntu 20.04

For inference, a single GPU is sufficient. -->

## Acknowledgements

This repository uses code from the following open-source projects:

#### ARM 
Original:  [https://github.com/stepjam/ARM](https://github.com/stepjam/ARM)  
License: [ARM License](https://github.com/stepjam/ARM/LICENSE)    
Changes: Data loading was modified for PerAct. Voxelization code was modified for DDP training.

#### PerceiverIO
Original: [https://github.com/lucidrains/perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch)   
License: [MIT](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)   
Changes: PerceiverIO adapted for 6-DoF manipulation.

#### ViT
Original: [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)     
License: [MIT](https://github.com/lucidrains/vit-pytorch/blob/main/LICENSE)   
Changes: ViT adapted for baseline.   

#### LAMB Optimizer
Original: [https://github.com/cybertronai/pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)   
License: [MIT](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)   
Changes: None.

#### OpenAI CLIP

Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  
Changes: Minor modifications to extract token and sentence features.

Thanks for open-sourcing! 

## Licenses
- [PerAct License (Apache 2.0)](LICENSE) - Perceiver-Actor Transformer
- [ARM License](ARM_LICENSE) - Voxelization and Data Preprocessing 
- [PyRep License (MIT)](https://github.com/stepjam/PyRep/blob/master/LICENSE)
- [Perceiver PyTorch License (MIT)](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)
- [LAMB License (MIT)](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)
- [CLIP License (MIT)](https://github.com/openai/CLIP/blob/main/LICENSE)

## Release Notes

TBC


## Citations 

**PerAct**
```
@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022},
}
```

**C2FARM**
```
@inproceedings{james2022coarse,
  title={Coarse-to-fine q-attention: Efficient learning for visual robotic manipulation via discretisation},
  author={James, Stephen and Wada, Kentaro and Laidlow, Tristan and Davison, Andrew J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13739--13748},
  year={2022}
}
```

**PerceiverIO**
```
@article{jaegle2021perceiver,
  title={Perceiver io: A general architecture for structured inputs \& outputs},
  author={Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and others},
  journal={arXiv preprint arXiv:2107.14795},
  year={2021}
}
```

