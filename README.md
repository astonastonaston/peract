# Perceiver-Actor



Perceiver-Actor (PerAct) ([Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation](https://arxiv.org/abs/2209.05451)) is an end-to-end, language-conditioned behavior cloning agent that learns policies for diverse robotic manipulation tasks using a small number of demonstrations per task. 
This repo reproduces PerAct on Maniskill. Codes are adapted from https://github.com/peract/peract


![](media/sim_tasks.gif)



## Guides

- Getting Started: [Installation](#installation)
- Data Generation: [Data Generation](#data-generation)
- Training & Evaluation: [Single-Task Training and Evaluation](#training-and-evaluation)
- Miscellaneous: [Recording Videos](#recording-videos)
- Acknowledgements: [Acknowledgements](#acknowledgements), [Citations](#citations)

## Installation

### Prerequisites

#### 1. Environment

```bash
# setup a virtualenv with whichever package manager you prefer
conda create -n "peract" "python==3.10"
conda activate peract
pip install --upgrade pip
```

#### 2. Maniskill Installation
Install Maniskill from the latest commit:

```bash
pip install git+https://github.com/haosulab/ManiSkill.git 
```

#### 3. Install other python libraries

Install other python libraries needed:

```bash
# Install peract package requirements
pip install -r requirements.txt

# Install pytorch3d from source
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl
```

## Data Generation

### Demonstration Generation

Download demonstrations for a desired task e.g. PushCube-v1. "-o" specifies the output directory of your downloaded demo.

```bash
mkdir demos
python -m mani_skill.utils.download_demo "PushCube-v1" -o "demos"
```

Then, we need to process the demonstrations (for PushCube-v1) in preparation for the learning workflow. Note that we replayed 60 demos though not all frames in them are useful: We only extract and use the keyframes. Hence, the replay size during training is not that big.

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

Make sure you have `config.yaml` under the directory `conf`. `conf` has fine-tuned configs for some tabletop tasks, so if you want to train on those tasks you can directly copy them. For example, if you want to train on **PushCube-v1**, you can run: 

```bash
cp conf/config_pushcube.yaml conf/config.yaml
```

Note that you need to change the following paths in `config.yaml` for your runtime environment: 

* `maniskill3.tasks`: The task to train on. We only support **StackCube-v1** (with success rate 0.5 trained on 50 demos and evaled on 10 demos) and **PushCube-v1** (with success rate 1 trained on 50 demos and evaled on 10 demos) for now. We only support single-task training for now, so only 1 task can be in the list
* `framework.logdir`: The directory to save your training results (weights, csv file with rotation and translation losses if enabled, tensorboard events, etc)
* `maniskill3.traj_path`: The path to your Maniskill demo trajectory (the h5 file)
* `maniskill3.json_path` : The path to your Maniskill demos trajectory metadata (the json) file. 
* `maniskill3.desc_pkl_path`: The path to your language goal file (should be a .pkl file generated from the previous section) 
* `replay.save_keypoints_dir`: The directory to save your detected keypoints in the replays (a json file)

In addition, you can change the following hyperparameters for ablation study. Here is an incomplete lits of them *(if you just want a quickstart, you can skip these readings and go to [Training](#training-1) directly)*:

* `maniskill3.episode_length`: The maximal number of steps (in terms of next-best pose) to reach the goal
* `maniskill3.demos`: The number of demo trajectories to train on
* `maniskill3.scene_bounds`: Scene bounds for voxelization from point clouds

Here is another incomplete list of hyperparameters for replay keypoint detection:

* `replay.skip_stopped_steps`: The number of frames skipped for robot stop checking at the beginning of the episode. This is to prevent detecting the first few frames, where the robot remains static, to be the keyframes
* `replay.stop_buffer_init_val`: Initial value of the stop buffer. This is the number of next frames skipped for robot stop checking in the middle of the episode when a stop frame is detected. See `helpers/demo_loading_utils.py`
* `replay.stopping_delta`: The delta of joint velocities for stop checking: A stop frame is marked if all joint velocities are closed to 0 in the range of this delta. See `helpers/demo_loading_utils.py`
* `replay.gripper_open_delta`: The delta for gripper open checking: A gripper is detected open if the gripper position (`qpos[..., -1]`) is greater than this delta. See `helpers/demo_loading_utils.py`

Here are some hyperparameters for the agent training framework:

* `framework.training_iterations`: The number of training steps
* `framework.save_freq`: The frequency for saving the weights: The number of intermediate steps between two weight saves
* `framework.tensorboard_logging`: Whether or not to use `tensorboard` to monitor training progress and save the results of losses, weights, etc. The logs are inside `framework.logdir`
* `framework.csv_logging`: Whether or not to save a `csv` file with logging results of losses, weights, etc. The logs are inside `framework.logdir`

#### Training
After preparing the above configs, you can simply run training with:
```bash
python train.py
```





### Evaluations

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    rlbench.tasks=[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \
    rlbench.task_name='multi_18T' \
    rlbench.demo_path=$PERACT_ROOT/data/test \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=25 \
    framework.eval_type='best' \
    rlbench.headless=True
```

The final results will be saved in `test_data.csv`.

## Recording Videos

To save high-resolution videos of agent executions, set `cinematic_recorder.enabled=True` with `eval.py`:

```bash
cd $PERACT_ROOT
CUDA_VISIBLE_DEVICES=0 python eval.py \
    rlbench.tasks=[open_drawer] \
    rlbench.task_name='multi' \
    rlbench.demo_path=$PERACT_ROOT/data/val \
    framework.gpu=0 \
    framework.logdir=$PERACT_ROOT/ckpts/ \
    framework.start_seed=0 \
    framework.eval_envs=1 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=3 \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_type='last' \
    rlbench.headless=True \
    cinematic_recorder.enabled=True
```

Videos will be saved at `$PERACT_ROOT/ckpts/multi/PERACT_BC/seed0/videos/open_drawer_w600000_s0_succ.mp4`.

## Hardware Requirements

Here the single-task PerAct agent was trained with 1 RTX 2080 card with batch_size=1 and 16GB of memory, 
and it's sufficient to solve basic tasks like PushCube-v1 and StackCube-v1.

Tested with:
- **GPU** - NVIDIA GTX 1660
- **CPU** - Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- **RAM** - 16GB
- **OS** - Ubuntu 20.04

For inference, a single GPU is sufficient.

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

