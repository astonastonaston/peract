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

Train a `PERACT_BC` agent with `50` demos on PushCube-v1:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    method=PERACT_BC \
    rlbench.tasks=[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \
    rlbench.task_name='multi_18T' \
    rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
    rlbench.demos=100 \
    rlbench.demo_path=$PERACT_ROOT/data/train \
    replay.batch_size=1 \
    replay.path=/tmp/replay \
    replay.max_parallel_processes=32 \
    method.voxel_sizes=[100] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=2048 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
    method.pos_encoding_with_lang=True \
    framework.training_iterations=600000 \
    framework.num_weights_to_keep=60 \
    framework.start_seed=0 \
    framework.log_freq=1000 \
    framework.save_freq=10000 \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=8
```

Make sure there is enough disk-space for `replay.path` and `framework.logdir`. Adjust `replay.max_parallel_processes` to fill the replay buffer in parallel based on your resources. You can also train on fewer GPUs, but training will take a long time to converge. 

To get started, you should probably train on a small number of `rlbench.tasks`. 

Use `tensorboard` to monitor training progress with logs inside `framework.logdir`.

### Testing

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

