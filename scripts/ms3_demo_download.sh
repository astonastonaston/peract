# Download PushCube demos
# pip install mani_skill==3.0.0b7
pip install git+https://github.com/haosulab/ManiSkill.git 
mkdir demos
# python -m mani_skill.utils.download_demo "PushCube-v1" -o "demos"
# python -m mani_skill.utils.download_demo "StackCube-v1" -o "demos"
# python -m mani_skill.utils.download_demo "PokeCube-v1" -o "demos"
python -m mani_skill.utils.download_demo "PegInsertionSide-v1" -o "demos"

# Replay the trajectory to get pointcloud observations (default in pd_joint_pos control mode)
# python -m replay_tools.replay_trajectory --traj-path demos/PushCube-v1/motionplanning/trajectory.h5 --save-traj --obs-mode pointcloud --num-procs 1 --count 60
# python -m replay_tools.replay_trajectory --traj-path demos/StackCube-v1/motionplanning/trajectory.h5 --save-traj --obs-mode pointcloud --num-procs 1 --count 150
# python -m replay_tools.replay_trajectory --traj-path demos/PokeCube-v1/motionplanning/trajectory.h5 --save-traj --obs-mode pointcloud --num-procs 1 --count 60
python -m replay_tools.replay_trajectory --traj-path demos/PegInsertionSide-v1/motionplanning/trajectory.h5 --save-traj --obs-mode pointcloud --num-procs 1 --count 60
