# Download PushCube demos
mkdir demos
python -m mani_skill.utils.download_demo "PushCube-v1" -o "demos"

# Replay the trajectory to get pointcloud observations (default in pd_joint_pos control mode)
python -m mani_skill.trajectory.replay_trajectory --traj-path demos/PushCube-v1/motionplanning/trajectory.h5 --save-traj --obs-mode pointcloud --num-procs 1 --count 100
# python -m mani_skill.trajectory.replay_trajectory --traj-path demos/PushCube-v1/motionplanning/trajectory.h5 --save-traj --obs-mode pointcloud --num-procs 1 --count 200