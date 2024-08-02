# Download PushCube demos
python -m mani_skill.utils.download_demo "PushCube-v1" -o "../demos"

# Replay the trajectory to get rgb and pointcloud observations 
python -m mani_skill.trajectory.replay_trajectory --traj-path ../demos/PushCube-v1/motionplanning/trajectory.h5 --save-traj --obs-mode pointcloud -c "pd_joint_delta_pos" --num-procs 1 --count 100