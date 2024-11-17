python desc_generator.py --task "PushCube-v1" --save_dir "demos/PushCube-v1/motionplanning"
python train.py

# save results to permanent storage
cp -r /tmp/arm_test /dev/nanxiao-vol1/
cp -r /tmp/arm /dev/nanxiao-vol1/