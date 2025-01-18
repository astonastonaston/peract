mkdir ckpts
mkdir ckpts/multi
mkdir ckpts/multi/PERACT_BC
mkdir ckpts/multi/PERACT_BC/seed0

# cp -r /dev/nanxiao-vol1/arm_test_stackcube_se3_smaller/arm_test/multi/PERACT_BC/seed0/weights ckpts/multi/PERACT_BC/seed0/
# cp -r /dev/nanxiao-vol1/arm_test_stackcube_50demos_80ksteps_wandb/arm_test/multi/PERACT_BC/seed0/weights ckpts/multi/PERACT_BC/seed0/
# cp -r /dev/nanxiao-vol1/arm_test_pushcube_50demos_80ksteps_wandb/arm_test/multi/PERACT_BC/seed0/weights ckpts/multi/PERACT_BC/seed0/
# cp -r /dev/nanxiao-vol1/arm_test_pushcube_se3_smaller/arm_test/multi/PERACT_BC/seed0/weights ckpts/multi/PERACT_BC/seed0/
expname=peginsertionside_nose3_fewerkfs
cp -r /dev/nanxiao-vol1/arm_test_$expname/arm_test/multi/PERACT_BC/seed0/weights ckpts/multi/PERACT_BC/seed0/