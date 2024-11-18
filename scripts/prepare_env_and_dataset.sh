# git clone https://github.com/astonastonaston/peract.git && cd peract && git checkout nau

# activate peract env
conda create -n "peract" "python==3.10" --yes
. ~/.bashrc
conda activate peract
pip install --upgrade pip

# install mani_skill
pip install git+https://github.com/haosulab/ManiSkill.git 

# install necessary python libraries
pip install -r requirements.txt

# install pytorch3d from the wheel
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu116_pyt1130/download.html

# Replace this with copying pre-generated demos will accelerate demo preparation greatly
sh scripts/ms3_demo_download.sh

# Generate language goal
python desc_generator.py --task "PushCube-v1" --save_dir "demos/PushCube-v1/motionplanning"

# config settings
mv conf/config_pushcube.yaml conf/config.yaml
mv conf/eval_pushcube.yaml conf/eval.yaml
