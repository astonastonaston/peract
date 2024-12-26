# git clone https://github.com/astonastonaston/peract.git && cd peract && git checkout nau

# activate peract env
pip install --upgrade pip

# install mani_skill and CLIP
pip install git+https://github.com/haosulab/ManiSkill.git 
pip install git+https://github.com/openai/CLIP.git

# install necessary python libraries
pip install -r requirements.txt

# install pytorch3d from the wheel
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu116_pyt1130/download.html

# # Generate language goal
# python desc_generator.py --task "StackCube-v1" --save_dir "./"
# Generate language goal
python desc_generator.py --task "PushCube-v1" --save_dir "./"

# # config settings
# mv conf/config_stackcube.yaml conf/config.yaml
# mv conf/eval_stackcube.yaml conf/eval.yaml
# config settings
mv conf/config_pushcube.yaml conf/config.yaml
mv conf/eval_pushcube.yaml conf/eval.yaml
