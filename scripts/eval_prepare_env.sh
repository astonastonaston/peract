# git clone https://github.com/astonastonaston/peract.git && cd peract && git checkout nau
. scripts/eval_exp_name.sh

# activate peract env
pip install --upgrade pip

# install mani_skill and CLIP
pip install mani_skill==3.0.0b18
pip install git+https://github.com/openai/CLIP.git

# install necessary python libraries
pip install -r requirements.txt

# install pytorch3d from the wheel
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu116_pyt1130/download.html

# # Generate language goal
# python desc_generator.py --task "StackCube-v1" --save_dir "./"
# # Generate language goal
# python desc_generator.py --task "PushCube-v1" --save_dir "./"
# Generate language goal
python desc_generator.py --task $taskname --save_dir "./"

# # config settings
# mv conf/config_stackcube.yaml conf/config.yaml
# mv conf/eval_stackcube.yaml conf/eval.yaml
# # config settings
# mv conf/config_pushcube.yaml conf/config.yaml
# mv conf/eval_pushcube.yaml conf/eval.yaml
# config settings
mv conf/config_$configname.yaml conf/config.yaml
mv conf/eval_$configname.yaml conf/eval.yaml
