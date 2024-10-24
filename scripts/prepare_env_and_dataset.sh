# git clone https://github.com/astonastonaston/peract.git && cd peract && git checkout nau

# cp -r /dev/nanxiao-vol1/demos_single_view_50 ./
# mv ./demos_single_view_50 ./demos

# Replace this with copying pre-generated demos will accelerate demo preparation greatly
# sh scripts/ms3_demo_download.sh
cp -r /dev/nanxiao-vol1/demos_multi_view_50_demos ./
mv ./demos_multi_view_50_demos ./demos

# install pytorch3d from the wheel
# pip install fvcore
pip install -r pod_requirements.txt

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl

python desc_generator.py

mv conf/config.yaml conf/__config.yaml
mv conf/eval.yaml conf/__eval.yaml
# mv conf/pod_config_single_cam.yaml conf/config.yaml
# mv conf/pod_eval_single_cam.yaml conf/eval.yaml
mv conf/pod_config.yaml conf/config.yaml
mv conf/pod_eval.yaml conf/eval.yaml