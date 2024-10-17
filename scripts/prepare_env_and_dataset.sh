# git clone https://github.com/astonastonaston/peract.git && cd peract && git checkout nau

# cp -r /dev/nanxiao-vol1/demos_multi_view ./
# mv ./demos_multi_view ./demos
sh scripts/ms3_demo_download.sh

# install pytorch3d from the wheel
# pip install fvcore
pip install -r pod_requirements.txt

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl

python desc_generator.py

mv conf/config.yaml conf/__config.yaml
mv conf/eval.yaml conf/__eval.yaml
mv conf/pod_config.yaml conf/config.yaml
mv conf/pod_eval.yaml conf/eval.yaml