# git clone https://github.com/astonastonaston/peract.git && cd peract && git checkout ms3

conda install -y nvidia/label/cuda-12.1.0::cuda-toolkit && pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install -r pod_requirements.txt

sh scripts/ms3_demo_download.sh

python desc_generator.py

mv conf/config.yaml conf/__config.yaml
mv conf/eval.yaml conf/__eval.yaml
mv conf/pod_config.yaml conf/config.yaml
mv conf/pod_eval.yaml conf/eval.yaml