# git clone https://github.com/astonastonaston/peract.git && cd peract && git checkout ms3

conda install nvidia/label/cuda-12.1.0::cuda-toolkit && pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install -r requirements.txt

sh scripts/ms3_demo_download.sh