sh scripts/eval_load_weights.sh

sh scripts/prepare_env_and_dataset.sh

pip install git+https://github.com/openai/CLIP.git

python eval.py 

sh scripts/eval_save_results.sh