# This script is used to collect runable scripts from the openpi-JL repo.

# Reset the environment
cd /workspace/openpi-JL
bash restart.sh

# Data conversion from npy collected by myself to lerobot
cd /workspace/openpi-JL
CUDA_VISIBLE_DEVICES=0 uv run examples/libero/convert_libero_data_to_lerobot.py
CUDA_VISIBLE_DEVICES=0 uv run examples/libero/convert_npy_to_lerobot.py

# Data conversion from two datasets (RLDS and NPY) to lerobot
CUDA_VISIBLE_DEVICES=0 python scripts/libero_dataset.py

# run the server
uv run scripts/serve_policy.py
uv run scripts/serve_policy.py --env LIBERO
uv run scripts/serve_policy.py --env LIBERO_NOISE

# run the libero evaluation
cd /workspace/openpi-JL
source examples/libero/.venv/bin/activate
export PYTHONPATH=/workspace/openpi-JL/third_party/libero:$PYTHONPATH
export PYTHONPATH=/workspace/openpi-JL/src:$PYTHONPATH
python examples/libero/noise_inject_main.py
python examples/libero/main.py

# training the model
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero --exp-name=naive_noise_exp --overwrite
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0_libero_lora_noise
uv run scripts/train.py pi0_libero_lora_noise
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_libero_lora_noise --exp-name=libero_noise_2 --overwrite


CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_libero_low_mem_finetune --overwrite
