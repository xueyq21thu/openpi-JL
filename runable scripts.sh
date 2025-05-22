# This script is used to collect runable scripts from the openpi-JL repo.

# Data conversion from npy collected by myself to lerobot
CUDA_VISIBLE_DEVICES=0 python examples/libero/convert_npy_to_lerobot.py

# Data conversion from two datasets (RLDS and NPY) to lerobot
CUDA_VISIBLE_DEVICES=0 python scripts/libero_dataset.py

# run the server
uv run scripts/serve_policy.py --env LIBERO

# run the libero evaluation
cd /workspace/openpi-JL
source examples/libero/.venv/bin/activate
export PYTHONPATH=/workspace/openpi-JL/third_party/libero:$PYTHONPATH
export PYTHONPATH=/workspace/openpi-JL/src:$PYTHONPATH
python examples/libero/noise_inject_main.py

# training the model
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero --exp-name=naive_noise_exp --overwrite
uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_noise_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_libero_low_mem_noise_finetune --exp-name=dummy_noise_exp0 --overwrite

