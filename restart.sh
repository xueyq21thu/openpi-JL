# Deploy pi0 on libero and ALOHA_SIM
# This script is used to set up the environment for running the OpenPI project on a new server.

apt update
apt install vim
apt install sudo

cd /workspace/openpi-JL
git submodule update --init --recursive

# Install UV
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install openpi dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Install additional dependencies for Fine Tuning
uv pip install imageio
uv pip install tensorflow tensorflow_datasets
CUDA_VISIBLE_DEVICES=0 python examples/libero/convert_libero_data_to_lerobot_noise.py

# Install libero dependencies
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
sudo apt-get install libegl-dev

# run the libero
export PYTHONPATH=/workspace/openpi-JL/third_party/libero:$PYTHONPATH
export PYTHONPATH=/workspace/openpi-JL/src:$PYTHONPATH
python examples/libero/main.py
python examples/libero/noise_inject_main.py

# run the server - second terminal
uv run scripts/serve_policy.py --env LIBERO

# data conversion - third terminal
# first install the conda environment
cd /workspace/rlds_dataset_builder
conda env create -f environment_ubuntu.yml

conda activate rlds_env
cd /workspace/openpi-JL/data/libero/npy/libero_spatial_no_noops
tfds build --overwrite

# training the model
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero --exp-name=naive_noise_exp --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_low_mem_noise_finetune --exp-name=naive_noise_exp --overwrite
