# Deploy pi0 on libero and ALOHA_SIM
# This script is used to set up the environment for running the OpenPI project on a new server.
cd /workspace
apt update
apt install vim -y
apt install sudo -y
sudo apt install tmux -y
apt-get install git-lfs -y
git lfs install

cd /workspace/openpi-JL
git submodule update --init --recursive

# Install UV
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Remove .venv if it exists
if [ -d ".venv" ]; then
    rm -rf .venv
fi
# Create a new virtual environment
# Install openpi dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Install additional dependencies for data conversion
uv pip install imageio
uv pip install tensorflow tensorflow_datasets

# Remove .venv if it exists
if [ -d "examples/libero/.venv" ]; then
    rm -rf examples/libero/.venv
fi
# Create a new virtual environment for libero
# Install libero dependencies
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
sudo apt-get install libegl-dev -y

# use the CLIP model
uv pip install transformers
uv pip upgrade transformers, tokenizers
uv pip install numpy==1.24.3 numba==0.57.1