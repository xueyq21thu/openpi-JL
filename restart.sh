# Deploy pi0 on libero and ALOHA_SIM
# This script is used to set up the environment for running the OpenPI project on a new server.

apt update
apt install vim
apt install sudo

# Install UV
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install openpi dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Install additional dependencies for Fine Tuning
uv pip install imageio
uv pip install tensorflow
uv pip install tensorflow_datasets
sh dataset.sh

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

# run the server - second terminal
uv run scripts/serve_policy.py --env LIBERO






# # install ALOHA_SIM dependencies
# # uv venv --python 3.10 examples/aloha_sim/.venv
# source examples/aloha_sim/.venv/bin/activate
# # uv pip sync examples/aloha_sim/requirements.txt
# # uv pip install -e packages/openpi-client
# sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev

# # run the ALOHA_SIM
# MUJOCO_GL=egl python examples/aloha_sim/main.py


# # run the server - second terminal
#  # if in aloha_sim
# uv run scripts/serve_policy.py --env ALOHA_SIM