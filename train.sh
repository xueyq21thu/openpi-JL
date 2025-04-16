# CUDA_VISIBLE_DEVICES=4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune --exp-name=my_experiment_varfreq --overwrite
# CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_variable_frequency --exp-name=my_experiment_varfreq --overwrite
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_libero_low_mem_finetune_no10 --exp-name=baseline_no_bowl --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_libero_low_mem_finetune --exp-name=baseline_no_bowl --overwrite