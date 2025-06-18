
"""
# mace/scripts/eval_configs.py
python3 <mace_repo_dir>/mace/cli/eval_configs.py \
    --configs="your_configs.xyz" \
    --model="your_model.model" \
    --output="./your_output.xyz"

# training
# mace/scripts/run_train.py
python <mace_repo_dir>/mace/cli/run_train.py \
    --name="MACE_model" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --swa \
    --start_swa=1200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \

# finetuning
# mace/scripts/run_train.py
mace_run_train \
    --name="MACE" \
    --foundation_model="small" \
    --multiheads_finetuning=False \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --energy_weight=1.0 \
    --forces_weight=1.0 \
    --E0s="average" \
    --lr=0.01 \
    --scaling="rms_forces_scaling" \
    --batch_size=2 \
    --max_num_epochs=6 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --default_dtype="float64" \
    --device=cuda \
    --seed=3
"""
