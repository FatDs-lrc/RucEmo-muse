feature=$1
target=$2
run_idx=$3
gpu_ids=$4

set -e

cmd="python train.py --dataroot='/data7/lrc/MuSe2020/MuSe2020_features/wild'
--dataset_mode=muse_wild --model=compose --gpu_ids=$gpu_ids
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --regress_layers=128,128 --hidden_size=-1
--feature_set=$feature --target=$target
--num_heads=4 --num_layers=2 --init_type=xavier
--batch_size=32 --lr=1e-4 --dropout_rate=0.5 --run_idx=$run_idx --verbose
--niter=50 --niter_decay=50
--name=baseline_compose_lt --suffix={feature_set}_{target}_hidden{hidden_size}_layers{num_layers}_heads{num_heads}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
