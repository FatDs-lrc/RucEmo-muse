feature=$1
run_idx=$2
gpu_ids=$3

set -e
cmd="python train_multitask.py --dataroot='/data7/lrc/MuSe2020/MuSe2020_features/wild'
--dataset_mode=muse_wild --model=baseline_multitask --gpu_ids=$gpu_ids
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --regress_layers=256,256 --hidden_size=-1
--feature_set=$feature
--batch_size=32 --lr=5e-4 --dropout_rate=0.2 --run_idx=$run_idx --verbose
--name=baseline_lstm_multitask --suffix={feature_set}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-------------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
done