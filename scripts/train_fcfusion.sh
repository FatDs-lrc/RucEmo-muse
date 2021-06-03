set -e
target=$1
task=$2 # stress physio
run_idx=$3
gpu_ids=$4

cmd="python train.py --dataset_mode=muse_$task --model=fcfusion --gpu_ids=$gpu_ids
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=2
--max_seq_len=100 --regress_layers=128,128 --hidden_size=1024
--feature_set=egemaps,bert,vggface --target=$target
--batch_size=8 --lr=1e-4 --dropout_rate=0.3 --run_idx=$run_idx --verbose
--niter=20 --niter_decay=30 --normalize --num_threads=0 --loss_type=mse
--name=baseline_fcfusion_norm --suffix={feature_set}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh