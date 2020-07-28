set -e
target=$1
run_idx=$2

cmd="python train.py --dataroot='dataset/wild'
--dataset_mode=muse_wild --model=fcfusion --gpu_ids=3
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=2
--max_seq_len=100 --regress_layers=128,128 --hidden_size=1024
--feature_set=wav2vec,noisy_sty_b3_raw_cp,au,bert_base_cover,glove_cover --target=$target
--batch_size=24 --lr=1e-4 --dropout_rate=0.5 --run_idx=$run_idx --verbose
--niter=50 --niter_decay=50 --normalize --num_threads=0 --loss_type=mse
--name=baseline_fcfusion_norm --suffix={feature_set}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
