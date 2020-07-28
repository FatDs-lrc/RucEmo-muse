set -e
target=$1
run_idx=$2

cmd="python train.py --dataroot='dataset/wild'
--dataset_mode=muse_wild_split --model=mmt_lstm --gpu_ids=2
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --hidden_size=64 --num_layers=3 --num_head=4
--a_features=lld_aligned --v_features=denseface_lrc_aug --l_features=bert_base_cover 
--batch_size=64 --lr=1e-4 --dropout_rate=0.5 --target=$target --run_idx=$run_idx
--niter=50 --niter_decay=50 --verbose
--name=mmt_lstm --suffix={a_features}_{v_features}_{l_features}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh 
