set -e
target=$1
run_idx=$2

cmd="python train.py --dataroot='dataset/wild'
--dataset_mode=muse_wild_split --model=mmt_lstm --gpu_ids=1
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --hidden_size=300 --regress_layers=128,128
--a_features=lld_aligned,wav2vec --v_features=denseface_lrc_aug,au --l_features=bert_base_cover,glove_cover 
--batch_size=64 --lr=1e-4 --dropout_rate=0.5 --target=$target --run_idx=$run_idx
--niter=50 --niter_decay=50 --verbose --normalize_a
--name=mmt_lstm --suffix={a_features}_{v_features}_{l_features}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh 
