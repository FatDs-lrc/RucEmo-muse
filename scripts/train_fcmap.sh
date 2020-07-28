set -e
target=$1
run_idx=$2

cmd="python train.py --dataroot='dataset/wild'
--dataset_mode=muse_wild --model=fcmap --gpu_ids=1
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --regress_layers=128,128 --hidden_size=-1
--feature_set=lld_aligned,wav2vec,vgg16_frame_raw,bert_base_cover --target=$target
--batch_size=32 --lr=1e-4 --dropout_rate=0.5 --run_idx=$run_idx --verbose
--niter=50 --niter_decay=50 --normalize --num_threads=0
--name=baseline_fcfusion_norm_bidirection --suffix={feature_set}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
