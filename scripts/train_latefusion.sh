set -e
target=$1
run_idx=$2
gpu_ids=$3

cmd="python train.py --dataroot='dataset/wild' 
--dataset_mode=muse_wild_split --model=late_fusion --gpu_ids=$gpu_ids
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --hidden_size=-1 --num_thread=8
--a_features=vggish --v_features=denseface_lrc,au --l_features=bert_base_cover 
--batch_size=32 --lr=1e-3 --dropout_rate=0.5 --target=$target --run_idx=$run_idx --verbose
--niter=60 --niter_decay=90 --normalize
--name=baseline_latefusion --suffix={a_features}_{v_features}_{l_features}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh 
