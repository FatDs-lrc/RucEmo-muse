set -e

audio=$1
visual=$2
text=$3
target=$4
run_idx=$5
gpu_ids=$6

cmd="python train.py --dataroot='dataset/wild' 
--dataset_mode=muse_wild_split --model=attention_fusion --gpu_ids=$gpu_ids
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --hidden_size=-1 --num_thread=8
--a_features=$audio --v_features=$visual --l_features=$text 
--batch_size=32 --lr=1e-4 --dropout_rate=0.5 --target=$target --run_idx=$run_idx --verbose
--niter=50 --niter_decay=100 
--name=baseline_attfusion --suffix={a_features}_{v_features}_{l_features}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh 
