set -e

cmd="python train.py --dataroot='/data7/lrc/MuSe2020/MuSe2020_features/wild'
--dataset_mode=muse_wild_split --model=mmt_seq --gpu_ids=6
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --hidden_size=20 --num_layers=3 --num_head=5
--a_features=vggish --v_features=denseface_lrc --l_features=bert_base_cover 
--batch_size=32 --lr=4e-3 --dropout_rate=0.5 --target=arousal --run_idx=3 --verbose
--niter=50 --niter_decay=50 
--name=baseline_mmt --suffix={a_features}_{v_features}_{l_features}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh 
