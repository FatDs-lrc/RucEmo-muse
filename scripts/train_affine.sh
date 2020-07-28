set -e
cmd="python train.py --dataroot='/data7/lrc/MuSe2020/MuSe2020_features/wild'
--dataset_mode=muse_wild --model=baseline_map --gpu_ids=1
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=100 --regress_layers=128,128 --hidden_size=-1
--feature_set=vggish,denseface_lrc,bert_base_cover --target=valence 
--batch_size=32 --lr=1e-3 --dropout_rate=0.5 --run_idx=2 --verbose
--niter=100 --niter_decay=100
--name=FcAffine_lstm --suffix={feature_set}_{target}_hidden{hidden_size}_seq{max_seq_len}_run{run_idx}"

# echo "\n-------------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
