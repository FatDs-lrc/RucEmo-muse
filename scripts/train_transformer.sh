set -e

cmd="python train.py --dataroot='/data7/lrc/MuSe2020/MuSe2020_features/wild'
--dataset_mode=muse_wild --model=transformer --gpu_ids=1
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
--max_seq_len=256 --regress_layers=128,128 --hidden_size=-1 --num_layers=2 --num_heads=4
--feature_set=senet50_pretrained_gs --target=valence
--batch_size=64 --lr=1e-4 --dropout_rate=0.5 --run_idx=2 --verbose
--niter=50 --niter_decay=150
--name=baseline_transformer --suffix={feature_set}_{target}_hidden{hidden_size}_head{num_heads}_layer{num_layers}_seq{max_seq_len}_run{run_idx}"

# cmd="python train.py --dataroot='/data7/lrc/MuSe2020/MuSe2020_features/wild'
# --dataset_mode=muse_wild --model=transformer --gpu_ids=3
# --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=1
# --max_seq_len=256 --regress_layers=128,128 --hidden_size=-1 --num_layers=2 --num_heads=4
# --feature_set=bert_base_cover --target=arousal
# --batch_size=64 --lr=1e-4 --dropout_rate=0.5 --run_idx=2 --verbose
# --niter=50 --niter_decay=150
# --name=baseline_transformer --suffix={feature_set}_{target}_hidden{hidden_size}_head{num_heads}_layer{num_layers}_seq{max_seq_len}_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
