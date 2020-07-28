set -e

# submit 1
# valence
# checkpoints="
# baseline_fcfusion_vggish-denseface_lrc_aug-bert_base_cover_valence_hidden-1_seq100_run2;
# baseline_fcfusion_vggish-denseface_lrc-au-bert_base_cover_valence_hidden-1_seq100_run1;
# baseline_fcfusion_vggish-denseface_lrc-bert_base_cover_valence_hidden-1_seq100_run2"
# arousal
# checkpoints="
# baseline_latefusion_vggish_denseface_lrc-au_bert_base_cover_arousal_hidden-1_seq100_run2;
# baseline_lstm_vggish-vggface-bert_base_cover_arousal_hidden-1_seq100_run1;
# baseline_lstm_vggish-vggface-bert_base_cover_arousal_hidden-1_seq100_run2
# "

# submit 2
# valence 
# checkpoints="
# baseline_fcfusion_vggish-denseface_lrc_aug-bert_base_cover_valence_hidden-1_seq100_run2;
# baseline_fcfusion_vggish-denseface_lrc-au-bert_base_cover_valence_hidden-1_seq100_run1;
# baseline_fcfusion_vggish-denseface_lrc-bert_base_cover_valence_hidden-1_seq100_run2;
# baseline_fcfusion_norm_lld_aligned-wav2vec-denseface_lrc_aug-au-bert_base_cover_valence_hidden-1_seq100_run1
# "

# arousal
# checkpoints="
# baseline_fcfusion_norm_lld_aligned-wav2vec-denseface_lrc_aug-au-bert_base_cover_arousal_hidden-1_seq100_run1;
# baseline_fcfusion_norm_lld_aligned-wav2vec-denseface_lrc_aug-au-bert_base_cover_arousal_hidden-1_seq100_run2;
# baseline_fcfusion_norm_lld_aligned-wav2vec-denseface_lrc_aug-bert_base_cover_arousal_hidden-1_seq100_run1;
# baseline_fcfusion_norm_lld_aligned-wav2vec-vgg16_frame_raw-au-bert_base_cover_arousal_hidden-1_seq100_run1
# " 

# submit 3
# valence 
checkpoints="
baseline_fcfusion_vggish-denseface_lrc_aug-bert_base_cover_valence_hidden-1_seq100_run2;
baseline_fcfusion_vggish-denseface_lrc-au-bert_base_cover_valence_hidden-1_seq100_run1;
baseline_fcfusion_norm_lld_aligned-wav2vec-denseface_lrc_aug-au-bert_base_cover_valence_hidden-1_seq100_run1;
baseline_fcfusion_norm_wav2vec-denseface_lrc_aug-bert_base_cover-glove_cover_valence_hidden-1_seq100_run2;
baseline_fcfusion_norm_wav2vec-bert_base_cover-glove_cover_valence_hidden-1_seq100_run2;
baseline_fcfusion_norm_lld_aligned-wav2vec-bert_base_cover-glove_cover_valence_hidden-1_seq100_run1;
baseline_fcfusion_norm_lld_aligned-wav2vec-bert_base_cover-glove_cover_valence_hidden-1_seq100_run2;
baseline_fcfusion_norm_lld_aligned-wav2vec-bert_base_cover-glove_cover_valence_hidden-1_seq100_run4;
baseline_fcfusion_norm_lld_aligned-wav2vec-au-bert_base_cover-glove_cover_valence_hidden1024_seq100_run2;
"

# arousal
# checkpoints="
# baseline_fcfusion_norm_lld_aligned-wav2vec-denseface_lrc_aug-au-bert_base_cover_arousal_hidden-1_seq100_run1;
# baseline_fcfusion_norm_lld_aligned-wav2vec-vgg16_frame_raw-au-bert_base_cover_arousal_hidden-1_seq100_run1;
# baseline_fcfusion_norm_lld_aligned-wav2vec-denseface_lrc_aug-bert_base_cover-glove_cover_arousal_hidden1024_seq100_run1;
# baseline_fcfusion_norm_lld_aligned-wav2vec-vgg16_frame_raw-au-bert_base_cover-glove_cover_arousal_hidden1024_seq100_run1;
# baseline_fcfusion_norm_lld_aligned-wav2vec-vgg16_frame_raw-au-bert_base_cover-glove_cover_arousal_hidden1024_seq100_run2
# " 

checkpoints=`echo $checkpoints | xargs | sed s/[[:space:]]//g`

cmd="python test.py --dataset_mode=None --model=None
--target=valence --submit_dir=submit --write_sub_results
--template_dir=submit/template/wild/label_segments
--checkpoints_dir=./checkpoints --gpu_ids=0
--name=submit3 --test_checkpoints='$checkpoints'"

# echo "\n-------------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
# echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

