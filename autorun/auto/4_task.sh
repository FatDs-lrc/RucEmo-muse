screen -dmS muse_latefusion_4
screen -x -S muse_latefusion_4 -p 0 -X stuff 'sh autorun/latefusion.sh vggish xception bert_base_cover valence 1 3
'
screen -x -S muse_latefusion_4 -p 0 -X stuff 'sh autorun/latefusion.sh vggish xception bert_base_cover valence 2 3
'
screen -x -S muse_latefusion_4 -p 0 -X stuff 'sh autorun/latefusion.sh vggish vgg16_frame_raw bert_base_cover arousal 1 3
'
# screen -x -S muse_latefusion_4 -p 0 -X stuff 'exit
# '
