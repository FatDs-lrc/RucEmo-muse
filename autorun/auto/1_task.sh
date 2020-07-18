screen -dmS muse_attnfusion_1
screen -x -S muse_attnfusion_1 -p 0 -X stuff 'sh autorun/attfusion.sh vggish xception bert_base_cover arousal 1 0
'
screen -x -S muse_attnfusion_1 -p 0 -X stuff 'sh autorun/attfusion.sh vggish xception bert_base_cover arousal 2 0
'
screen -x -S muse_attnfusion_1 -p 0 -X stuff 'sh autorun/attfusion.sh vggish xception bert_base_cover valence 1 0
'
screen -x -S muse_attnfusion_1 -p 0 -X stuff 'sh autorun/attfusion.sh vggish xception bert_base_cover valence 2 0
'
screen -x -S muse_attnfusion_1 -p 0 -X stuff 'exit
'
