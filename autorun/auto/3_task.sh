screen -dmS muse_attnfusion_3
screen -x -S muse_attnfusion_3 -p 0 -X stuff 'sh autorun/attfusion.sh vggish denseface_lrc,au bert_base_cover arousal 1 1
'
screen -x -S muse_attnfusion_3 -p 0 -X stuff 'sh autorun/attfusion.sh vggish denseface_lrc,au bert_base_cover arousal 2 1
'
screen -x -S muse_attnfusion_3 -p 0 -X stuff 'sh autorun/attfusion.sh vggish denseface_lrc,au bert_base_cover valence 1 1
'
screen -x -S muse_attnfusion_3 -p 0 -X stuff 'sh autorun/attfusion.sh vggish denseface_lrc,au bert_base_cover valence 2 1
'
screen -x -S muse_attnfusion_3 -p 0 -X stuff 'exit
'
