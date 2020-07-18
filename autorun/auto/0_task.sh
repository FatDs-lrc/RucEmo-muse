screen -dmS muse_attnfusion_0
screen -x -S muse_attnfusion_0 -p 0 -X stuff 'sh autorun/attfusion.sh vggish denseface_lrc bert_base_cover arousal 1 0
'
screen -x -S muse_attnfusion_0 -p 0 -X stuff 'sh autorun/attfusion.sh vggish denseface_lrc bert_base_cover arousal 2 0
'
screen -x -S muse_attnfusion_0 -p 0 -X stuff 'sh autorun/attfusion.sh vggish denseface_lrc bert_base_cover valence 1 0
'
screen -x -S muse_attnfusion_0 -p 0 -X stuff 'sh autorun/attfusion.sh vggish denseface_lrc bert_base_cover valence 2 0
'
screen -x -S muse_attnfusion_0 -p 0 -X stuff 'exit
'
