set -ex 
name=$1
scp -r -i /data1/lrc/key_leo_avec_2230 -P 2230 \
root@202.112.113.78:/root/lrc_git/Muse2020/checkpoints/$name \
./checkpoints