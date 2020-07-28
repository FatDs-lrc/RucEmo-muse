# Workspace For Muse2020 challenge

## run this project
### requirementes

+ python==3.7

+ pytorch>=1.0.0

using scripts to run experiments

+ run `sh scripts/train_fcfusion arousal/valence 1` and modify --feature_set args

## result
### basic result on validation set

|   Features  |                               | arousal | valence |
|:-----------:|:-----------------------------:|:-------:|:-------:|
|  Uni-modal  | lld                           | 0.3841  |  0.03   |
|             | wav2vec                       | 0.3092  |  0.1737 |
|             | denseface                     | 0.2954  | 0.0524  |
|             | vggface                       | 0.2996  | 0.0852  |
|             | bert_base                     | 0.1643  | 0.3131  |
| multi-modal | lld-bert-glove                | 0.3378  | 0.3447  |
|             | wav2vec-bert_base-glove       | 0.3440  | 0.3556  |
|             | lld-wav2vec-denseface-au-bert | 0.4670  | 0.3571  |
|             | lld-wav2vec-vgg16-au-bert     | 0.4514  | 0.3153  |

Note that is table is not complete yet.


### submission result

|   name   | arousal | valence |
|:--------:|:-------:|:-------:|
| baseline |  0.2834 |  0.2431 |
|  submit1 |  0.3082 |  40.814 |
|  submit2 |  0.3753 |  0.4194 |
|  submit3 |  0.4346 |  0.4513 |

