import h5py
import os, glob

def get_dim(feat_name):
    dim_dict = {
        'landmarks_3d': 204,
        'fasttext': 300, 
        'pdm': 40, 
        'xception': 2048, 
        'landmarks_2d': 136, 
        'vggface': 512, 
        'pose': 6, 
        'egemaps': 88, 
        'gocar': 350, 
        'gaze': 288, 
        'openpose': 54, 
        'au': 35, 
        'deepspectrum': 4096,
        'bert_base_cover': 768,
        'bert_medium_cover': 512,
        'bert_mini_cover': 256,
        'albert_base_cover': 768,
        'vggish': 128,
        'denseface': 342,
        'glove': 300,
        'senet50': 256,
        'noisy_stu_effb3': 1536,
        'effnet_finetune': 256,
        'effnet_finetune_e7': 256,
        'effnet_finetune_aug': 256,
        'vgg16': 512,
        'lld': 130
    }
    if dim_dict.get(feat_name) is not None:
        return dim_dict[feat_name]
    else:
        return dim_dict[feat_name.split('_')[0]]

def calc_total_dim(feature_set):
    return sum(map(lambda x: get_dim(x), feature_set))

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# def get_all_dims():
#     root = '/data7/lrc/MuSe2020/MuSe2020_features/wild/feature'
#     h5s = glob.glob(os.path.join(root, '*.h5'))
#     ans = {}
#     for h5 in h5s:
#         name = h5.split('/')[-1].split('.')[0]
#         h5f = h5py.File(h5, 'r')
#         size = h5f['trn']['100']['feature'][()].shape[1]
#         ans[name] = size
    
#     print(ans)

# get_all_dims()