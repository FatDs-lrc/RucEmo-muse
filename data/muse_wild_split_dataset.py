import os
import h5py
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .base_dataset import BaseDataset


class MuseWildSplitDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--a_features', type=str, default='None', help='audio feature to use, split by comma, eg: "egemaps,vggish"')
        parser.add_argument('--v_features', type=str, default='None', help='visual feature to use, split by comma, eg: "vggface"')
        parser.add_argument('--l_features', type=str, default='None', help='lexical feature to use, split by comma, eg: "bert_base"')
        return parser
    
    def __init__(self, opt, set_name):
        ''' MuseWild dataset
        Parameter:
        --------------------------------------
        set_name: [trn, val, tst]
        '''
        super().__init__(opt)
        self.a_features = list(map(lambda x: x.strip(), opt.a_features.split(',')))
        self.v_features = list(map(lambda x: x.strip(), opt.v_features.split(',')))
        self.l_features = list(map(lambda x: x.strip(), opt.l_features.split(',')))
        self.set_name = set_name
        self.load_label()
        self.load_feature()
        self.manual_collate_fn = True
        print(f"MuseWild dataset {set_name} created with total length: {len(self)}")
    
    def load_label(self):
        partition_h5f = h5py.File(os.path.join(self.root, 'target', 'partition.h5'), 'r')
        self.seg_ids = sorted(partition_h5f[self.set_name])
        self.seg_ids = list(map(lambda x: str(x), self.seg_ids))
        label_h5f = h5py.File(os.path.join(self.root, 'target', '{}_target.h5'.format(self.set_name)), 'r')
        self.target = {}
        for _id in self.seg_ids:
            if self.set_name != 'tst':
                self.target[_id] = {
                    'arousal': torch.from_numpy(label_h5f[_id]['arousal'][()]).float(),
                    'valence': torch.from_numpy(label_h5f[_id]['valence'][()]).float(),
                    'length': torch.as_tensor(label_h5f[_id]['length'][()]).long(),
                    'timestamp': torch.from_numpy(label_h5f[_id]['timestamp'][()]).long(),
                }
            else:
                self.target[_id] = {
                    'length': torch.as_tensor(label_h5f[_id]['length'][()]).long(),
                    'timestamp': torch.from_numpy(label_h5f[_id]['timestamp'][()]).long(),
                }

    def load_feature(self):
        for part in ['a_feat_data', 'v_feat_data', 'l_feat_data']:
            setattr(self, part, {})
            for feature_name in getattr(self, part[0]+'_features'):
                h5f = h5py.File(os.path.join(self.root, 'feature', '{}.h5'.format(feature_name)), 'r')
                feature_data = {}
                for _id in self.seg_ids:
                    feature_data[_id] = h5f[self.set_name][_id]['feature'][()]
                    assert (h5f[self.set_name][_id]['timestamp'][()] == self.target[_id]['timestamp'].numpy()).all(), '\
                        Data Error: In feature {}, seg_id: {}, timestamp does not match label timestamp'.format(feature_name, _id)
                getattr(self, part)[feature_name] = feature_data

    def __getitem__(self, index):
        seg_id = self.seg_ids[index]
        a_features, v_features, l_features = [], [], []
       
        for feature_name in self.a_features:
            a_features.append(self.a_feat_data[feature_name][seg_id])
        a_features = torch.from_numpy(np.concatenate(a_features, axis=1)).float()

        for feature_name in self.v_features:
            v_features.append(self.v_feat_data[feature_name][seg_id])
        v_features = torch.from_numpy(np.concatenate(v_features, axis=1)).float()

        for feature_name in self.l_features:
            l_features.append(self.l_feat_data[feature_name][seg_id])
        l_features = torch.from_numpy(np.concatenate(l_features, axis=1)).float()
       

        target_data = self.target[seg_id]
        return {**{"a_feature": a_features.squeeze(),
                   "v_feature": v_features.squeeze(),
                   "l_feature": l_features.squeeze(),
                   "vid": seg_id},
                **target_data}
    
    def __len__(self):
        return len(self.seg_ids)
    
    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        a_feature = pad_sequence([sample['a_feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        v_feature = pad_sequence([sample['v_feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        l_feature = pad_sequence([sample['l_feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        timestamp = pad_sequence([sample['timestamp'] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
        length = torch.tensor([sample['length'] for sample in batch])
        vid = [sample['vid'] for sample in batch] 
        if self.set_name != 'tst':
            arousal = pad_sequence([sample['arousal'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            valence = pad_sequence([sample['valence'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        
        # make mask
        batch_size = length.size(0)
        batch_max_length = torch.max(length)
        mask = torch.zeros([batch_size, batch_max_length]).float()
        for i in range(batch_size):
            mask[i][:length[i]] = 1.0
        
        return {
            'a_feature': a_feature.float(), 
            'v_feature': v_feature.float(), 
            'l_feature': l_feature.float(), 
            'arousal': arousal.float(), 
            'valence': valence.float(),
            'mask': mask.float(),
            'length': length,
            'timestamp': timestamp,
            'vid': vid
        } if self.set_name != 'tst' else {
            'a_feature': a_feature.float(), 
            'v_feature': v_feature.float(), 
            'l_feature': l_feature.float(), 
            'mask': mask.float(),
            'length': length,
            'timestamp': timestamp,
            'vid': vid
        }

if __name__ == '__main__':
    class test:
        a_features = 'egemaps,vggish'
        v_features = 'vggface'
        l_features = 'bert_base_cover'
        dataroot = '/data7/lrc/MuSe2020/MuSe2020_features/wild/'
        max_seq_len = 100
    
    opt = test()
    a = MuseWildSplitDataset(opt, 'trn')
    iter_a = iter(a)
    data1 = next(iter_a)
    data2 = next(iter_a)
    data3 = next(iter_a)
    batch_data = a.collate_fn([data1, data2, data3])
    print(batch_data.keys())
    print(batch_data['a_feature'].shape)
    print(batch_data['v_feature'].shape)
    print(batch_data['l_feature'].shape)
    print(batch_data['arousal'].shape)
    print(batch_data['valence'].shape)
    print(batch_data['mask'].shape)
    print(batch_data['length'])
    print(torch.sum(batch_data['mask'][0]), torch.sum(batch_data['mask'][1]), torch.sum(batch_data['mask'][2]))
    print(batch_data['vid'])


