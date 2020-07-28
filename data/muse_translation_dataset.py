import os
import h5py
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from base_dataset import BaseDataset


class MuseTranslationDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--input_feature', type=str, default='None', help='audio feature to use, split by comma, eg: "egemaps,vggish"')
        parser.add_argument('--output_feature', type=str, default='None', help='visual feature to use, split by comma, eg: "vggface"')
        return parser
    
    def __init__(self, opt, set_name):
        ''' MuseWild dataset
        Parameter:
        --------------------------------------
        set_name: [trn, val, tst]
        '''
        super().__init__(opt)
        self.in_features = list(map(lambda x: x.strip(), opt.in_features.split(',')))
        self.out_features = list(map(lambda x: x.strip(), opt.out_features.split(',')))
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
        for part in ['in_feat_data', 'out_feat_data']:
            setattr(self, part, {})
            for feature_name in getattr(self, part.split('_')[0]+'_features'):
                h5f = h5py.File(os.path.join(self.root, 'feature', '{}.h5'.format(feature_name)), 'r')
                feature_data = {}
                for _id in self.seg_ids:
                    feature_data[_id] = h5f[self.set_name][_id]['feature'][()]
                    assert (h5f[self.set_name][_id]['timestamp'][()] == self.target[_id]['timestamp'].numpy()).all(), '\
                        Data Error: In feature {}, seg_id: {}, timestamp does not match label timestamp'.format(feature_name, _id)
                getattr(self, part)[feature_name] = feature_data

    def __getitem__(self, index):
        seg_id = self.seg_ids[index]
        in_features, out_features = [], []
       
        for feature_name in self.in_features:
            in_features.append(self.in_feat_data[feature_name][seg_id])
        in_features = torch.from_numpy(np.concatenate(in_features, axis=1)).float()

        for feature_name in self.out_features:
            out_features.append(self.out_feat_data[feature_name][seg_id])
        out_features = torch.from_numpy(np.concatenate(out_features, axis=1)).float()

        target_data = self.target[seg_id]
        return {**{"in_feature": in_features.squeeze(),
                   "out_feature": out_features.squeeze(),
                   "vid": seg_id},
                **target_data}
    
    def __len__(self):
        return len(self.seg_ids)
    
    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        in_feature = pad_sequence([sample['in_feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        out_feature = pad_sequence([sample['out_feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
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
            'in_feature': in_feature.float(), 
            'out_feature': out_feature.float(), 
            'arousal': arousal.float(), 
            'valence': valence.float(),
            'mask': mask.float(),
            'length': length,
            'timestamp': timestamp,
            'vid': vid
        } if self.set_name != 'tst' else {
            'in_feature': in_feature.float(), 
            'out_feature': out_feature.float(), 
            'mask': mask.float(),
            'length': length,
            'timestamp': timestamp,
            'vid': vid
        }

if __name__ == '__main__':
    class test:
        in_features = 'egemaps,vggish'
        out_features = 'vggface'
        dataroot = 'dataset/wild'
        max_seq_len = 100
    
    opt = test()
    a = MuseTranslationDataset(opt, 'trn')
    iter_a = iter(a)
    data1 = next(iter_a)
    data2 = next(iter_a)
    data3 = next(iter_a)
    batch_data = a.collate_fn([data1, data2, data3])
    print(batch_data.keys())
    print(batch_data['in_feature'].shape)
    print(batch_data['out_feature'].shape)
    print(batch_data['arousal'].shape)
    print(batch_data['valence'].shape)
    print(batch_data['mask'].shape)
    print(batch_data['length'])
    print(torch.sum(batch_data['mask'][0]), torch.sum(batch_data['mask'][1]), torch.sum(batch_data['mask'][2]))
    print(batch_data['vid'])


