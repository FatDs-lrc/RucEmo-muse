import os
import h5py
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .base_dataset import BaseDataset


class MuseWildDataset(BaseDataset):
    def __init__(self, opt, set_name):
        ''' MuseWild dataset
        Parameter:
        --------------------------------------
        set_name: [trn, val, tst]
        '''
        super().__init__(opt)
        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
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
        self.feature_data = {}
        for feature_name in self.feature_set:
            h5f = h5py.File(os.path.join(self.root, 'feature', '{}.h5'.format(feature_name)), 'r')
            feature_data = {}
            for _id in self.seg_ids:
                feature_data[_id] = h5f[self.set_name][_id]['feature'][()]
                assert (h5f[self.set_name][_id]['timestamp'][()] == self.target[_id]['timestamp'].numpy()).all(), '\
                    Data Error: In feature {}, seg_id: {}, timestamp does not match label timestamp'.format(feature_name, _id)
            self.feature_data[feature_name] = feature_data

    def __getitem__(self, index):
        seg_id = self.seg_ids[index]
        feature_data = []
        feature_len = []
        for feature_name in self.feature_set:
            feature_data.append(self.feature_data[feature_name][seg_id])
            feature_len.append(self.feature_data[feature_name][seg_id].shape[1])
        feature_data = torch.from_numpy(np.concatenate(feature_data, axis=1)).float()
        feature_len = torch.from_numpy(np.array(feature_len)).long()

        target_data = self.target[seg_id]
        return {**{"feature": feature_data.squeeze(), "feature_lens": feature_len, "vid": seg_id},
                **target_data, **{"feature_names": self.feature_set}}
    
    def __len__(self):
        return len(self.seg_ids)
    
    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        feature = pad_sequence([sample['feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        timestamp = pad_sequence([sample['timestamp'] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
        length = torch.tensor([sample['length'] for sample in batch])
        vid = [sample['vid'] for sample in batch]

        if self.set_name != 'tst':
            arousal = pad_sequence([sample['arousal'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            valence = pad_sequence([sample['valence'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        
        feature_lens = batch[0]['feature_lens']
        feature_names = batch[0]['feature_names']
        # make mask
        batch_size = length.size(0)
        batch_max_length = torch.max(length)
        mask = torch.zeros([batch_size, batch_max_length]).float()
        for i in range(batch_size):
            mask[i][:length[i]] = 1.0
        
        return {
            'feature': feature.float(), 
            'arousal': arousal.float(), 
            'valence': valence.float(),
            'timestamp': timestamp.long(),
            'mask': mask.float(),
            'length': length,
            'feature_lens': feature_lens,
            'feature_names': feature_names,
            'vid': vid
        } if self.set_name != 'tst' else {
            'feature': feature.float(), 
            'timestamp': timestamp.long(),
            'mask': mask.float(),
            'length': length,
            'feature_lens': feature_lens,
            'feature_names': feature_names,
            'vid': vid
        }

if __name__ == '__main__':
    class test:
        feature_set = 'egemaps,au,fasttext'
        dataroot = '/data7/lrc/MuSe2020/MuSe2020_features/wild/'
        max_seq_len = 100
    
    opt = test()
    a = MuseWildDataset(opt, 'trn')
    iter_a = iter(a)
    data1 = next(iter_a)
    data2 = next(iter_a)
    data3 = next(iter_a)
    batch_data = a.collate_fn([data1, data2, data3])
    print(batch_data.keys())
    print(batch_data['feature'].shape)
    print(batch_data['arousal'].shape)
    print(batch_data['valence'].shape)
    print(batch_data['mask'].shape)
    print(batch_data['length'])
    print(torch.sum(batch_data['mask'][0]), torch.sum(batch_data['mask'][1]), torch.sum(batch_data['mask'][2]))
    print(batch_data['feature_names'])
    print(batch_data['feature_lens'])
    print(batch_data['vid'])
    # print(data['feature'].shape)
    # print(data['feature_lens'])
    # print(data['feature_names'])
    # print(data['length'])

