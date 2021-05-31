import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm

def make_partitial(input_csv, output_path):
    df = pd.read_csv(input_csv)
    h5f = h5py.File(output_path, 'w')
    trn = []
    val = []
    tst = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        _id, partition = row['Id'], row['Proposal']
        if partition == 'train':
            trn.append(_id)
        elif partition == 'devel':
            val.append(_id)
        elif partition == 'test':
            tst.append(_id)
        else:
            raise ValueError('Error with line: {}'.format(row))

    h5f['trn'] = trn
    h5f['val'] = val
    h5f['tst'] = tst

def get_csv_feature(csv_path):
    df = pd.read_csv(csv_path)
    timestamp = np.array(df['timestamp'])
    segment_id = np.array(df['segment_id'])
    feature = np.array(df.iloc[:, 2:])
    assert len(timestamp) == len(segment_id) == len(feature)
    return timestamp, segment_id, feature

def get_target(csv_path):
    df = pd.read_csv(csv_path)
    timestamp = np.array(df['timestamp'])
    segment_id = np.array(df['segment_id'])
    value = np.array(df['value'])
    assert len(timestamp) == len(segment_id) == len(value)
    return timestamp, segment_id, value

def make_target(config, target_types):
    '''
    config = config['c3_muse_stress']
    target_type: arousal or valence or anno12_EDA
    '''
    partition = h5py.File(osp.join(config['save_dir'], 'target', 'partition.h5'), 'r')
    for set_name in ['trn', 'val', 'tst']:
        h5f = h5py.File(osp.join(config['save_dir'], 'target', f'{set_name}_target.h5'), 'w')
        _ids = partition[set_name]
        for _id in _ids:
            group = h5f.create_group(str(_id))
            timestamp, segment_id = None, None
            for target_type in target_types:
                csv_path = osp.join(config['root'], 'label_segments', target_type, f'{_id}.csv')
                _timestamp, _segment_id, value = get_target(csv_path)
                if not isinstance(timestamp, np.ndarray):
                    timestamp = _timestamp
                else:
                    assert (timestamp == _timestamp).all()
                
                if not isinstance(segment_id, np.ndarray):
                    segment_id = _segment_id
                else:
                    assert (segment_id == _segment_id).all()
                
                group[target_type] = value
            
            group['timestamp'] = timestamp
            group['segment_id'] = segment_id
            group['length'] = len(timestamp)

def make_feature(config, feat_name):
    partition = h5py.File(osp.join(config['save_dir'], 'target', 'partition.h5'), 'r')
    h5f = h5py.File(osp.join(config['save_dir'], 'feature', f'{feat_name}.h5'), 'w')
    for set_name in ['trn', 'val', 'tst']:
        set_group = h5f.create_group(set_name)
        _ids = partition[set_name]
        for _id in tqdm(_ids, desc=set_name):
            group = set_group.create_group(str(_id))
            csv_path = osp.join(config['root'], 'feature_segments', feat_name, f'{_id}.csv')
            timestamp, _, feature = get_csv_feature(csv_path)
            group['timestamp'] = timestamp
            group['feature'] = feature
        

if __name__ == '__main__':
    all_config = json.load(open('path.json'))
    ''' c3_muse_stress '''
    config = all_config['c3_muse_stress']
    # save_dir = osp.join(config['save_dir'], 'target')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # make_partitial(
    #     osp.join(config['root'], 'metadata', 'partition.csv'), 
    #     osp.join(save_dir, 'partition.h5'), 
    # )

    save_dir = osp.join(config['save_dir'], 'feature')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    make_target(config, config['target'])
    # for feat_name in ['bert', 'BPM', 'deepspectrum', 'ECG', 'egemaps', 'fau_intensity', 'resp', 'vggface', 'vggish']:
    #     print('making', feat_name)
    #     make_feature(config, feat_name)

    ''' c4_muse_physio '''
    config = all_config['c4_muse_physio']
    # save_dir = osp.join(config['save_dir'], 'target')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # make_partitial(
    #     osp.join(config['root'], 'metadata', 'partition.csv'), 
    #     osp.join(save_dir, 'partition.h5'), 
    # )

    save_dir = osp.join(config['save_dir'], 'feature')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    make_target(config, config['target'])
    # for feat_name in ['bert', 'BPM', 'deepspectrum', 'ECG', 'egemaps', 'fau_intensity', 'resp', 'vggface', 'vggish']:
    #     print('making', feat_name)
    #     make_feature(config, feat_name)