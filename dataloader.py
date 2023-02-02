import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append('./preprocess')
from generate_dataset import generate_dataset

# get csv file from osmnx (datasets/cities/original에 원본 city csv 파일이 없을 경우)
make_original = False

# pre_process (self-loop, multi-edge 제거 및 node 좌표계 변환)
pre_process = False

# transform to dgl
make_dgl = True
generate_dataset(make_original, pre_process, make_dgl)

# data

class RoadNetworkDataset(Dataset):
    def __init__(self, phase, data_dir):
        if phase=='train':
            data_npz_path=os.path.join(data_dir, 'train_dataset.npz')
        else: # test
            data_npz_path=os.path.join(data_dir, 'test_dataset.npz')

        Rdataset=np.load(data_npz_path, allow_pickle=True)
        self.node=Rdataset["node"] # example!!!
        """
        implementation for self.edge, self.vertex, ...
        """
        data_norm_dir=os.path.join(data_dir, 'norm')
        
        if Rdataset is None:
            assert Rdataset, 'Rdataset is None'

    def __len__(self):
        return len(self.node)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index=index.tolist()