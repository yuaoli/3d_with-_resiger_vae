'''
Author: tom
Date: 2024-10-28 15:27:29
LastEditors: Do not edit
LastEditTime: 2024-10-28 17:34:42
FilePath: /3d_resiger_vae/3d-knee-vae/experiments/custom.py
'''
import os
from torch.utils.data import Dataset
import numpy as np
from monai import transforms
from monai.data import Dataset as MonaiDataset
from monai.transforms import MapTransform
import nibabel as nib
from model.register_world_coordinate import create_nifti_dirs
import torch
import gc

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

def get_knee_file_paths(data_path):
    """仅保存文件路径，避免一次性加载数据"""
    images = []
    if os.path.isfile(data_path):
        images = [i for i in np.genfromtxt(data_path, dtype=np.str, encoding='utf-8')]
    else:
        assert os.path.isdir(data_path), '%s is not a valid directory' % data_path
        for root, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if 'SAG' in path:
                    sag_path = path
                    sag_dir = sag_path.rsplit('/', 1)[0]
                    obj = sag_path.split('/')[-1]
                    cor_dir = sag_dir.replace('SAG','COR')
                    for cor_file in os.listdir(cor_dir):
                        cor_path = os.path.join(cor_dir, cor_file)
                        images.append({"subject_id": obj, "sag_path": sag_path, "cor_path": cor_path})
    return images

class CustomBase(Dataset):
    def __init__(self, data_path,batch_size=1, device=None):
        super().__init__()
        self.data_paths = get_knee_file_paths(data_path)
        self.device = device

    def __len__(self):
        return len(self.data_paths)

    def load_dirs(self, path, device):
        """用于按需加载数据的辅助函数"""
        datum = create_nifti_dirs(path, force_v1=True, device=device)
        datum['images'] = datum['images'][None,...]
        return datum

    def __getitem__(self, idx):
        item_paths = self.data_paths[idx]
        sag_dir = self.load_dirs(item_paths['sag_path'],device=self.device)
        cor_dir = self.load_dirs(item_paths['cor_path'],device=self.device)
        return {
            "subject_id": item_paths['subject_id'],
            "sag_dirs": sag_dir,
            "cor_dirs": cor_dir
        }
    
    def clear_cache(self):
        """手动清理缓存"""
        gc.collect()
        torch.cuda.empty_cache()

class CustomTrain(CustomBase):
    def __init__(self, data_path, device, **kwargs):
        super().__init__(data_path=data_path,device=device)

class CustomTest(CustomBase):
    def __init__(self, data_path,batch_size, device, **kwargs):
        super().__init__(data_path=data_path,batch_size=1,device=device)
