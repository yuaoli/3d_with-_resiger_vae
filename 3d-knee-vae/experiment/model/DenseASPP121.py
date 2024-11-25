'''
Author: tom
Date: 2024-11-15 22:27:51
LastEditors: Do not edit
LastEditTime: 2024-11-18 14:51:30
FilePath: /3d_resiger_vae2/3d-knee-vae/experiments/model/DenseASPP121.py
'''
Model_CFG = {
    'bn_size': 4,
    'drop_rate': 0,
    'growth_rate': 2,  #32
    'num_init_features': 4, #64
    'block_config': (4, 8, 16, 8), #(6, 12, 24, 16),  #二倍

    'dropout0': 0.1,
    'dropout1': 0.1,
    'd_feature0': 8, #128
    'd_feature1': 4, #64

    'pretrained_path': "./pretrained/densenet121.pth"
}