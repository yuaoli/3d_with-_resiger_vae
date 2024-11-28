import logging
from typing import Sequence, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .transforms_cuda_det import build_cuda_transform
# from .nnunet import UNet3D, UpsampleBlock, ConvBlock
# from timm.utils.metric_seg import compute_stat_from_pred_gt
from .vae import Register_VAE
from .register import image_register,get_T_natural_to_torch,get_T_torch_to_natural
from .extract_bone_contusion_dualview_3d_patches import DualSegExtractor
from .register_world_coordinate import DummySeries
import nibabel as nib
import numpy as np
import os
from monai.networks.blocks import SimpleASPP

try:
    from .registry import register_model
except:
    register_model = lambda x: x


logger = logging.getLogger('train')

def save_nifti(pixel_data, affine, fpath):
    pixel_data = np.transpose(pixel_data, (2, 1, 0))  # WHD
    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nifti_img, fpath)

def save_vis(img,aff,path,name):
    img_arry = img[0,0,...].detach().cpu().numpy()
    aff = aff[0,...].detach().cpu().numpy()
    # aff[0] *= -1
    # aff[1] *= -1  # to affine itk
    path = os.path.join(path,name)
    save_nifti(img_arry,aff,path)
    

class FeatureRegistration(nn.Module):
    def __init__(self, spatial_dim1, spatial_dim2, T1, T2, feat_dim, use_out_embedding=True):
        super(FeatureRegistration, self).__init__()
        self.T1 = T1 # d = 32
        self.T2 = T2 # d
        self.feat_dim = feat_dim
        self.spatial_dim1 = spatial_dim1
        self.spatial_dim2 = spatial_dim2
        self.use_out_embedding = use_out_embedding
        if use_out_embedding:
            self.out_fov_embedding = nn.Embedding(1, feat_dim)

        self.aspp_1 = SimpleASPP(3,in_channels=2,conv_out_channels=1)#,kernel_sizes=[1,3],dilations=[1,2]
        self.aspp_2 = SimpleASPP(3,in_channels=4,conv_out_channels=1)

        self.conv_1 = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1)
        self.up_1 = nn.Sequential(nn.Conv3d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
                                  nn.Upsample(scale_factor=(1,2,2), mode='nearest'))
        self.upconv1 = nn.Conv3d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_2 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.up_2 = nn.Sequential(nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
                                  nn.Upsample(scale_factor=(1,2,2), mode='nearest'))
        self.upconv2 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True)

        # self.pixel_shuffle = nn.PixelShuffle(2)
        self.deconv_1 = nn.ConvTranspose3d(2, 2, kernel_size=4, stride=2, padding=1)
        self.deconv_2 = nn.ConvTranspose3d(4, 4, kernel_size=4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    @staticmethod
    def _transform_affine_to_feature(feats, affine, spatial_dim):   #new_affine
        feat_shape = feats.shape[-3:]
        scale_f = [cdim / rdim for cdim, rdim in zip(spatial_dim, feat_shape)]
        T = torch.diag(torch.tensor(scale_f[::-1] + [1.])).to(affine)  # no scaling on z
        return affine @ T

    @staticmethod
    def _get_t_natural_to_torch(feats):
        shape = feats.shape[-3:]
        T = torch.eye(4)
        for i, dim in enumerate(shape[::-1]):
            if dim == 1:
                T[i, :] = 0
            else:
                T[i, i] = 2 / (dim - 1)
                T[i, -1] = -1
        return T.to(feats)

    @staticmethod
    def _get_t_torch_to_natural(feats):
        shape = feats.shape[-3:]
        T = torch.eye(4)
        for i, dim in enumerate(shape[::-1]):
            # warning: not appliable for dim == 1
            T[i, i] = (dim - 1) / 2
            T[i, -1] = (dim - 1) / 2
        return T.to(feats)

    # @profile
    def _regist_feats2_to_feats1(self, feats1, affine1, feats2, affine2, align_corners=False,aspp=True):
        affine1, affine2 = affine1.float(), affine2.float()
        inv_affine2 = torch.linalg.inv(affine2)


        if aspp:
            feats = feats2.clone()
            # aspp
            if feats.shape[1] == 2:
                feats1 = feats1.repeat(1,2,1,1,1)
                feats2 = feats2.repeat(1,2,1,1,1)
                theta = self._get_t_natural_to_torch(feats2) @ inv_affine2 @ affine1 @ self._get_t_torch_to_natural(feats1)
                theta = theta[:, :3]
                grid = F.affine_grid(theta, feats1.shape, align_corners=align_corners)
                feats = self.aspp_1(feats) #[1 2 32 192 192]->[1 4 32 192 192]

            elif feats.shape[1] == 4:
                theta = self._get_t_natural_to_torch(feats2) @ inv_affine2 @ affine1 @ self._get_t_torch_to_natural(feats1)
                theta = theta[:, :3]
                grid = F.affine_grid(theta, feats1.shape, align_corners=align_corners)
                feats = self.aspp_2(feats) #[1 4 32 96 96]->[1 4 32 96 96]

            feats = F.grid_sample(feats, grid, mode="bilinear", align_corners=align_corners) #[1 4 32 192 192] [1 8 32 96 96]

    
        else:
            theta = self._get_t_natural_to_torch(feats2) @ inv_affine2 @ affine1 @ self._get_t_torch_to_natural(feats1)
            theta = theta[:, :3]
            grid = F.affine_grid(theta, feats1.shape, align_corners=align_corners)
            feats = F.grid_sample(feats2, grid, mode="bilinear", align_corners=align_corners)


        if self.use_out_embedding:
            in_fov = F.grid_sample(torch.ones_like(feats2[:, :1]), grid, mode="bilinear", align_corners=align_corners)
            # in_fov 的值在视野内的区域接近 1，在视野外的区域接近 0
            # print((in_fov == 1).float().mean(dim=(4, 3))[:, 0])
            if not aspp: #mask
                binary_mask = (in_fov > 0.5).float()
                feats = None
            else:
                binary_mask = None
                feats = self.out_fov_embedding.weight[:, :, None, None, None].to(feats1.device) * (1 - in_fov) + in_fov * feats  # todo: use expand dims

        del inv_affine2, theta, grid
        return feats,binary_mask
    
    # @profile
    def forward(self, feats1, affine1, feats2, affine2,aspp=True):
        affine1 = self._transform_affine_to_feature(feats1, affine1, (self.T1,) + self.spatial_dim1)
        affine2 = self._transform_affine_to_feature(feats2, affine2, (self.T2,) + self.spatial_dim2)
        feats2_to_1,mask= self._regist_feats2_to_feats1(feats1, affine1, feats2, affine2, True, aspp=aspp)  # seems to align better

        return feats2_to_1,mask,affine1,affine2




class DualViewSegNet(nn.Module):
    def __init__(self, n_channels, gf_dim, lat_ch1, lat_ch2, aux_seg=True, single_view=False, ct_eval=False, ld=False, encoder_pre=False, decoder_pre=False):
        super(DualViewSegNet, self).__init__()
        self.aux_seg = aux_seg
        self.ct_eval = ct_eval
        self.ld = ld

        self.filters = [lat_ch1,lat_ch2]
        self.feat_regist = []
        self.feat_regist.append(FeatureRegistration((384, 384), (384, 384), 32, 32, self.filters[0]*2, use_out_embedding=True).to('cuda'))  #[（ori_w,ori_h）,（ori_w,ori_h）,ori_d,ori_d]
        self.feat_regist.append(FeatureRegistration((384, 384), (384, 384), 32, 32, self.filters[1], use_out_embedding=True).to('cuda'))
        
        self.net_depth = 2 #384-->96 步长为2
        self.net = Register_VAE(n_channels, gf_dim,self.net_depth, self.feat_regist, lightingdecoder=self.ld, encoder_pre= encoder_pre, decoder_pre=decoder_pre)


    # @profile
    def forward(self, x, device, save_resize_vis=None,save_path=None):
        sag_series = DummySeries2(x['sag_dirs'],device)
        cor_series = DummySeries2(x['cor_dirs'],device)
        x1_dict,x2_dict = self.net(sag_series, cor_series, device, save_resize_vis, save_path)
        return x1_dict, x2_dict
    
class DummySeries2(object):
    def __init__(self, dic, device):
        for key in dic:
            setattr(self, key, dic[key].float().to(device))
    def clone(self):
        # 创建当前对象的副本
        return DummySeries2(self.__dict__,self.images.device)



