import torch
from torch import nn, optim
from .register import image_register,get_T_natural_to_torch,get_T_torch_to_natural,DummySeries
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import time
import SimpleITK as sitk
import torchvision
# from .DenseASPP import DenseASPP
# from .DenseASPP121 import Model_CFG
import gc

def save_nifti(pixel_data, affine, fpath):
    pixel_data = np.transpose(pixel_data, (2, 1, 0))  # WHD
    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nifti_img, fpath)

# def save_vis(img,aff,path,name):
#     img_arry = img[0,0,...].detach().cpu().numpy()
#     aff = aff[0,...].detach().cpu().numpy()
#     # aff[0] *= -1
#     # aff[1] *= -1  # to affine itk
#     path = os.path.join(path,name)
#     save_nifti(img_arry,aff,path)

def save_vis(img,aff=None,path=None):
    if aff is not None:
        img_arry = img[0,0,...].detach().cpu().numpy()
        aff = aff.detach().cpu().numpy()
        # aff[0] *= -1
        # aff[1] *= -1  # to affine itk
        path = path
        save_nifti(img_arry,aff,path)
    else:
        img_arry = img[0,0,...].detach().cpu().numpy()
        path = path
        img = sitk.GetImageFromArray(img_arry)
        sitk.WriteImage(img,path)

# def save_vis(tensor, path):
#     # 将 PyTorch 张量转换为 NumPy 数组，并确保维度顺序为 (depth, height, width)
#     np_array = tensor.cpu().numpy()[0, 0]
#     print(np_array.shape)

#     # 将 NumPy 数组转换为 SimpleITK 图像
#     sitk_image = sitk.GetImageFromArray(np_array)

#     # 保存为 NIfTI 文件
#     sitk.WriteImage(sitk_image, path)


class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockUp, self).__init__()
        self.act = act
        self.conv1_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_in, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=(1,2,2), mode='nearest'),
            nn.BatchNorm3d(filters_in),
            nn.LeakyReLU(0.2, inplace=False))

        self.conv2_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm3d(filters_out))

        self.conv3_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=(1,2,2), mode='nearest'),
            nn.BatchNorm3d(filters_out),
            nn.LeakyReLU(0.2, inplace=False))

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)

        return conv2 + conv3
    
class ResBlockMid(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockMid, self).__init__()
        self.act = act
        self.conv1_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm3d(filters_out),
            nn.LeakyReLU(0.2, inplace=False))
        self.conv2_block = nn.Sequential(
            nn.Conv3d(filters_out, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm3d(filters_out),
            nn.LeakyReLU(0.2, inplace=False))
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        
    def forward(self, x):
        conv1 = self.conv1_block(x)
        if self.act:
            conv1 = self.lrelu(conv1)        
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        return conv1 + conv2
        


class ResBlockDown(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockDown, self).__init__()
        self.act = act
        self.conv1_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_in, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(filters_in),
            nn.LeakyReLU(0.2, inplace=False))

        self.conv2_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm3d(filters_out))

        self.conv3_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_out, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(filters_out)
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)

        return conv2 + conv3


class Encoder(nn.Module):
    def __init__(self, n_channels, gf_dim=16):
        super(Encoder, self).__init__()

        self.conv1_block = nn.Sequential(
            nn.Conv3d(n_channels, gf_dim, 3, stride=1, padding=1),
            nn.BatchNorm3d(gf_dim),
            nn.LeakyReLU(0.2, inplace=False))

        self.res1 = ResBlockDown(gf_dim * 1, gf_dim * 1) #2 2
        self.res2 = ResBlockDown(gf_dim * 1, gf_dim * 2) #1 4

        self.res2_mu = ResBlockDown(gf_dim * 2, gf_dim * 2) #2 4
        self.res2_stdev = ResBlockDown(gf_dim * 2, gf_dim * 2, act=False)  


    def encode(self, x):#[1 1 32 384 384]

        conv1 = self.conv1_block(x) #[1 2 32 384 384]
        z = self.res1(conv1)        #[1 2 32 192 192]        
        z_mean = self.res2(z)       #[1, 4, 32, 96, 96]
        z_std = self.res2_stdev(z)  #[1, 4, 32, 96, 96]

        return z_mean, z_std

    def reparameterize(self, mu, std):
        std = torch.exp(std)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        return z, mu, std
    
    def encode_1(self, x):#[1 1 32 384 384]
        conv1 = self.conv1_block(x) #[1 2 32 384 384]
        z = self.res1(conv1)        #[1 2 32 192 192]
        z = self.res2(z)        
        return z

    def encode_2(self, z):#[1 1 32 384 384]
        z_mean = self.res2_mu(z)       #[1, 4, 32, 96, 96]
        z_std = self.res2_stdev(z)  #[1, 4, 32, 96, 96]
        z =  self.reparameterize(z_mean, z_std)
        return z,z_mean,z_std


class Decoder(nn.Module):
    def __init__(self, n_channels, gf_dim=16):
        super(Decoder, self).__init__()

        self.res5 = ResBlockUp(gf_dim*2, gf_dim * 1)
        self.res6 = ResBlockUp(gf_dim*1, gf_dim * 1)
        self.conv_1_block = nn.Sequential(
            nn.Conv3d(gf_dim, gf_dim, 3, stride=1, padding=1),
            nn.BatchNorm3d(gf_dim),
            nn.LeakyReLU(0.2, inplace=False))#inplace=true

        self.out = nn.Conv3d(gf_dim, n_channels, 3, stride=1, padding=1)

    def forward(self, z): #[1, 4, 32, 96, 96]

        x = self.res5(z) #[1, 2, 32, 192, 192]
        x = self.res6(x)#[1, 2, 32, 384, 384]
        x = self.conv_1_block(x)
        x = self.out(x)#[1, 1, 32, 384, 384]

        return x
    
class LightingDecoder(nn.Module):
    def __init__(self, n_channels, gf_dim=16):
        super(LightingDecoder, self).__init__()

        self.up = nn.Sequential(
            nn.Conv3d(gf_dim*2, gf_dim, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=(1,4,4), mode='nearest'),
            nn.BatchNorm3d(gf_dim),
            nn.ReLU(0.2))
        self.out = nn.Conv3d(gf_dim, n_channels, 3, stride=1, padding=1)

    def forward(self, z): #[1, 4, 32, 96, 96]
        x = self.up(z) #[1, 2, 32, 384, 384]
        x = self.out(x)#[1, 1, 32, 384, 384]
        return x
    
class Midblock(nn.Module):
    def __init__(self, n_channels, gf_dim=16):
        super(Midblock, self).__init__()

        # gf_dim = int(gf_dim/2)
        self.res0 = ResBlockMid(gf_dim*2, gf_dim * 2)
        self.res1 = ResBlockMid(gf_dim*2, gf_dim * 2)
        self.conv_1_block = nn.Sequential(
            nn.Conv3d(gf_dim*2, gf_dim*2, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=(1,2,2), mode='nearest'),
            nn.BatchNorm3d(gf_dim*2),
            nn.LeakyReLU(0.2, inplace=False))

    def forward(self, z): #[1, 4, 32, 96, 96]
        x = self.res0(z) #[1, 4, 32, 96, 96]
        x = self.res1(x)#[1, 4, 32, 96, 96]
        x = self.conv_1_block(x)
        return x #[1, 4, 32, 192, 192]

    
class Register_VAE(nn.Module):
    def __init__(self, n_channels, gf_dim=16, depth=2,feat_regist=None,mask_regist=None,lightingdecoder=False, encoder_pre=False, decoder_pre=False):
        super(Register_VAE, self).__init__()
        self.encoder = Encoder(n_channels, gf_dim)
        if not lightingdecoder:
            self.decoder = Decoder(n_channels, gf_dim)
        else:
            self.decoder = LightingDecoder(n_channels, gf_dim)
        if encoder_pre:
            print(f"Loaded pretrained encoder weights ...")
            self.encoder.eval()
        if decoder_pre:
            print(f"Loaded pretrained decoder weights ...")
            self.decoder.eval()

        self.midblock = Midblock(n_channels, gf_dim)
        self.feat_regist = feat_regist
        self.mask_regist = mask_regist
        self.latloss = nn.MSELoss(reduction='mean')


    # @profile
    def forward(self,series1_, series2_, device=None ,save_resize_vis=False,save_path=None ):#(x1--原仿射矩阵,x2--目标仿射矩阵) #im1, aff1, im2, aff2
        register = True
        dic_list = []
        
        for series1,series2 in [(series1_,series2_),(series2_,series1_)]:  #,(series2_,series1_)

            x1 = series1.images.to(device) #[1 1 32 384 384]
            aff1 = series1.affine.to(device)
            assert x1.shape[0] == 1,f'x1.shape[0] must 1!'

            x1_latent = self.encoder.encode_1(x1) #[1 4 32 96 96] downsample
            m1_latent = torch.ones_like(x1_latent)  #因为全是1 所以可视化的时候会是黑的 这个时候只需要令某些值为0 即可显示出来
        
            # =======vae_w_register=======
            if register:

                feat_regist = self.feat_regist[0]

                x2 = series2.images.to(device)
                aff2 = series2.affine.to(device)


                x2_latent = self.encoder.encode_1(x2) # downsample
                m2_latent = torch.ones_like(x2_latent)

                x1tox2_latent,_,_ = feat_regist(x2_latent, aff2, x1_latent, aff1) #[1 4 32 96 96] # 1 --> 2  #糊的原因就是在r的地方丢失了太多东西
                m1tom2_latent,_,_ = feat_regist(m2_latent, aff2, m1_latent, aff1, aspp=False) #[1 1 32 192 192]

                x1tox2_latent = self.midblock(x1tox2_latent)#[1 4 32 192 192] midblock
                x1tox2_latent,mu1tomu2,std1tostd2 = self.encoder.encode_2(x1tox2_latent) #[1 4 32 96 96] 重参数 这个是最终想要得到的


                feat_regist = self.feat_regist[0]
                x1_latent_new,_,_ = feat_regist(x1_latent, aff1, x1tox2_latent, aff2)
                m1_latent_new,_,_ = feat_regist(m1_latent, aff1, m1tom2_latent, aff2, aspp=False) #因为feat_regist中有aspp所以 m1也不是全1了,但是m2没有经过aspp所以还全是1
                m1_new = F.interpolate(m1_latent_new[0:1,0:1,...], size=x1.shape[2:], mode='nearest')
                m2_new = F.interpolate(m1tom2_latent[0:1,0:1,...], size=x2.shape[2:], mode='nearest')
                m2 = F.interpolate(m2_latent[0:1,0:1,...], size=x2.shape[2:], mode='nearest')

                x1_reconstruction = self.decoder(x1_latent_new)
                x2_reconstruction = self.decoder(x1tox2_latent)


                # lat_loss_x1 = self.latloss(x1_latent_new * m1_latent_new, x1_latent.detach() * m1_latent_new)
                # lat_loss_x2 = self.latloss(x1tox2_latent * m1tom2_latent, x2_latent.detach() * m1tom2_latent)


                dic = { 
                        'x1_mu':None,
                        'x1_std':None, 
                        'mu1tomu2':mu1tomu2,
                        'std1tostd2':std1tostd2, 

                        'x1_reconstruction':x1_reconstruction * m1_new,
                        'aff1':aff1,
                        'x1_input':x1 * m1_new, #
                        'x1_input_ori':x1,


                        'x2_reconstruction':x2_reconstruction * m2_new,
                        'aff2':aff2,
                        'x2_input':x2 * m2_new,
                        'x2_input_ori':x2,

                        # 'lat_loss_x1':lat_loss_x1,
                        'lat_x1':x1_latent.detach() * m1_latent_new,
                        'lat_new_x1':x1_latent_new * m1_latent_new,

                        # 'lat_loss_x2':lat_loss_x2,
                        'lat_x2':x2_latent.detach() * m1tom2_latent,
                        'lat_x1tox2':x1tox2_latent * m1tom2_latent

                        }
                dic_list.append(dic)
                del x1, x2, x1_latent, x2_latent, x1tox2_latent, x1_latent_new, m1_latent_new, m1tom2_latent
                del series1, series2, m1_new, m2_new, m2, feat_regist, x1_reconstruction, x2_reconstruction
                # gc.collect()
                

            # =======vae_wo_register=======
            else:            
                x1_reconstruction = self.decoder(x1_latent)
                return x1_mu, x1_std, x1_reconstruction,aff1,x1

        if len (dic_list)==1:
            dic_list.append(dic)
        assert len (dic_list)==2
        return dic_list
