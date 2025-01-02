__author__ = 'aoao'

import os
import numpy as np
import io
import glob
from PIL import Image
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
from torch.utils import tensorboard
import torchvision
import functools
import matplotlib
import collections
from custom import CustomTrain,CustomTest
import math
import time
from datetime import datetime

from model.dual_view_vae import DualViewSegNet

from model.register import image_register
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm
import gc
from line_profiler import LineProfiler

import sys
sys.path.append('/mnt/users/3d_resiger_vae/taming-transformers')
from taming.modules.losses.contperceptual import LPIPSWithDiscriminator
# from taming.modules.discriminator.model import NLayerDiscriminator3D
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

# 用于多GPU训练的必要导入
import torch.distributed as dist
import torch.multiprocessing as mp

import signal

matplotlib.use('Agg')


np.seterr(all='raise')
np.random.seed(2019)
torch.manual_seed(2019)

profiler = LineProfiler()
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    profiler.dump_stats('main_experiment_256.py.lprof')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save_epoch_interval', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--val_epoch_interval', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--latent-dimension', type=int, default=256, metavar='N',
                    help=' ')
parser.add_argument('--n-channels', type=int, default=1, metavar='N', #n_channels
                    help=' ')
parser.add_argument('--img-size', type=int, default=256, metavar='N',
                    help=' ')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='N',
                    help=' ')
parser.add_argument('--architecture', type=str, default='old', metavar='N',
                    help=' ')
parser.add_argument('--reconstruction-data-loss-weight', type=float, default=1,
                    help=' ')
parser.add_argument('--kl-latent-loss-weight', type=float, default=0.00001,
                    help=' ')
parser.add_argument("--train_data_dir", type = str, 
                    default ="your_train_data_path",
                    help = "the data directory")
parser.add_argument("--test_data_dir", type = str, 
                    default ="your_test_data_path",
                    help = "the data directory")

def remove_module_prefix(state_dict):
    """
    去掉 state_dict 中的 'module.' 前缀
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
        else:
            new_state_dict[k] = v
    return new_state_dict

def remove_module_prefix2(state_dict):
    """
    Remove 'net.' prefix from the state_dict if it exists.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('net.encoder.'):
            new_key = key[len('net.encoder.'):]  # 去除 'net.encoder.' 前缀
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict



def save_nifti(pixel_data, affine, fpath):
    pixel_data = np.transpose(pixel_data, (2, 1, 0))  # WHD
    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nifti_img, fpath)

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


def init_process(rank, world_size, fn, args):

    os.environ['MASTER_ADDR'] = 'localhost'  # 或者你的主节点IP
    os.environ['MASTER_PORT'] = '12355'       # 选择一个未被占用的端口

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    fn(rank, world_size, *args)


def make_grid_3d(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Make a grid of 3D images.
    
    Args:
        tensor (Tensor): 5D mini-batch Tensor of shape (B x C x H x W x D).
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default is 8.
        padding (int, optional): Amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default is False.
        value_range (tuple, optional): Tuple (min, max) where min and max are numbers.
            These numbers are used to normalize the image. By default, min and max are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of images separately
            rather than (min, max) over all images. Default is False.
        pad_value (float, optional): Value for the padded pixels. Default is 0.
    
    Returns:
        grid (Tensor): The tensor containing grid of images.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)

    if tensor.dim() != 5:
        raise ValueError('Expected 5D tensor, got {}D tensor'.format(tensor.dim()))

    if normalize:
        tensor = tensor.clone()  # Avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError('value_range has to be a tuple (min, max) if specified. min and max are numbers')

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each:
            for t in tensor:  # Loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    # Make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))

    height, width, depth = tensor.size(2), tensor.size(3), tensor.size(4)
    grid = tensor.new_full((tensor.size(1), height * ymaps + padding, width * xmaps + padding, depth), pad_value)

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * (height + padding), height).narrow(
                2, x * (width + padding), width
            ).copy_(tensor[k])
            k += 1
    return grid



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


# Reconstruction + KL divergence losses summed over all elements and batch
reconstruction_loss = nn.MSELoss(reduction='mean')#sum #非负


# def KLLoss(z_mean, z_log_sigma):
#     return -0.5 * torch.mean(   #sum
#         1 + 2 * z_log_sigma - torch.pow(z_mean, 2) -
#         torch.exp(2 * z_log_sigma))

def KLLoss(z_mean, z_log_sigma):
    # 计算 KL 散度
    loss = -0.5 * torch.mean(
        1 + 2 * z_log_sigma - torch.pow(z_mean, 2) - torch.exp(2 * z_log_sigma)
    )
    return loss  # 这是一个标量值




def add_grid(writer, images, name, step,
             batch_size=32, n_channels=1, img_size=128):
    _,_,h,w,d = images.shape
    grid = make_grid_3d(
        images.view(batch_size, n_channels, img_size, img_size, d),
        nrow=1,
        normalize=True,
        value_range=(-1, 1))
    
    d = grid.shape[-1]
    mid_d = d//2

    writer.add_image(name, grid[:, :, :, mid_d], step)


def add_detailed_summaries(writer, decoder, phase, data, reconstruction, latent,
                           step,
                           batch_size=32,
                           n_channels=1,
                           img_size=128):
    add_grid(writer, data, 'Data/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)
    add_grid(writer, reconstruction, 'Reconstruction/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)

    zs = torch.randn_like(latent)
    samples = decoder(zs)
    add_grid(writer, samples, 'Samples/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)

# @profile
def train(model, optimizer_vae,optimizer_disc, epoch, step, train_loader, Discriminator, writer,
          log_interval=1000,
          rank = 0,
          results_dir = None,
          ddp = False):
    
    rank = torch.device(f'cuda:{rank}')

    model.train()
    Discriminator.train()

    def get_last_layer():
        if ddp:
            return model.module.net.decoder.out.weight
        else:
            return model.net.decoder.out.weight
        
    train_vae_loss_sum = 0  #对走过的所有步数计算平均值
    train_disc_loss_sum = 0 
    for i, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):

        # if i == 19:
        #     signal_handler(signal.SIGINT, None)

        
        sagtocor_dict,cortosag_dict = model(batch,rank)#[1 16 32 4 32] #x1_mu, x1_std,x1_reconstruction


        for dic in (sagtocor_dict,cortosag_dict):   #

            optimizer_vae.zero_grad()
            optimizer_disc.zero_grad()

            vae_loss_3d = 0 #每对数据算一下 一组数据算两次

            #x1
            x1_rec_loss = 0
            x1_kl_loss = 0              
            x1_reconstruction = dic['x1_reconstruction']
            x1_input = dic['x1_input']
            # x1_mu = dic['x1_mu']
            # x1_std = dic['x1_std']
            x1_rec_loss = reconstruction_loss(x1_reconstruction,x1_input)
            # x1_kl_loss += KLLoss(x1_mu, x1_std) * 0.5     #这个是不是也可以设置一下开始步长，最初让先稳定一下encode
            x1_kl_loss = 0.

            #x2
            x2_rec_loss = 0
            x2_kl_loss = 0  
            x2_reconstruction = dic['x2_reconstruction']
            x2_input = dic['x2_input']
            x2_mu = dic['mu1tomu2']
            x2_std = dic['std1tostd2']
            x2_rec_loss = reconstruction_loss(x2_reconstruction,x2_input)
            x2_kl_loss = KLLoss(x2_mu, x2_std)

            #lat_recloss
            
            lat_x1 = dic['lat_x1']
            lat_new_x1 = dic['lat_new_x1']
            dic['lat_loss_x1'] = reconstruction_loss(lat_x1,lat_new_x1)
            lat_x2 = dic['lat_x2']
            lat_new_x2 = dic['lat_x1tox2']
            dic['lat_loss_x2'] = reconstruction_loss(lat_x2,lat_new_x2)
            lat_rec_loss =  0.5 * dic['lat_loss_x1']+ dic['lat_loss_x2']

            rec_loss = x1_rec_loss  + lat_rec_loss  #+ x2_rec_loss
            kl_loss = x1_kl_loss + x2_kl_loss
            vae_loss_3d = rec_loss * args.reconstruction_data_loss_weight + kl_loss * args.kl_latent_loss_weight

            img_dis = True

            #img_disloss
            if img_dis:
                vae_losses_2d = []
                disc_losses = []
                B,C,D,H,W = x1_input.shape
                for input, reconstruction in (x1_input, x1_reconstruction),(x2_input, x2_reconstruction):  #两个重构回来的都计算gan_loss
                    input_2d_ = input.reshape(B*D,C,H,W)
                    reconstruction_2d_ = reconstruction.reshape(B*D,C,H,W)
                    gap = 4
                    for b in range(0,B*D,gap):
                        input_2d = input_2d_[b:b+1,:,:,:]
                        reconstruction_2d = reconstruction_2d_[b:b+1,:,:,:] #[:,:,s,:,:]

                        try:
                            loss_vae_2d, log_vae_2d = Discriminator(input_2d,
                                                                reconstruction_2d,
                                                                posteriors = DiagonalGaussianDistribution(x2_mu, x2_std), 
                                                                optimizer_idx=0, 
                                                                global_step = step,
                                                                last_layer=get_last_layer(),
                                                                )
                            loss_disc, log_disc = Discriminator(input_2d.detach(),
                                                                reconstruction_2d.detach(),
                                                                posteriors = DiagonalGaussianDistribution(x2_mu, x2_std),
                                                                optimizer_idx=1,
                                                                global_step = step,
                                                                last_layer=get_last_layer(),)
                        
                            vae_losses_2d.append(loss_vae_2d)
                            disc_losses.append(loss_disc)

                        except RuntimeError as e:
                            print(f"Error in batch {i}, sub-batch {b}: {e}")
                            continue
            #lat_disloss
            else:
                vae_losses_2d = []
                disc_losses = []
                B,C,D,H,W = lat_x2.shape
                for input, reconstruction in [(lat_x2, lat_new_x2),]:  #两个重构回来的都计算gan_loss
                    input_2d_ = torch.mean(input.reshape(B*D,C,H,W),dim=1,keepdim=True)
                    reconstruction_2d_ = torch.mean(reconstruction.reshape(B*D,C,H,W),dim=1,keepdim=True)
                    gap = 8
                    for b in range(0,B*D,gap):
                        input_2d = input_2d_[b:b+1,:,:,:]
                        reconstruction_2d = reconstruction_2d_[b:b+1,:,:,:] #[:,:,s,:,:]

                        try:
                            loss_vae_2d, log_vae_2d = Discriminator(input_2d,
                                                                reconstruction_2d,
                                                                posteriors = DiagonalGaussianDistribution(x2_mu, x2_std), 
                                                                optimizer_idx=0, 
                                                                global_step = step,
                                                                last_layer=get_last_layer(),
                                                                )
                            loss_disc, log_disc = Discriminator(input_2d.detach(),
                                                                reconstruction_2d.detach(),
                                                                posteriors = DiagonalGaussianDistribution(x2_mu, x2_std),
                                                                optimizer_idx=1,
                                                                global_step = step,
                                                                last_layer=get_last_layer(),)
                        
                            vae_losses_2d.append(loss_vae_2d)
                            disc_losses.append(loss_disc)

                        except RuntimeError as e:
                            print(f"Error in batch {i}, sub-batch {b}: {e}")
                            continue



            train_vae_loss = torch.mean(torch.stack(vae_losses_2d)) #len = 16 = 2*(32/4)
            train_vae_loss += vae_loss_3d 
            train_disc_loss = torch.mean(torch.stack(disc_losses))

            train_vae_loss_sum += train_vae_loss
            train_disc_loss_sum += train_disc_loss

            train_vae_loss.backward()
            optimizer_vae.step()
            del vae_losses_2d, disc_losses


            train_disc_loss.backward()
            optimizer_disc.step() 
            del input_2d_, reconstruction_2d_, loss_vae_2d, loss_disc, input_2d, reconstruction_2d


            step += 1

            writer.add_scalar('Loss/train_vae_sum', train_vae_loss.item(), step)

            writer.add_scalar('Loss/train_3d_vae', vae_loss_3d.item(), step)
            writer.add_scalar('Loss/train_3d_reconstruction_data_loss', x1_rec_loss.item(), step)
            writer.add_scalar('Loss/train_3d_kl_latent_loss', kl_loss.item(), step)
            writer.add_scalar('Loss/train_3d_latent_loss', lat_rec_loss.item(), step)

            writer.add_scalar('Loss/train_2d_reconstruction_data_loss', log_vae_2d['train/rec_loss'].item(), step)
            writer.add_scalar('Loss/train_nll_loss', log_vae_2d['train/nll_loss'].item(), step)
            writer.add_scalar('Loss/train_g_loss', log_vae_2d['train/g_loss'].item(), step)

            writer.add_scalar('Loss/train_disc_sum', train_disc_loss.item(), step)
            writer.add_scalar('Loss/train_real', log_disc['train/logits_real'].item(), step)
            writer.add_scalar('Loss/train_fake', log_disc['train/logits_fake'].item(), step)

        if step % log_interval == 0: #因为数据太少 所以这个每个只在每个epoch开始的时候保存一次
            # print('Train Vae Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, i * len(batch['sag_dirs']['images']), len(train_loader.dataset),
            #         100. * i / len(train_loader) * 2,
            #         train_vae_loss_sum.item() / len(batch['sag_dirs']['images'])))
            # print('Train Disc Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, i * len(batch['sag_dirs']['images']), len(train_loader.dataset),
            #         100. * i / len(train_loader),
            #         train_disc_loss_sum.item() / len(batch['sag_dirs']['images'])))
            
            save_path = os.path.join(results_dir,'train',str(epoch)+'_epoch',str(i)+'_step')
            os.makedirs(save_path,exist_ok=True)


            #暂时先只可视化sag to cor 单向的结果 但是训练的是双向的
            sag_img, cor_img,sag_ori_img,cor_ori_img = sagtocor_dict['x1_input'],sagtocor_dict['x2_input'],sagtocor_dict['x1_input_ori'],sagtocor_dict['x2_input_ori']
            sag_reconstruction,cor_reconstruction = sagtocor_dict['x1_reconstruction'],sagtocor_dict['x2_reconstruction']
            save_vis(sag_reconstruction , path = os.path.join(save_path,'sag_reconstruction_sitk'+'.nii.gz'))
            save_vis(sag_img,  path = os.path.join(save_path,'sag_input_sitk'+'.nii.gz'))
            save_vis(cor_reconstruction , path = os.path.join(save_path,'cor_reconstruction_sitk'+'.nii.gz'))
            save_vis(cor_img,  path = os.path.join(save_path,'cor_input_sitk'+'.nii.gz'))
            save_vis(sag_ori_img,  path = os.path.join(save_path,'sag_input_ori'+'.nii.gz'))
            save_vis(cor_ori_img,  path = os.path.join(save_path,'cor_input_ori'+'.nii.gz'))   
            del sag_img, sag_reconstruction, cor_img, cor_reconstruction, sag_ori_img, cor_ori_img


    print('====> Epoch: {} Average vae loss: {:.4f}'.format(epoch, train_vae_loss_sum / len(train_loader.dataset)))
    print('====> Epoch: {} Average disc loss: {:.4f}'.format(epoch, train_disc_loss_sum / len(train_loader.dataset)))
    
    
    return model, Discriminator, optimizer_vae, optimizer_disc, step



def test(model, epoch, step, test_loader, Discriminator, writer,
         batch_size = 1,
         rank = '0',
         results_dir = None,
         ddp = False
         ):
    
    rank = torch.device(f'cuda:{rank}')
    n_test = 20


    def get_last_layer():
        if ddp:
            return model.module.net.decoder.out.weight
        else:
            return model.net.decoder.out.weight
        


    model.eval()
    Discriminator.eval()

    test_vae_loss_sum = 0
    test_disc_loss_sum = 0         

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing", unit="batch")): #tqdm(train_loader, desc="Training", unit="batch")
            if i >= n_test:
                break

            sagtocor_dict,cortosag_dict = model(batch,rank)


            test_vae_loss = 0
            test_disc_loss = 0     
            vae_loss_3d = 0
            vae_losses_2d = []
            disc_losses = []
            for dic in (sagtocor_dict,cortosag_dict):#,
                #x1
                x1_rec_loss = 0
                x1_kl_loss = 0              
                x1_reconstruction = dic['x1_reconstruction']
                x1_input = dic['x1_input']
                x1_mu = dic['x1_mu']
                x1_std = dic['x1_std']
                x1_rec_loss += reconstruction_loss(x1_reconstruction,x1_input)
                # x1_kl_loss += KLLoss(x1_mu, x1_std) * 0.5     #这个是不是也可以设置一下开始步长，最初让先稳定一下encode
                x1_kl_loss = 0.

                #x2
                x2_rec_loss = 0
                x2_kl_loss = 0  
                x2_reconstruction = dic['x2_reconstruction']
                x2_input = dic['x2_input']
                x2_mu = dic['mu1tomu2']
                x2_std = dic['std1tostd2']
                x2_rec_loss += reconstruction_loss(x2_reconstruction,x2_input)
                x2_kl_loss += KLLoss(x2_mu, x2_std)

                #lat_recloss
                lat_rec_loss = dic['lat_loss_x1'] #dic['lat_loss_1'] + dic['lat_loss_2']

                rec_loss = x1_rec_loss  + lat_rec_loss  #+ x2_rec_loss
                kl_loss = x1_kl_loss + x2_kl_loss
                vae_loss_3d += rec_loss * args.reconstruction_data_loss_weight + kl_loss * args.kl_latent_loss_weight

            
                B,C,D,H,W = x1_input.shape
                for input, reconstruction in (x1_input, x1_reconstruction),(x2_input, x2_reconstruction):  #两个重构回来的都计算gan_loss
                    input_2d_ = input.reshape(B*D,C,H,W)
                    reconstruction_2d_ = reconstruction.reshape(B*D,C,H,W)
                    gap = 4
                    for b in range(0,B*D,gap):
                        input_2d = input_2d_[b:b+1,:,:,:]
                        reconstruction_2d = reconstruction_2d_[b:b+1,:,:,:] #[:,:,s,:,:]

                        try:
                            loss_vae_2d, log_vae_2d = Discriminator(input_2d,
                                                                reconstruction_2d,
                                                                posteriors = DiagonalGaussianDistribution(x2_mu, x2_std), 
                                                                optimizer_idx=0, 
                                                                global_step = step,
                                                                last_layer=get_last_layer(),
                                                                split = 'test'
                                                                )
                            loss_disc, log_disc = Discriminator(input_2d.detach(),
                                                                reconstruction_2d.detach(),
                                                                posteriors = DiagonalGaussianDistribution(x2_mu, x2_std),
                                                                optimizer_idx=1,
                                                                global_step = step,
                                                                last_layer=get_last_layer(),
                                                                split = 'test')
                    
                            vae_losses_2d.append(loss_vae_2d)
                            disc_losses.append(loss_disc)

                            torch.cuda.empty_cache()

                        except RuntimeError as e:
                            print(f"Error in batch {i}, sub-batch {b}: {e}")
                            continue

            test_vae_loss = torch.mean(torch.stack(vae_losses_2d))
            test_vae_loss += vae_loss_3d
            test_disc_loss = torch.mean(torch.stack(disc_losses))

            test_vae_loss_sum += test_vae_loss
            test_disc_loss_sum += test_disc_loss
 


            writer.add_scalar('Loss/test_vae_sum', test_vae_loss.item(), step)

            writer.add_scalar('Loss/test_3d_vae', vae_loss_3d.item(), step)
            writer.add_scalar('Loss/test_3d_reconstruction_data_loss', rec_loss.item(), step)
            writer.add_scalar('Loss/test_3d_kl_latent_loss', kl_loss.item(), step)

            writer.add_scalar('Loss/test_2d_reconstruction_data_loss', log_vae_2d['test/rec_loss'].item(), step)
            writer.add_scalar('Loss/test_nll_loss', log_vae_2d['test/nll_loss'].item(), step)
            writer.add_scalar('Loss/test_g_loss', log_vae_2d['test/g_loss'].item(), step)

            writer.add_scalar('Loss/test_disc_sum', test_disc_loss.item(), step)
            writer.add_scalar('Loss/test_real', log_disc['test/logits_real'].item(), step)
            writer.add_scalar('Loss/test_fake', log_disc['test/logits_fake'].item(), step)

            
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(batch['sag_dirs']['images']), len(test_loader.dataset),
                    100. * i / len(test_loader),
                    test_vae_loss.item() / len(batch['sag_dirs']['images'])))
            
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(batch['sag_dirs']['images']), len(test_loader.dataset),
                    100. * i / len(test_loader),
                    test_disc_loss.item() / len(batch['sag_dirs']['images'])))
            
            save_path = os.path.join(results_dir,'test','epoch_' + str(epoch),'img_'+str(i))
            os.makedirs(save_path,exist_ok=True)

            # if epoch % (args.val_epoch_interval*2) == 0:
            #     #暂时先只可视化sag to cor 单向的结果 但是训练的是双向的
            #     sag_img, cor_img,sag_ori_img,cor_ori_img = sagtocor_dict['x1_input'],sagtocor_dict['x2_input'],sagtocor_dict['x1_input_ori'],sagtocor_dict['x2_input_ori']
            #     sag_reconstruction,cor_reconstruction = sagtocor_dict['x1_reconstruction'],sagtocor_dict['x2_reconstruction']
            #     save_vis(sag_reconstruction , path = os.path.join(save_path,'sag_reconstruction_sitk'+'.nii.gz'))
            #     save_vis(sag_img,  path = os.path.join(save_path,'sag_input_sitk'+'.nii.gz'))
            #     save_vis(cor_reconstruction , path = os.path.join(save_path,'cor_reconstruction_sitk'+'.nii.gz'))
            #     save_vis(cor_img,  path = os.path.join(save_path,'cor_input_sitk'+'.nii.gz'))
            #     save_vis(sag_ori_img,  path = os.path.join(save_path,'sag_input_ori'+'.nii.gz'))
            #     save_vis(cor_ori_img,  path = os.path.join(save_path,'cor_input_ori'+'.nii.gz'))            
            #     del sag_img, sag_reconstruction, cor_img, cor_reconstruction 

    writer.flush()
    print('====> Test vae set loss: {:.4f}'.format(test_vae_loss_sum / n_test))
    print('====> Test disc set loss: {:.4f}'.format(test_disc_loss_sum / n_test))
    return model, step


def make_train_loader(train_data_dir,device):
    train_data = CustomTrain(train_data_dir,device)
    return train_data

def make_test_loader(test_data_dir,device):
    test_data = CustomTrain(test_data_dir,device)
    return test_data



def main(rank, world_size, args):

    torch.manual_seed(args.seed)

    model = DualViewSegNet(args.n_channels, gf_dim=2, lat_ch1=4, lat_ch2 = 4, ld=False, encoder_pre=True, decoder_pre=True)
    Discriminator = LPIPSWithDiscriminator(disc_start = 8001,
                                            kl_weight = 1.0e-06,
                                            disc_in_channels=3,
                                            disc_weight = 0.5,
                                            )#perceptual_weight=1.0

    if args.distributed:
        Discriminator = Discriminator.to(rank)
        Discriminator = nn.parallel.DistributedDataParallel(Discriminator, device_ids=[rank], find_unused_parameters=True)
        model = model.to(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = model.to(rank)
        Discriminator = Discriminator.to(rank)

    optimizer_vae = optim.Adam(list(model.parameters()), lr=args.learning_rate)
    optimizer_disc = optim.Adam(list(Discriminator.parameters()), lr=args.learning_rate)
    print('Cuda is {}available'.format('' if torch.cuda.is_available() else 'not '))
    print(model)
    print(Discriminator)


    if args.resume_path is None:
        timestamp = time.time()
        EXPERIMENT = '3d_vae_sag&cor_pdfs' + datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S')
        if args.distributed:
            EXPERIMENT = EXPERIMENT+'_ddp'
        run_name = 'vol_256_lr_{}' \
                '_kl_{}_' \
                '_bsize_{}' \
                ''.format(
            args.learning_rate,
            args.kl_latent_loss_weight,
            args.batch_size)

        load_from_ckpt_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'checkpoints')
        os.makedirs(load_from_ckpt_dir, exist_ok=True)

        print('testing complete')
        print("=> no checkpoint found")
        start_epoch = 1
        step = 0
    else:
        resume_path = args.resume_path #os.path.join(load_from_ckpt_dir, sorted(os.listdir(load_from_ckpt_dir))[-1])
        if not os.path.exists(resume_path):
            print("=> no checkpoint found")
            start_epoch = 1
            step = 0
        else:
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device(f'cuda:{rank}'))
            start_epoch = checkpoint['epoch']
            try:
                model.load_state_dict(remove_module_prefix(checkpoint['model']))
                Discriminator.load_state_dict(remove_module_prefix(checkpoint['Discriminator']))
            except:
                model.load_state_dict(checkpoint['model'])
                Discriminator.load_state_dict(checkpoint['Discriminator'])
            optimizer_vae.load_state_dict(checkpoint['optimizer_vae'])
            optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
            step = checkpoint['step']
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
            
        EXPERIMENT = resume_path.split('/')[5]
        run_name = resume_path.split('/')[7]

    ckpt_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    tb_dir = os.path.join('experiments', EXPERIMENT, 'gen', 'tensorboard', run_name)
    if not os.path.isdir(tb_dir):
        os.makedirs(tb_dir, exist_ok=True)

    frame_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'frames')
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)

    results_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    if args.pretrain_path is not None:
        pre_model = torch.load(args.pretrain_path, map_location=torch.device(f'cuda:{rank}'))
        try:
            model.net.encoder.load_state_dict(remove_module_prefix(pre_model['model'],strict=False))
            model.net.decoder.load_state_dict(remove_module_prefix(pre_model['model'],strict=False))
            # model = DualViewSegNet(args.n_channels, gf_dim=2, lat_ch1 = 4,lat_ch2 = 4, ld=False, encoder_pre=True, decoder_pre=True)
            model = model.to(rank)
        except:
            model.net.encoder.load_state_dict(pre_model['model'],strict=False) 
            model.net.decoder.load_state_dict(pre_model['model'],strict=False)
            # model = DualViewSegNet(args.n_channels, gf_dim=2, lat_ch1 = 4,lat_ch2 = 4, ld=False, encoder_pre=True, decoder_pre=True)        
            model = model.to(rank)

    writer = tensorboard.SummaryWriter(log_dir=tb_dir)
    
    if args.phase == 'train':
        train_dataset = make_train_loader(args.train_data_dir, device = rank)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                    num_workers=args.num_workers,
                                                    # shuffle=True,
                                                    drop_last=True,
                                                    # multiprocessing_context='spawn',
                                                    )
        
        for epoch in range(start_epoch, args.epochs + 1):
            model, Discriminator, optimizer_vae, optimizer_disc, step = train(model, optimizer_vae,optimizer_disc, epoch, step, train_loader,
                                                                Discriminator,
                                                                writer,
                                                                log_interval=args.log_interval,#args.log_interval
                                                                rank = rank,
                                                                results_dir = results_dir,
                                                                ddp = True if args.distributed else False)
            
            torch.cuda.empty_cache()  # Clear GPU memory
            gc.collect()  # Garbage collect to clear unused CPU memory

            if epoch % args.save_epoch_interval == 0:
                save_path = os.path.join(ckpt_dir, 'model_{0:08d}.pth.tar'.format(epoch))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer_vae': optimizer_vae.state_dict(),
                    'optimizer_disc': optimizer_disc.state_dict(),
                    'Discriminator': Discriminator.state_dict(),
                    'epoch': epoch,
                    'step': step
                },
                    save_path)
                print('Saved model')

                if epoch!=0 and epoch % args.val_epoch_interval == 0:
                    args.phase = 'test'        
                    test_dataset = make_test_loader(args.test_data_dir, device = rank)
                    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                                                num_workers=args.num_workers,
                                                                shuffle=False,
                                                                drop_last=True,
                                                                )
                    _, step = test(model, epoch, step, test_loader, Discriminator, writer,
                                                    batch_size=1,
                                                    rank = rank,
                                                    results_dir = results_dir,
                                                    ddp = True if args.distributed else False
                                                    )
                    args.phase = 'train'

        writer.close()    

    else:  

        test_dataset = make_test_loader(args.test_data_dir, device = rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                                    num_workers=args.num_workers,
                                                    shuffle=False,
                                                    drop_last=True,
                                                    )   
        model, step = test(model, 1, step, test_loader, Discriminator, writer,
                            batch_size=1,
                            rank = rank,
                            results_dir = results_dir,
                            ddp = True if args.distributed else False
                            )
        



if __name__ == "__main__":
    # mp.set_start_method('spawn')

    args = parser.parse_args()
    args.train_data_dir = '/mnt/users/read_side_and_weighting_data/registrated_image/cor_pdfs-sag_pdfs-norm-crop/train/SAG'
    args.test_data_dir = '/mnt/users/read_side_and_weighting_data/registrated_image/cor_pdfs-sag_pdfs-norm-crop/val/SAG'
    args.num_workers = 3
    args.gpu_ids = [0,1,2,3,4,5]
    args.distributed = False
    args.resume_path = None
    # args.resume_path = '/mnt/users/3d_resiger_vae2/experiments/3d_vae_sag&cor_pdfs2024-12-15-16-23-16/gen/vol_256_lr_0.0001_kl_1e-05__bsize_1/checkpoints/model_00000500.pth.tar'
    args.phase = 'train'
    args.pretrain_path = None
    args.pretrain_path = '/mnt/users/3d_resiger_vae/experiments/3d_vae_sag_pd2024-11-29-16-00-33/gen/vol_256_lr_0.0001_kl_1e-05__bsize_1/checkpoints/model_00000060.pth.tar'

    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if args.distributed:
        ngpus_per_node = len(args.gpu_ids) # or torch.cuda.device_count()
        args.world_size = ngpus_per_node
        mp.spawn(init_process, args=(args.world_size, main, (args,)), nprocs=args.world_size, join=True)
    else:
        args.world_size = 1 
        main(0, 1, args)

    # main(args)