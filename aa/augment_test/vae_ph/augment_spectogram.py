#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri February 12 17:13:00 2021
From https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb

@author: Jose Antonio Lopez @ Infirmary Rd

Use the following python

ENVIRONMENT=
source /python/anaconda3/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:
"""

from collections import namedtuple
import random
import torch
import torchaudio

from nb_SparseImageWarp import sparse_image_warp

##own imports
import scipy.io.wavfile as wav
from mini_methods import *
import argparse
import sys
import soundpy as sp

#import libraries for graphics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

####select if cuda is available
#use_cuda = torch.cuda.is_available()
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

device = torch.device("cuda:"+str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")


sample  = ''

parser = argparse.ArgumentParser()

parser.add_argument('--sample_rate',default=16000)
parser.add_argument('--n_mfcc',default=13)
parser.add_argument('--n_fft',default=int(0.025*16000))
parser.add_argument('--hop_length',default=int(0.010*16000))
parser.add_argument('--n_mels',default=80)
parser.add_argument('--win_length',default=int(0.025*16000))
parser.add_argument('--add_noise',default=False,action='store_true')
parser.add_argument('--SNR',type=int,default=10)
#for the melspec
parser.add_argument('--frame_length_ms', help='frame length in ms',default=int(25))
parser.add_argument('--frame_step_ms', help='frame shift in ms',default=int(10))
parser.add_argument('--preemph', help='pre-emphasis factor',default=float(0.97))
parser.add_argument('--low_freq_hz', help='low frequency in Hz',default=int(0))
parser.add_argument('--high_freq_hz', help='High frequency in Hz',default=int(8000))
parser.add_argument('--num_filters', help='number of mel-filters',default=int(80))
parser.add_argument('--num_ceps', help='number of cepstral coefficients (excluding C0)',default=int(12))
parser.add_argument('--cep_lifter', help='Cepstral liftering order',default=int(22))

args = parser.parse_args()
args.num_filters = int(args.num_filters)
print(args)

def tensor_to_img(spectrogram, name):
    plt.figure(figsize=(10,10)) # arbitrary, looks good on my screen.
    plt.imshow(spectrogram[0], origin="lower")
    savename = f"/share/mini1/res/t/asr/call/childread-nl/its/aa/augment_test/augment_demo/tensor2img_{name}.png"
    plt.savefig(savename, dpi = 300)
    print(f"spectoshape: {spectrogram.shape}")
    
def triple_plot(spectrogram, spectoname, title):
#this to plot 3 different tests for data augmentation
#spectogram is a list of 2d tensors
#spectoname is a list corresponding to the spectogram list


    fig = plt.figure(figsize=(25*len(spectoname),10))
    st = fig.suptitle(title, fontsize="x-large")
    
    figplot = int(f"{len(spectoname)}11")
    
    ax = fig.add_subplot(figplot)
    ax.imshow(spectrogram[0], origin="lower")
    ax.set_title(spectoname[0])
    
    if len(spectoname) > 1 :
        ax = fig.add_subplot(figplot+1)
        ax.imshow(spectrogram[1], origin="lower")
        ax.set_title(spectoname[1])

    if len(spectoname) > 2:
        ax = fig.add_subplot(figplot+2)
        ax.imshow(spectrogram[2], origin="lower")
        ax.set_title(spectoname[2])

    if len(spectoname) > 3:
        ax = fig.add_subplot(figplot+3)
        ax.imshow(spectrogram[3], origin="lower")
        ax.set_title(spectoname[3])
        
    if len(spectoname) > 4:
        ax = fig.add_subplot(figplot+4)
        ax.imshow(spectrogram[4], origin="lower")
        ax.set_title(spectoname[4])
    

    
    fig.subplots_adjust(hspace = 0.5)
    savename = f"/augment_test/augment_demo/{title}.png"
    
    plt.savefig(savename, dpi = 300)
    print(f"triple plot saved as: {savename}")  
    
    
def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device
    
    y = num_rows//2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len
    
    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device), 
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)
    
    
def freq_mask(spec, F=20, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    
    for i in range(0, num_masks):        
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f):
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f) 
        
        if replace_with_zero: 
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()
    
    return cloned
    
    
def time_mask(spec, T=20, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t):
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        
        if replace_with_zero:
            cloned[0][:,t_zero:mask_end] = 0
        else:
            cloned[0][:,t_zero:mask_end] = cloned.mean()
            
    return cloned

AudioData = namedtuple('AudioData', ['sig', 'sr'])

# get original mel
mel,  meldB, freq_bins, siglen = get_melspectrogram(sample, args)

plot_mel(mel, 'mel', '', freq_bins, siglen, args)
plot_mel(meldB, 'meldB', '', freq_bins, siglen, args)

#torchaudio.transforms.Spectrogram()
#sys.exit()



meldB = FloatTensor(meldB)
meldB = meldB.unsqueeze(0)
meldB = meldB.permute(0,2,1)
#2nd dimension is the frequency bins
#3rd dimension is the timeframes


spectrogram = [ meldB[0], time_warp(meldB, W=3)[0], time_warp(meldB, W=5)[0], time_warp(meldB, W=10)[0]]
spectoname=['original', 'w=3', 'w=5', 'w=10']
triple_plot(spectrogram, spectoname, 'timewarp_full')

spectrogram = [ meldB[0,:,:150], time_warp(meldB[:,:,:150], W=3)[0], time_warp(meldB[:,:,:150], W=5)[0], time_warp(meldB[:,:,:150], W=10)[0]]
spectoname=['0:150', 'w=3', 'w=5', 'w=10']
triple_plot(spectrogram, spectoname, 'timewarp_0-150')

spectrogram = [ meldB[0,:,150:], time_warp(meldB[:,:,150:], W=3)[0], time_warp(meldB[:,:,150:], W=5)[0], time_warp(meldB[:,:,150:], W=10)[0]]
spectoname=['150:', 'w=3', 'w=5', 'w=10']
triple_plot(spectrogram, spectoname, 'timewarp_150-272')



spectrogram = [ meldB[0],freq_mask(meldB, num_masks=2)[0], freq_mask(meldB, num_masks=2, replace_with_zero=True)[0] ]
spectoname=['original', 'nm=2', 'nm=2/repzero']
triple_plot(spectrogram, spectoname, 'freqmask_full')

spectrogram = [ meldB[0,:,:150],freq_mask(meldB[:,:,:150], num_masks=2)[0], freq_mask(meldB[:,:,:150], num_masks=2, replace_with_zero=True)[0] ]
spectoname=['original', 'nm=2', 'nm=2/repzero']
triple_plot(spectrogram, spectoname, 'freqmask_0-150')

spectrogram = [ meldB[0,:,150:],freq_mask(meldB[:,:,150:], num_masks=2)[0], freq_mask(meldB[:,:,150:], num_masks=2, replace_with_zero=True)[0] ]
spectoname=['original', 'nm=2', 'nm=2/repzero']
triple_plot(spectrogram, spectoname, 'freqmask_150-272')



spectrogram = [ meldB[0],time_mask(meldB, num_masks=2)[0], time_mask(meldB, num_masks=2, replace_with_zero=True)[0] ]
spectoname=['original', 'nm=2', 'nm=2/repzero']
triple_plot(spectrogram, spectoname, 'timemask_full')

spectrogram = [ meldB[0,:,:150],time_mask(meldB[:,:,:150], num_masks=2)[0], time_mask(meldB[:,:,:150], num_masks=2, replace_with_zero=True)[0] ]
spectoname=['original', 'nm=2', 'nm=2/repzero']
triple_plot(spectrogram, spectoname, 'timemask_0-150')

spectrogram = [ meldB[0,:,150:],time_mask(meldB[:,:,150:], num_masks=2)[0], time_mask(meldB[:,:,150:], num_masks=2, replace_with_zero=True)[0] ]
spectoname=['original', 'nm=2', 'nm=2/repzero']
triple_plot(spectrogram, spectoname, 'timemask_150-272')


##################################################################################
#torchaudio transform
##################################################################################


freqmaskSeq = torch.nn.Sequential( torchaudio.transforms.FrequencyMasking(freq_mask_param = 20) )

#lol = freqmaskSeq(torch.clone(meldB))
#tensor_to_img(meldB, 'basemel')
#tensor_to_img(lol, 'freqmasklol')

spectrogram = [ meldB[0],freqmaskSeq(torch.clone(meldB))[0], freqmaskSeq(torch.clone(meldB))[0], freqmaskSeq(freqmaskSeq(torch.clone(meldB)))[0]]
spectoname=['original', 'freq_mask_param = 20', 'freq_mask_param = 20', 'freq_mask_param = 20 x 2']
triple_plot(spectrogram, spectoname, 'torch_freqmask')


TimemaskSeq = torch.nn.Sequential( torchaudio.transforms.TimeMasking(time_mask_param = 10) )

#lol = freqmaskSeq(torch.clone(meldB))
#tensor_to_img(meldB, 'basemel')
#tensor_to_img(lol, 'freqmasklol')

spectrogram = [ meldB[0],TimemaskSeq(torch.clone(meldB))[0], TimemaskSeq(torch.clone(meldB))[0], TimemaskSeq(TimemaskSeq(torch.clone(meldB)))[0]]
spectoname=['original', 'time_mask_param = 10', 'time_mask_param = 10', 'time_mask_param = 10 x 2']
triple_plot(spectrogram, spectoname, 'torch_timemask')

spectrogram = [ meldB[0] ]
spectoname=['original']
tensor_to_img(meldB, 'orig')

_,  meldB, _, _ = get_melspectrogram("noise.wav", args)
spectrogram.append(meldB)
spectoname.append('noisy')

meldB = FloatTensor(meldB)
meldB = meldB.unsqueeze(0)
meldB = meldB.permute(0,2,1)
tensor_to_img(meldB, 'soundpy_noise')

_,  meldB, _, _ = get_melspectrogram("pitchd.wav", args)
spectrogram.append(meldB)
spectoname.append('pitchd')

meldB = FloatTensor(meldB)
meldB = meldB.unsqueeze(0)
meldB = meldB.permute(0,2,1)
tensor_to_img(meldB, 'soundpy_pitchd')

_,  meldB, _, _ = get_melspectrogram("pitchi.wav", args)
spectrogram.append(meldB)
spectoname.append('pitchi')

meldB = FloatTensor(meldB)
meldB = meldB.unsqueeze(0)
meldB = meldB.permute(0,2,1)
tensor_to_img(meldB, 'soundpy_pitchi')

_,  meldB, _, _ = get_melspectrogram("vtlp.wav", args)
spectrogram.append(meldB)
spectoname.append('vtlp')

meldB = FloatTensor(meldB)
meldB = meldB.unsqueeze(0)
meldB = meldB.permute(0,2,1)
tensor_to_img(meldB, 'soundpy_vtlp')



print("len spectrogram")
print(len(spectrogram))
print("len spectoname")
print(len(spectoname))

triple_plot(spectrogram, spectoname, 'soundpy')




