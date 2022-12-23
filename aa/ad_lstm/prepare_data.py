#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Thur 11 16:16:00 2021
From Mingjie Chen
From Rossana Milner
From Asif Jalal
check project:blstmatt
@author: Jose Antonio Lopez @ The University of Sheffield

"""

import sys, os
import json
import pickle
import argparse
from data import INAWinData
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
sys.path.append('/aa/ad_lstm/tools')
from welford import Welford
import numpy as np
import load_txt
import save_txt

####select if cuda is available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}
device = torch.device("cuda:"+str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
     print(f"Torch version cuda: {torch.version.cuda}")
     print(torch.cuda.get_device_name(0))
     print('Memory Usage:')
     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


parser = argparse.ArgumentParser('LSTM+ATTN Training.')
parser.add_argument('--use_cuda', default=False,action='store_true')
parser.add_argument('--feat',type=str,default='mel80')
parser.add_argument('--pre_data_h5',type=str,default='h5/INA_window.h5')
parser.add_argument('--mvn_pa',type=str,default='mvn/wav_mel_train', help='Location for the statistics for mvn.')
parser.add_argument('--win_mel_json',type=str,default='json/win_mel.json.x')
parser.add_argument('--win_label_json',type=str,default='json/win_label.json.x')
parser.add_argument('--ph_mode',type=str,default='ax')
parser.add_argument('--ref_file',type=str,default='ref/INA.v1.cut1-6.phone.$ref_mode.ref')
parser.add_argument('--ref_mode',type=str,default='a1')
parser.add_argument('--wind',type=str,default='0.5')
parser.add_argument('--str',type=str,default='0.1')
parser.add_argument('--ctf',type=int,default=0)
parser.add_argument('--subset',type=str,default='train')

					   
					   

args = parser.parse_args()

args.mvn_pa = args.mvn_pa +f".w_{args.wind}.str_{args.str}"
if args.ctf > 0:
    args.mvn_pa = args.mvn_pa +f".ctf_{args.ctf}"
    args.win_mel_json = args.win_mel_json +f".ctf_{args.ctf}"
    args.win_label_json = args.win_label_json +f".ctf_{args.ctf}"
    

print(args)

### ----------------------------------------- seed
seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
        

        

if __name__ == '__main__':

    if args.subset == 'train':
        print("Working on train set")
        dset = INAWinData(args.win_mel_json+'.train', args.win_label_json+'.train' ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = args.ph_mode, subset = 'train', ref_mode = args.ref_mode )
                                                                                                                            
    else:
        print("Working on test set")
        dset = INAWinData(args.win_mel_json+'.test', args.win_label_json+'.test' ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = args.ph_mode, subset = 'test', ref_mode = args.ref_mode )
    
    print(f"nclass: {dset.nclass}")
    print("test run")
    
    mel, y = dset[1]
    
    print(f"mel shape: {mel.shape}")
    print(mel)
    
    print(f"y shape: {y.shape}")
    print(y)
    
    _dataloader = DataLoader(dset, batch_size= 2 )
    
    print("On dataloader")
    for step, (feat, target) in enumerate(_dataloader):
        print(f"feat shape: {feat.shape}")
        print(f"target shape: {target.shape}")
        
        if step > 1:
            break

    
