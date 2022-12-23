#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri December 04 15:46:00 2020

@author: Jose Antonio Lopez @ The University of Sheffield

This script generates VAE embeddings for the acoustic segments.

"""

import h5py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from data import INADataset
from models.vae_model import Model,SingleModel
from models.ffwd_model import *
import pandas as pd
import json

import argparse
import os,time, sys
sys.path.append('/aa/ad/tools')
import load_txt
import save_txt

from da_env import VAE_IN_DIM, VAE_HID, VAE_SEG_LEN, VAE_EPOCH, VAE_MODEL
from da_env import TRAIN_LDAPOST, TEST_LDAPOST, LDA_TOPICS, LDA_COMP, LDA_EPOCHS
from da_env import REF_FILE

use_cuda = torch.cuda.is_available()


parser = argparse.ArgumentParser("Assessor's Decision")
parser.add_argument('--use_cuda', default=False,action='store_true')
parser.add_argument('--tr_ph_json',type=str,default='json/sample_mfcc.json.train')
parser.add_argument('--te_ph_json',type=str,default='json/sample_mfcc.json.test')
parser.add_argument('--tr_ph_len_json',type=str,default='json/ph_length_plptrain.json')
parser.add_argument('--te_ph_len_json',type=str,default='json/ph_length_plptrain.json')
parser.add_argument('--pre_data_h5',type=str,default='h5/timit_mfcc39.h5')
parser.add_argument('--mvn_pa',type=str,default='mvn/wav_plp39_train', help='Location for the statistics for mvn.')
parser.add_argument('--log_dir',type=str,default='runs/vae/')
parser.add_argument('--save_model',default=False,action='store_true')
parser.add_argument('--model_dir',type=str,default='./ckpt/vae/')
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--epoch',type=int,default=100)
parser.add_argument('--stats_dict',type=str,default='./json/train_stats/')
parser.add_argument('--model_name',type=str,default='VAE')
parser.add_argument('--model',type=str,default='FFWD')
parser.add_argument('--ref_mode',type=str,default='a1')
parser.add_argument('--output',type=str,default='./output/model1.lr')
parser.add_argument('--gop', default=False,action='store_true')
parser.add_argument('--vae', default=False,action='store_true')
parser.add_argument('--lda', default=False,action='store_true')
#######################################################


args = parser.parse_args()
REF_FILE = REF_FILE.replace('REF_MODE', args.ref_mode)

DETAIL_NAME=''
#adjust the name given the inputs used
if args.gop:
    DETAIL_NAME = DETAIL_NAME + f"_gop"
if args.vae:
    DETAIL_NAME = DETAIL_NAME + f"_VAEh{VAE_HID}_VAEep{VAE_EPOCH}"
if args.lda: 
    DETAIL_NAME = DETAIL_NAME + f"_LDAt{LDA_TOPICS}_LDAd{LDA_COMP}_LDAep{LDA_EPOCHS}"
    
args.stats_dict = args.stats_dict + DETAIL_NAME
args.model_name = args.model_name + DETAIL_NAME
args.log_dir = args.log_dir + DETAIL_NAME
args.output = args.output + DETAIL_NAME + f"_ep{args.epoch}"
args.tr_enc_json = args.output[:args.output.rfind('/')+1]  + "ph_tr"
args.te_enc_json = args.output[:args.output.rfind('/')+1]  + "ph_te"

print(args)
    
    
def get_enc(dataset,model, output, f_h5, enc_json):
#generates the label outputs for each subset

    dataloader = DataLoader(dataset, batch_size=512)
    
    segment_list = []
    output_list = []

    for step, batch in enumerate(dataloader):
    
        X = batch[0]
        X_hat = model.encode(X)
        X_hat = list( np.round( X_hat.detach().cpu().numpy(), 8) )
        output_list += X_hat
        
    #here all the embeddings should have been obtained
    for i,j in enumerate(dataset.samples):
        h5_path = j[0]+'/'+ '.'.join( map(str, j[1:])  )
        f_h5.create_dataset(f"{h5_path}",data=np.array(output_list[i]),dtype=np.float32)
        
        segment_list.append(h5_path)
        
    if os.path.exists(enc_json):
        os.remove(enc_json)
        
    with open(enc_json,'w') as f:
        for item in segment_list:
            f.write("%s\n" % item)

    print(f"data instances: {len(segment_list)}",flush=True)
    print(f"saved as: {enc_json}")
       
    return f_h5

if __name__ == '__main__':

    in_dim = 1 #the phone tag
    if args.gop:
        in_dim += 1
    if args.vae:
        in_dim += int(VAE_HID)*int(VAE_SEG_LEN)
    if args.lda:
        in_dim += int(LDA_TOPICS)
        

	# create h5 file
    if os.path.exists(args.output):
        os.remove(args.output)
    f_h5 = h5py.File(args.output+'.h5','w')


    #build the feedforward model
    model = eval(args.model)(in_dim=in_dim, num_cls=1)
    load_model_dir = os.path.join(args.model_dir, args.model_name +'_'+str(args.epoch)+'.pkl')
    print("This is load_model_dir")
    print(load_model_dir)
    if not os.path.exists(load_model_dir):
        raise Exception
    model_f = open(load_model_dir,'rb')
    state_dict = torch.load(model_f)
    model.load_state_dict(state_dict['model'])
    model.eval()
    if use_cuda:
        model.cuda()

    trainset = INADataset(samp_json_pa = args.tr_ph_json, h5_pa = args.pre_data_h5, ph_len = args.tr_ph_len_json, mvn_pa = args.mvn_pa, vae_name = VAE_MODEL, vae_in_dim = VAE_IN_DIM, vae_hid = VAE_HID, seg_len = VAE_SEG_LEN, lda_post = TRAIN_LDAPOST, lda_t = LDA_TOPICS, lda_d = LDA_COMP, lda_e = LDA_EPOCHS, use_cuda = args.use_cuda, ref_file = REF_FILE, ref_mode = args.ref_mode, gop = args.gop, vae = args.vae, lda = args.lda)
    testset = INADataset(samp_json_pa = args.te_ph_json, h5_pa = args.pre_data_h5, ph_len = args.te_ph_len_json, mvn_pa = args.mvn_pa, vae_name = VAE_MODEL, vae_in_dim = VAE_IN_DIM, vae_hid = VAE_HID, seg_len = VAE_SEG_LEN, lda_post = TEST_LDAPOST, lda_t = LDA_TOPICS, lda_d = LDA_COMP, lda_e = LDA_EPOCHS, use_cuda = args.use_cuda, ref_file = REF_FILE, ref_mode = args.ref_mode, gop = args.gop, vae = args.vae, lda = args.lda)

    print(f"Train seg len: {len(trainset.samples) }")
    print(f"Test seg len: {len(testset.samples) }")
    
    
    f_h5 = get_enc(testset,model, args.output+'.test', f_h5, args.te_enc_json)
    f_h5 = get_enc(trainset,model, args.output+'.train', f_h5, args.tr_enc_json)
    
    f_h5.close()
    #remember the output is negated. because for the model mispronunciation = 1.
