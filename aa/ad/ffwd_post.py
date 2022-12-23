#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri December 04 15:46:00 2020

@author: Jose Antonio Lopez @ The University of Sheffield

Outputs and saves the posterior probabilities for the INA set

"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from data import INADataset
from models.vae_model import Model,SingleModel
from models.ffwd_model import *
import pandas as pd

import argparse
import os,time, sys
sys.path.append('/share/mini1/res/t/asr/call/childread-nl/its/aa/ad/tools')
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
parser.add_argument('--epst',type=int,default=0)
parser.add_argument('--epend',type=int,default=100)
parser.add_argument('--step',type=int,default=5)
parser.add_argument('--stats_dict',type=str,default='./json/train_stats/')
parser.add_argument('--model_name',type=str,default='VAE')
parser.add_argument('--model',type=str,default='FFWD')
parser.add_argument('--ref_mode',type=str,default='a1')
parser.add_argument('--output',type=str,default='./output/model1.lr')
parser.add_argument('--gop', default=False,action='store_true')
parser.add_argument('--vae', default=False,action='store_true')
parser.add_argument('--lda', default=False,action='store_true')
parser.add_argument('--phseg', default=False,action='store_true')
parser.add_argument('--phone', default=False,action='store_true')
parser.add_argument('--cntxt', default=False,action='store_true')
parser.add_argument('--word', default=False,action='store_true')
parser.add_argument('--preref', default=False,action='store_true')
parser.add_argument('--filt', default=False,action='store_true')
parser.add_argument('--filtdata',type=str,default='FFWD1_filtlo_hi')
parser.add_argument('--ctlo',type=float,default=1.0)
parser.add_argument('--cthi',type=float,default=0.0)
parser.add_argument('--ctep',type=int,default=0)
#######################################################


args = parser.parse_args()
REF_FILE = REF_FILE.replace('REF_MODE', args.ref_mode)


DETAIL_NAME=''
#adjust the name given the inputs used
if args.phone:
    DETAIL_NAME = DETAIL_NAME + f"_ph"
if args.phseg:
    DETAIL_NAME = DETAIL_NAME + f"_phseg"
if args.word:
    DETAIL_NAME = DETAIL_NAME + f"_wd"
if args.cntxt:
    DETAIL_NAME = DETAIL_NAME + f"_ctx"
if args.preref:
    DETAIL_NAME = DETAIL_NAME + f"_pre" 
if args.gop:
    DETAIL_NAME = DETAIL_NAME + f"_gop"
if args.vae:
    DETAIL_NAME = DETAIL_NAME + f"_VAEh{VAE_HID}_VAEep{VAE_EPOCH}"
if args.lda: 
    DETAIL_NAME = DETAIL_NAME + f"_LDAt{LDA_TOPICS}_LDAd{LDA_COMP}_LDAep{LDA_EPOCHS}"
if args.filt: 
    DETAIL_NAME = DETAIL_NAME + f"_filt_ep{args.ctep}_lo{round(args.ctlo, 2 )}_hi{round(args.cthi, 2 )}"
    
args.stats_dict = args.stats_dict + DETAIL_NAME
args.model_name = args.model_name + DETAIL_NAME
args.filtdata = args.filtdata + DETAIL_NAME.split('_filt')[0] + f"_ep{args.ctep}_lo{round( args.ctlo, 2)}_hi{round( args.cthi, 2)}.train"  #_ep20_lo0.4_hi0.7.t
args.log_dir = args.log_dir + DETAIL_NAME
#args.output = args.output + DETAIL_NAME + f"_ep{args.epoch}"
args.output = args.output + DETAIL_NAME# + f"_ep{args.epoch}"

print(args)
    
    
def get_post(dataset,model, output):
#generates the label outputs for each subset

    dataloader = DataLoader(dataset, batch_size=256)
    
    output_list = []
    target_list = []

    for step, batch in enumerate(dataloader):
    
        X = batch[0]
        target = batch[1]
        X_hat = model(X)
        
        output_list = output_list + list( np.round( X_hat.sigmoid().detach().cpu().numpy(), 6).flat )
        target_list = target_list + list( target.detach().cpu().numpy().flat )
        
    df = pd.DataFrame.from_records(dataset.samples)
    
    df[len(df.columns)] = target_list
    df[len(df.columns)] = output_list
        
    df.to_csv(output, sep = ' ', index = False, header = False)
    
    print(f"Output file saved as: {output}")
        

if __name__ == '__main__':

    in_dim = 0 #starting dimension
    
    if args.gop:
        in_dim += 1
    if args.vae:
        in_dim += int(VAE_HID)*int(VAE_SEG_LEN)
    if args.lda:
        in_dim += int(LDA_TOPICS)
    if args.phone:
        in_dim += 1  
    if args.phseg:
        in_dim += 1  
    if args.word:
        in_dim += 1
    if args.cntxt:
        in_dim += 2
    if args.preref:
        in_dim += 1
        
        
    if args.filt:
        if os.path.isfile(args.filtdata):
            filttrainset = INADataset(samp_json_pa = args.tr_ph_json, h5_pa = args.pre_data_h5, ph_len = args.tr_ph_len_json, mvn_pa = args.mvn_pa, vae_name = VAE_MODEL, vae_in_dim = VAE_IN_DIM, vae_hid = VAE_HID, seg_len = VAE_SEG_LEN, lda_post = TRAIN_LDAPOST, lda_t = LDA_TOPICS, lda_d = LDA_COMP, lda_e = LDA_EPOCHS, use_cuda = args.use_cuda, ref_file = REF_FILE, ref_mode = args.ref_mode, gop = args.gop, vae = args.vae, lda = args.lda, phseg = args.phseg, phone = args.phone, cntxt = args.cntxt, word = args.word, preref = args.preref, filt = args.filtdata ) 
            #maybe later you can also load the reject train set to see how the model performs.
            print(f"Filtered train seg len: {len(filttrainset.samples) }")
            
        else:
            print(f"File {args.filtdata}\n Does not exist.")
            sys.exit()

    trainset = INADataset(samp_json_pa = args.tr_ph_json, h5_pa = args.pre_data_h5, ph_len = args.tr_ph_len_json, mvn_pa = args.mvn_pa, vae_name = VAE_MODEL, vae_in_dim = VAE_IN_DIM, vae_hid = VAE_HID, seg_len = VAE_SEG_LEN, lda_post = TRAIN_LDAPOST, lda_t = LDA_TOPICS, lda_d = LDA_COMP, lda_e = LDA_EPOCHS, use_cuda = args.use_cuda, ref_file = REF_FILE, ref_mode = args.ref_mode, gop = args.gop, vae = args.vae, lda = args.lda, phseg = args.phseg, phone = args.phone, cntxt = args.cntxt, word = args.word, preref = args.preref)
    testset = INADataset(samp_json_pa = args.te_ph_json, h5_pa = args.pre_data_h5, ph_len = args.te_ph_len_json, mvn_pa = args.mvn_pa, vae_name = VAE_MODEL, vae_in_dim = VAE_IN_DIM, vae_hid = VAE_HID, seg_len = VAE_SEG_LEN, lda_post = TEST_LDAPOST, lda_t = LDA_TOPICS, lda_d = LDA_COMP, lda_e = LDA_EPOCHS, use_cuda = args.use_cuda, ref_file = REF_FILE, ref_mode = args.ref_mode, gop = args.gop, vae = args.vae, lda = args.lda, phseg = args.phseg, phone = args.phone, cntxt = args.cntxt, word = args.word, preref = args.preref)

    
    print(f"seg len: {len(testset.samples) }")
    
    if args.epst == args.epend:
        args.epend = args.epst + 1
        args.step = 1
    
    for epoch in np.arange(args.epst, args.epend, args.step):
    
    
        #build the feedforward model
        model = eval(args.model)(in_dim=in_dim, num_cls=1)
        load_model_dir = os.path.join(args.model_dir, args.model_name +'_'+str(epoch)+'.pkl')
        print("This is lad_model_dir")
        print(load_model_dir)
        if not os.path.exists(load_model_dir):
            raise Exception
        model_f = open(load_model_dir,'rb')
        state_dict = torch.load(model_f)
        model.load_state_dict(state_dict['model'])
        model.eval()
        if use_cuda:
            model.cuda()
    
    
        get_post(testset,model, args.output + f"_ep{epoch}" +'.test')
        
        if args.filt:
            get_post(filttrainset,model, args.output + f"_ep{epoch}" +'.filttrain')
            
        get_post(trainset,model, args.output + f"_ep{epoch}" +'.train')
        #remember the output is negated. because for the model mispronunciation = 1.
