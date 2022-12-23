#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sun November 29 13:48:00 2020
From Mingjie Chen

@author: Jose Antonio Lopez @ The University of Sheffield

"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
#from tensorboardX import SummaryWriter
from data import INADataset
from models.vae_model import Model,SingleModel
from models.ffwd_model import *

####select if cuda is available
#use_cuda = torch.cuda.is_available()
use_cuda = torch.cuda.is_available()
print(torch.version.cuda)
print(f"use cuda?: {use_cuda}")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

device = torch.device("cuda:"+str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

import argparse
import os,time, sys
sys.path.append('/aa/ad/tools')
import load_txt
import save_txt

from da_env import VAE_IN_DIM, VAE_HID, VAE_SEG_LEN, VAE_EPOCH, VAE_MODEL
from da_env import TRAIN_LDAPOST, TEST_LDAPOST, LDA_TOPICS, LDA_COMP, LDA_EPOCHS
from da_env import REF_FILE


parser = argparse.ArgumentParser("Assessor's Decision")
parser.add_argument('--use_cuda', default=False,action='store_true')
parser.add_argument('--tr_ph_json',type=str,default='json/sample_mfcc.json.train')
parser.add_argument('--te_ph_json',type=str,default='json/sample_mfcc.json.test')
parser.add_argument('--tr_ph_len_json',type=str,default='json/ph_length_plptrain.json')
parser.add_argument('--te_ph_len_json',type=str,default='json/ph_length_plptrain.json')
parser.add_argument('--pre_data_h5',type=str,default='h5/timit_mfcc39.h5')
parser.add_argument('--mvn_pa',type=str,default='mvn/wav_plp39_train', help='Location for the statistics for mvn.')
parser.add_argument('--start_epo',type=int,default=0)
parser.add_argument('--hid',type=int,default=64)
parser.add_argument('--log_dir',type=str,default='runs/vae/')
parser.add_argument('--save_model',default=False,action='store_true')
parser.add_argument('--load_model',default=False,action='store_true')
parser.add_argument('--save_model_dir',type=str,default='./ckpt/vae/')
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--stats_dict',type=str,default='./json/train_stats/')
parser.add_argument('--model_name',type=str,default='VAE')
parser.add_argument('--end_epo',type=int,default=0)
parser.add_argument('--model',type=str,default='FFWD')
parser.add_argument('--ref_mode',type=str,default='a1')
parser.add_argument('--gop', default=False,action='store_true')
parser.add_argument('--vae', default=False,action='store_true')
parser.add_argument('--lda', default=False,action='store_true')
#
parser.add_argument('--phseg', default=False,action='store_true')
parser.add_argument('--phone', default=False,action='store_true')
parser.add_argument('--cntxt', default=False,action='store_true')
parser.add_argument('--word', default=False,action='store_true')
parser.add_argument('--preref', default=False,action='store_true')
parser.add_argument('--nw', default=False,action='store_true')
parser.add_argument('--cw', type=str,default='auto')
#
parser.add_argument('--filt', default=False,action='store_true')
parser.add_argument('--filtdata',type=str,default='FFWD1_filtlo_hi')
parser.add_argument('--ctlo',type=float,default=1.0)
parser.add_argument('--cthi',type=float,default=0.0)
parser.add_argument('--ctep',type=int,default=0)
#######################################################

args = parser.parse_args()
REF_FILE = REF_FILE.replace('REF_MODE', args.ref_mode)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
if isfloat(args.cw):
    args.cw=round(float(args.cw), 2)



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

if args.nw: 
    DETAIL_NAME = DETAIL_NAME + '_nw'
else:
    DETAIL_NAME = DETAIL_NAME + f"_n{args.cw}"
    
    
args.stats_dict = args.stats_dict + DETAIL_NAME
args.model_name = args.model_name + DETAIL_NAME
args.filtdata = args.filtdata + DETAIL_NAME.split('_filt')[0] + f"_ep{args.ctep}_lo{round( args.ctlo, 2)}_hi{round( args.cthi, 2)}.train"      #_ep20_lo0.4_hi0.7.t
args.log_dir = args.log_dir + DETAIL_NAME

print(args)


def train(dataloader,dev_loader,model, stats_dict, pos_weight):

    print(f"batchs {len(dataloader)}",flush=True)
    print(model)
    if args.use_cuda:
        model.cuda()
    #writer = SummaryWriter(args.log_dir)
    if args.nw:
        loss_function = nn.BCEWithLogitsLoss( )
        print("Without possitive weigths")
    else:
        loss_function = nn.BCEWithLogitsLoss( pos_weight = pos_weight )
        print(f"Using possitive weigths:{pos_weight.item()}")
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    end = time.time()
    #for epo in range(args.start_epo,1000):
    st_epoch = args.start_epo
    if args.start_epo != 0:
        st_epoch = args.start_epo + 1
        
    for epo in range(st_epoch, args.end_epo+1):
        stats_dict['epo'].append(epo)
        model.train()
        epo_loss = 0

        for step, batch in enumerate(dataloader):
        	
            X = batch[0]
            target = batch[1]
            
            X_hat = model(X)

            loss = loss_function(X_hat,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epo_loss += loss.item()
            
        if args.save_model:
            checkpoint = {'model': model.state_dict()}
            model_path = args.save_model_dir
            model_name = args.model_name+'_'+str(epo)+'.pkl'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(checkpoint, os.path.join( model_path,model_name) )
            

        loss_tot = epo_loss / len(dataloader)
        print(f"Epoch {epo} loss_tot {loss_tot}",flush=True)
        #writer.add_scalar('train_loss',loss_tot,epo)
        
        stats_dict['train_loss'].append(loss_tot)

        
        model.eval()
        eval_loss_tot = 0

        for batch in dev_loader:
            X = batch[0]
            target = batch[1]

            X_hat = model(X)
            eval_loss = loss_function(X_hat,target)
            eval_loss_tot += eval_loss.item()
            
        eval_loss_tot = eval_loss_tot / len(dev_loader)

        print(f"Epoch {epo} eval_loss {eval_loss_tot} ")

        #writer.add_scalar("eval_loss", eval_loss_tot,epo)
        stats_dict['eval_loss'].append(eval_loss_tot)
        
        print("Lat iteration model saved:")
        print(os.path.join( model_path,model_name))
        
    save_txt.dict2txtfile(stats_dict, args.stats_dict)



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
        
     

    #build the feedforward model
    #load it first in case there is an error from an early script and you don't waste time loading data
    model = eval(args.model)(in_dim=in_dim, num_cls=1)
    if args.load_model or args.start_epo != 0:
        load_model_dir = os.path.join(args.save_model_dir, args.model_name +'_'+str(args.start_epo)+'.pkl')
        print("This is lad_model_dir")
        print(load_model_dir)
        if not os.path.exists(load_model_dir):
            raise Exception
        model_f = open(load_model_dir,'rb')
        state_dict = torch.load(model_f)
        model.load_state_dict(state_dict['model'])


    if args.filt:
        if os.path.isfile(args.filtdata):
            trainset = INADataset(samp_json_pa = args.tr_ph_json, h5_pa = args.pre_data_h5, ph_len = args.tr_ph_len_json, mvn_pa = args.mvn_pa, vae_name = VAE_MODEL, vae_in_dim = VAE_IN_DIM, vae_hid = VAE_HID, seg_len = VAE_SEG_LEN, lda_post = TRAIN_LDAPOST, lda_t = LDA_TOPICS, lda_d = LDA_COMP, lda_e = LDA_EPOCHS, use_cuda = args.use_cuda, ref_file = REF_FILE, ref_mode = args.ref_mode, gop = args.gop, vae = args.vae, lda = args.lda, phseg = args.phseg, phone = args.phone, cntxt = args.cntxt, word = args.word, preref = args.preref, filt = args.filtdata ) 
            #maybe later you can also load the reject train set to see how the model performs.
        else:
            print(f"File {args.filtdata}\n Does not exist.")
            sys.exit()
    
    else:
        trainset = INADataset(samp_json_pa = args.tr_ph_json, h5_pa = args.pre_data_h5, ph_len = args.tr_ph_len_json, mvn_pa = args.mvn_pa, vae_name = VAE_MODEL, vae_in_dim = VAE_IN_DIM, vae_hid = VAE_HID, seg_len = VAE_SEG_LEN, lda_post = TRAIN_LDAPOST, lda_t = LDA_TOPICS, lda_d = LDA_COMP, lda_e = LDA_EPOCHS, use_cuda = args.use_cuda, ref_file = REF_FILE, ref_mode = args.ref_mode, gop = args.gop, vae = args.vae, lda = args.lda, phseg = args.phseg, phone = args.phone, cntxt = args.cntxt, word = args.word, preref = args.preref)
        
    testset = INADataset(samp_json_pa = args.te_ph_json, h5_pa = args.pre_data_h5, ph_len = args.te_ph_len_json, mvn_pa = args.mvn_pa, vae_name = VAE_MODEL, vae_in_dim = VAE_IN_DIM, vae_hid = VAE_HID, seg_len = VAE_SEG_LEN, lda_post = TEST_LDAPOST, lda_t = LDA_TOPICS, lda_d = LDA_COMP, lda_e = LDA_EPOCHS, use_cuda = args.use_cuda, ref_file = REF_FILE, ref_mode = args.ref_mode, gop = args.gop, vae = args.vae, lda = args.lda, phseg = args.phseg, phone = args.phone, cntxt = args.cntxt, word = args.word, preref = args.preref)
        
    
    train_dataloader = DataLoader(trainset, batch_size= 256,shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=256)
    
    #load the training stats dictionary
    if os.path.isfile(args.stats_dict):
        stats_dict = load_txt.dict_from_txtfile(args.stats_dict)
    else:
        stats_dict = {}
        stats_dict['epo'] = []
        stats_dict['train_loss'] = []
        stats_dict['eval_loss'] = []
        
    if args.cw == 'auto':
        pos_weight = FloatTensor([ trainset.pos_weight ])
    else:
        pos_weight = FloatTensor([ args.cw ])
        
    train(train_dataloader,test_dataloader,model, stats_dict, pos_weight)
