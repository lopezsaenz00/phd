#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 17 19:32:00 2020
From Mingjie Chen
From Rossana Milner
From Asif Jalal
check project: blstmatt
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
from models.attention_network import LstmNet
from models.attention_network import Add_N_Norm
from models.attention_network import Predictor
from models.attention_network import CNNPredictor
from models.attention_network import BahdanauAttention
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
parser.add_argument('--win_samp_json',type=str,default='json/win_mel.json.x')
parser.add_argument('--win_label_json',type=str,default='json/win_label.json.x')
parser.add_argument('--ref_file',type=str,default='ref/INA.v1.cut1-6.phone.$ref_mode.ref')
parser.add_argument('--save_model',default=False,action='store_true')
parser.add_argument('--save_model_dir',type=str,default='./ckpt/vae/')
parser.add_argument('--model_name',type=str,default='VAE')
parser.add_argument('--ref_mode',type=str,default='a1')
parser.add_argument('--ph_mode',type=str,default='ax')
parser.add_argument('--stats_dict',type=str,default='./json/train_stats/')
parser.add_argument('--wind',type=str,default='0.5')
parser.add_argument('--str',type=str,default='0.1')
parser.add_argument('--ctf',type=int,default=0)
##network params
parser.add_argument('--lstm_hid',type=int,default=1)
parser.add_argument('--lstm_layers',type=int,default=1)
parser.add_argument('--pred_layers',type=int,default=0)
parser.add_argument('--pred_hid',type=str,default='128')
parser.add_argument('--o_channels',type=str,default='4')
parser.add_argument('--cnv_k_size',type=int,default=3)
parser.add_argument('--cnv_stride',type=int,default=1)
parser.add_argument('--att_hid',type=int,default=1)
parser.add_argument('--att_soft_dim',type=int,default=1)
parser.add_argument('--dropp',type=float,default=0.1)
##training
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--nw', default=False,action='store_true')
parser.add_argument('--cw', type=str,default='auto')
parser.add_argument('--start_epo',type=int,default=0)
parser.add_argument('--end_epo',type=int,default=0)
parser.add_argument('--batch_size',type=int,default=256)
					   
					   

args = parser.parse_args()

DETAIL_NAME = f"_ref{args.ref_mode}_{args.feat}_ph{args.ph_mode}_bs{args.batch_size}_LSTMl{args.lstm_layers}_LSTMh{args.lstm_hid}_ATTh{args.att_hid}_ATTd{args.att_soft_dim}"

if 'CNN' in args.model_name:
    DETAIL_NAME = DETAIL_NAME + f"_CNNO{args.o_channels}_CNNK{args.cnv_k_size}_CNNSTR{args.cnv_stride}"
    if '-' in args.o_channels:
        args.o_channels = [ int(i) for i in args.o_channels.split('-')]
    else:
        args.o_channels = int(args.o_channels)

if args.pred_layers > 0:
	DETAIL_NAME = DETAIL_NAME + f"_PREDl{args.pred_layers}_PREDh{args.pred_hid}"
    
#get the hidden size of the fully connected network  
if '-' in args.pred_hid:
    args.pred_hid = [ int(i) for i in args.pred_hid.split('-')]
else:
    args.pred_hid = int(args.pred_hid)

if args.nw: 
    DETAIL_NAME = DETAIL_NAME + '_nw'
else:
    DETAIL_NAME = DETAIL_NAME + f"_cw{args.cw}"


DETAIL_NAME= DETAIL_NAME+f".win_{args.wind}.str_{args.str}"

if args.ctf > 0:
    DETAIL_NAME= DETAIL_NAME+f".ctf_{args.ctf}"

args.stats_dict = args.stats_dict + DETAIL_NAME
args.model_name = args.model_name + DETAIL_NAME

args.mvn_pa = args.mvn_pa +f".w_{args.wind}.str_{args.str}"
if args.ctf > 0:
    args.mvn_pa = args.mvn_pa +f".ctf_{args.ctf}"
    args.win_samp_json = args.win_samp_json +f".ctf_{args.ctf}"
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

# computes a value that estimates how far away the output is from the target
def define_loss():
    
    if args.cw == 'auto':
        pos_weight = FloatTensor([ trainset.pos_weight ])
    else:
        pos_weight = FloatTensor([ args.cw ])
    
    if args.nw:
        print("Without possitive weigths")
        return nn.BCEWithLogitsLoss( )
    else:
        print(f"Using possitive weigths.")
        return nn.BCEWithLogitsLoss( pos_weight = pos_weight )
    
### ----------------------------------------- Convert to numpy
def to_npy(x):
    # convert tensor to numpy format
    return x.data.cpu().numpy() 
    
### ----------------------------------------- Return current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


    
### ----------------------------------------- save model
def save_model( epoch, network, train_loss, test_loss):
#    [encoder, attention, predictor, domainclassifier, optimizer] = network
    encoder, attention, add_n_norm, predictor, optimizer = network

    # save intermediate models
    state = {
        'epoch': epoch,
        'train_loss' : train_loss,
        'test_loss' : test_loss,
        'LEARNING_RATE' : get_lr(optimizer),
        'encoder' : encoder.state_dict(),
        'attention' : attention.state_dict(),
        'add_n_norm' : add_n_norm.state_dict(), 
        'predictor' : predictor.state_dict(),
#        'domainclassifier': domainclassifier.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }

    #filename = "%s/epoch%03d-samples%d-loss%.10f-LR%.10f.pth.tar" % (SAVEDIR, state['epoch'], state['samples'], state['loss'], state['LEARNING_RATE'])
    model_name = os.path.join( args.save_model_dir, args.model_name+'_'+str(epoch)+'.pkl')
    
    print("Saved model:\n %s" % model_name)
    torch.save(state, model_name)
    
    
    
### ----------------------------------------- load model
def load_model(pretrained_model, network, TRAIN_MODE):
#    [encoder, attention, predictor, domainclassifier, optimizer] = network
    encoder, attention, add_n_norm, predictor, optimizer = network

    if use_cuda:
        checkpoint = torch.load(pretrained_model)
    else:
        checkpoint = torch.load(pretrained_model, map_location=lambda storage, location: storage)
        
    epoch = checkpoint['epoch']
    # train_loss = checkpoint['train_loss']
    # test_loss = checkpoint['test_loss']
    #LEARNING_RATE = checkpoint['LEARNING_RATE']
    encoder.load_state_dict(checkpoint['encoder'])
    attention.load_state_dict(checkpoint['attention'])
    add_n_norm.load_state_dict(checkpoint['add_n_norm'])
    predictor.load_state_dict(checkpoint['predictor'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #print("Loaded model (%s) at epoch (%d) with loss (%.4f) and LEARNING_RATE (%f)" % (pretrained_model, epoch, accumulated_loss, LEARNING_RATE))
    print("Loaded model (%s) at epoch (%d) " % (pretrained_model, epoch))

    if TRAIN_MODE:
        encoder.train()
        attention.train()
        add_n_norm.train()
        predictor.train()
    else:
        encoder.eval()
        attention.eval()
        add_n_norm.eval()
        predictor.eval()

    return [encoder, attention, add_n_norm, predictor, optimizer]

### ----------------------------------------- model initialisation
def model_init(args, TRAIN_MODE):
    # model
    encoder = LstmNet(args.input_size, args.lstm_hid, args.lstm_layers )
    #attention = Attention( args.lstm_hid*2 , args.att_hid)  #input is the lstmnet hidden size *2 because it is bidirectional
    attention = BahdanauAttention(  args.lstm_hid*2, args.lstm_hid*2, args.att_hid, args.att_soft_dim, normalize=False )
    
    if 'CNN' in args.model_name:
        add_n_norm = Add_N_Norm( [args.seglen ,args.lstm_hid*2 ], args.dropp, flat = False)
        predictor = CNNPredictor(o_channels = args.o_channels, in_channel = 1, k_size = args.cnv_k_size, stride = args.cnv_stride, hidden_size = args.pred_hid, num_layers = args.pred_layers, nclasses = args.nclass, w=args.seglen, h =args.lstm_hid*2)
    else:
        add_n_norm = Add_N_Norm(args.lstm_hid*2*args.seglen, args.dropp, flat = True)
        predictor = Predictor(args.nclass, args.lstm_hid*2*args.seglen, args.pred_hid,args.pred_layers)
#    domainclassifier = DomainClassifier(num_domains, dan_hidden_size, c, num_dom_layers)


    # use cuda
    if use_cuda:
        encoder = encoder.cuda()
        attention = attention.cuda()
        add_n_norm = add_n_norm.cuda()
        predictor = predictor.cuda()
#        domainclassifier = domainclassifier.cuda()

    # train or test mode
    if TRAIN_MODE: 
        # (useful for batchnorm, dropout)
        encoder.train()
        attention.train()
        add_n_norm.train()
        predictor.train()
#        domainclassifier.train()
    else:
        encoder.eval()
        attention.eval()
        add_n_norm.eval()
        predictor.eval()
#        domainclassifier.eval()

    params = list(encoder.parameters()) + list(attention.parameters()) + list(add_n_norm.parameters()) + list(predictor.parameters())
    print('Parameters:encoder = %d' % len(list(encoder.parameters())))
    print('Parameters:attention = %d' % len(list(attention.parameters())))
    print('Parameters:add n norm = %d' % len(list(add_n_norm.parameters())))
    print('Parameters:predictor = %d' % len(list(predictor.parameters())))
    print('Parameters:total = %d' % len(params))

    # optimizer
    # different update rules - Adam: A Method for Stochastic Optimization
    optimizer = torch.optim.Adam(params, lr=args.lr)

#    return [encoder, attention, predictor, domainclassifier, optimizer]
    return [encoder, attention, add_n_norm, predictor, optimizer]
    
#performs update step for the model
def do_step( network ):

    encoder, attention, add_n_norm, predictor, optimizer = network
    
    # update weights    
    optimizer.step()
    # zero the gradient
    encoder.zero_grad()
    attention.zero_grad()
    add_n_norm.zero_grad()
    predictor.zero_grad()
    optimizer.zero_grad()
    
    return [encoder, attention, add_n_norm, predictor, optimizer]
    
    

def train(tr_loader,te_loader,network, stats_dict, criterion ):

    #decompose network list
    encoder, attention, add_n_norm, predictor, optimizer = network

    st_epoch = args.start_epo
    if args.start_epo != 0:
        st_epoch = args.start_epo + 1
        
    for epo in range(st_epoch, args.end_epo+1):
        stats_dict['epoch'].append(epo)
        epoch_loss = Welford()
        overall_hyp = np.zeros((0,args.nclass))
        overall_ref = np.zeros((0,args.nclass))
        
        encoder.train()
        attention.train()
        add_n_norm.train()
        predictor.train()

        #go tru the dataloader
        for step, (feat, target) in enumerate(tr_loader):
            #train
            hyp = encoder(feat)
            context, bahd_w = attention(hyp, hyp)
            emb = add_n_norm(hyp, context)
            output = predictor(emb)
            
            if args.nclass > 1:
                loss = criterion(output, target.squeeze(1))
            else:
                loss = criterion(output, target)
            
            loss.backward()
            
            encoder, attention, add_n_norm, predictor, optimizer = do_step( [encoder, attention, add_n_norm, predictor, optimizer] )
            epoch_loss( loss.item() )
            
        
        #compute the loss on the test data
        test_loss = Welford()
        encoder.eval()
        attention.eval()
        add_n_norm.eval()
        predictor.eval()
        for step, (feat, target) in enumerate(te_loader):
            hyp = encoder(feat)
            context, bahd_w = attention(hyp, hyp)
            emb = add_n_norm(hyp, context)
            output = predictor(emb)
            
            if args.nclass > 1:
                loss = criterion(output, target.squeeze(1))
            else:
                loss = criterion(output, target)
            
            test_loss( loss.item() )
            
        stats_dict['test_loss'].append( test_loss.mean )
        stats_dict['train_loss'].append( epoch_loss.mean )
        
        #save model after the epoch
        save_model( epo, [encoder, attention, add_n_norm, predictor, optimizer], stats_dict['train_loss'][-1], stats_dict['test_loss'][-1])
        save_txt.dict2txtfile(stats_dict, args.stats_dict)
        
        
        print(f"Epoch {epo} Train_loss {stats_dict['train_loss'][-1]} Test_loss {stats_dict['test_loss'][-1]}")
        
        

        

if __name__ == '__main__':


    #check if the required model exist or not. to save gpu time.
    if args.start_epo != 0:
        load_model_dir = os.path.join(args.save_model_dir, args.model_name +'_'+str(args.start_epo)+'.pkl')
        if not os.path.exists(load_model_dir):
            print(f"Model file does not exist:\n{load_model_dir}")
            sys.exit()
        else:
            print(f"Model file found:\n {load_model_dir}")

    trainset = INAWinData(args.win_samp_json+'.train', args.win_label_json+'.train' ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = args.ph_mode, subset = 'train', ref_mode = args.ref_mode )
    testset = INAWinData(args.win_samp_json+'.test', args.win_label_json+'.test' ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = args.ph_mode, subset = 'test', ref_mode = args.ref_mode )
    
    train_dataloader = DataLoader(trainset, batch_size= args.batch_size ,shuffle=True)
    test_dataloader = DataLoader(testset, batch_size= args.batch_size )

    args.input_size = trainset.nfilters
    args.nclass = trainset.nclass
    args.train_possw = trainset.pos_weight
    args.seglen = trainset.seglen
    if args.nclass == 1:
        print(f"Tr Possitive examples: {trainset.y.count(1)}")
        print(f"Tr Negative examples: {trainset.y.count(0)}")
        print(f"Te Possitive examples: {testset.y.count(1)}")
        print(f"Te Negative examples: {testset.y.count(0)}")
    
    network = model_init(args, True)
    
    #create the folder for the model if it doesn't exist
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    
    #build the feedforward model
    #load it first in case there is an error from an early script and you don't waste time loading data
    if args.start_epo != 0:
            network = load_model(load_model_dir, network, True)
            print(f"Model Loaded:\n {load_model_dir}")   
          
    total_params = 0
    for model in network[:-1]:
    	print(model)
    	total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
    	
    print(f"Trainable Params: {total_params}")
    
    
    #load the training stats dictionary
    if os.path.isfile(args.stats_dict):
        stats_dict = load_txt.dict_from_txtfile(args.stats_dict)
    else:
        stats_dict = {}
        stats_dict['epoch'] = []
        stats_dict['train_loss'] = []
        stats_dict['test_loss'] = []
    
    critetion = define_loss()
    
    train(train_dataloader,test_dataloader,network, stats_dict, critetion )
