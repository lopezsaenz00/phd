#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Tue Feb 17 19:32:00 2020
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
sys.path.append('/share/mini1/res/t/asr/call/childread-nl/its/aa/ad_lstm/tools')
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
parser.add_argument('--win_mel_json',type=str,default='json/win_mel.json.x')
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
##network params
parser.add_argument('--lstm_hid',type=int,default=1)
parser.add_argument('--lstm_layers',type=int,default=1)
parser.add_argument('--pred_layers',type=int,default=0)
parser.add_argument('--pred_hid',type=int,default=128)
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

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
if isfloat(args.cw):
    args.cw=round(float(args.cw), 2)
    
DETAIL_NAME = f"_{args.feat}_ph{args.ph_mode}_bs{args.batch_size}_LSTMl{args.lstm_layers}_LSTMh{args.lstm_hid}_DANh{args.att_hid}"

if 'CNN' in args.model_name:
    DETAIL_NAME = DETAIL_NAME + f"_CNNO{args.o_channels}_CNNK{args.cnv_k_size}_CNNSTR{args.cnv_stride}"
    if '-' in args.o_channels:
        args.o_channels = [ int(i) for i in args.o_channels.split('-')]
    else:
        args.o_channels = int(args.o_channels)

if args.pred_layers > 0:
	DETAIL_NAME = DETAIL_NAME + f"_PREDl{args.pred_layers}_PREDh{args.pred_hid}"

if args.nw: 
    DETAIL_NAME = DETAIL_NAME + '_nw'
else:
    DETAIL_NAME = DETAIL_NAME + f"_cw{args.cw}"

DETAIL_NAME= DETAIL_NAME+f".win_{args.wind}.str_{args.str}"

args.stats_dict = args.stats_dict + DETAIL_NAME
args.model_name = args.model_name + DETAIL_NAME
args.mvn_pa = args.mvn_pa +f".w_{args.wind}.str_{args.str}"

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
        print(f"Using possitive weigths:{pos_weight.item()}")
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
    encoder, attention, predictor, optimizer = network

    # save intermediate models
    state = {
        'epoch': epoch,
        'train_loss' : train_loss,
        'test_loss' : test_loss,
        'LEARNING_RATE' : get_lr(optimizer),
        'encoder' : encoder.state_dict(),
        'attention' : attention.state_dict(),
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
    encoder, attention, predictor, optimizer = network

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
    predictor.load_state_dict(checkpoint['predictor'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #print("Loaded model (%s) at epoch (%d) with loss (%.4f) and LEARNING_RATE (%f)" % (pretrained_model, epoch, accumulated_loss, LEARNING_RATE))
    print("Loaded model (%s) at epoch (%d) " % (pretrained_model, epoch))

    if TRAIN_MODE:
        encoder.train()
        attention.train()
        predictor.train()
    else:
        encoder.eval()
        attention.eval()
        predictor.eval()

    return [encoder, attention, predictor, optimizer]

### ----------------------------------------- model initialisation
def model_init(args, TRAIN_MODE):
    # model
    encoder = LstmNet(args.input_size, args.lstm_hid, args.lstm_layers, args.lstm_out, args.nclass)
    attention = Attention( args.lstm_out, args.att_hid)
    predictor = Predictor(args.nclass, args.lstm_out)
#    domainclassifier = DomainClassifier(num_domains, dan_hidden_size, c, num_dom_layers)

    # use cuda
    if use_cuda:
        encoder = encoder.cuda()
        attention = attention.cuda()
        predictor = predictor.cuda()
#        domainclassifier = domainclassifier.cuda()

    # train or test mode
    if TRAIN_MODE: 
        # (useful for batchnorm, dropout)
        encoder.train()
        attention.train()
        predictor.train()
#        domainclassifier.train()
    else:
        encoder.eval()
        attention.eval()
        predictor.eval()
#        domainclassifier.eval()

    params = list(encoder.parameters()) + list(attention.parameters()) + list(predictor.parameters())
    print('Parameters:encoder = %d' % len(list(encoder.parameters())))
    print('Parameters:attention = %d' % len(list(attention.parameters())))
    print('Parameters:predictor = %d' % len(list(predictor.parameters())))
    print('Parameters:total = %d' % len(params))

    # optimizer
    # different update rules - Adam: A Method for Stochastic Optimization
    optimizer = torch.optim.Adam(params, lr=args.lr)

#    return [encoder, attention, predictor, domainclassifier, optimizer]
    return [encoder, attention, predictor, optimizer]
    
#performs update step for the model
def do_step( network ):

    encoder, attention, predictor, optimizer = network
    
    # update weights    
    optimizer.step()
    # zero the gradient
    encoder.zero_grad()
    attention.zero_grad()
    predictor.zero_grad()
    optimizer.zero_grad()
    
    return [encoder, attention, predictor, optimizer]
    
    

def train(tr_loader,te_loader,network, stats_dict, criterion ):

    #decompose network list
    encoder, attention, predictor, optimizer = network

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
        predictor.train()
        
        
#        if LR_schedule == "StepLR":
#            scheduler.step()
#            print("LR scheduler: %s, LR=%f" % (LR_schedule,get_lr(network[-1])))

        #go tru the dataloader
        batchloss = Welford()
        for step, (feat, target) in enumerate(tr_loader):
            incomplete_batch = True
            #train
            hyp = encoder(feat)
            output = attention(hyp, args.lstm_out, args.att_hid, BATCHSIZE=1)
            outputs = predictor(output)
         
            loss = criterion(outputs, target) / args.batch_size
            
            loss.backward()
            
            overall_hyp = np.concatenate((overall_hyp, to_npy(outputs)),axis=0)
            overall_ref = np.concatenate((overall_ref, to_npy(target)),axis=0) 
            
            batchloss( loss.item() * args.batch_size )
            
            if step % args.batch_size == 0 and not(step == 0):
                encoder, attention, predictor, optimizer = do_step( [encoder, attention, predictor, optimizer] )
                epoch_loss( batchloss.mean )
                batchloss = Welford()
                incomplete_batch = False
                
        
        #the last batch is smaller than batch size, hence update
        if incomplete_batch:
            encoder, attention, predictor, optimizer = do_step( [encoder, attention, predictor, optimizer] )
            epoch_loss( batchloss.mean )
            batchloss = Welford()
        
        #compute the loss on the test data
        encoder.eval()
        attention.eval()
        predictor.eval()
        for step, (feat, target) in enumerate(te_loader):
            hyp = encoder(feat)
            output = attention(hyp, args.lstm_out, args.att_hid, BATCHSIZE=1)
            outputs = predictor(output)
            loss = criterion(outputs, target)
            
            batchloss( loss.item() )
            
        stats_dict['test_loss'].append( batchloss.mean )
        stats_dict['train_loss'].append( epoch_loss.mean )
        
        #save model after the epoch
        save_model( epo, [encoder, attention, predictor, optimizer], stats_dict['train_loss'][-1], stats_dict['test_loss'][-1])
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

    trainset = INAWinData(args.win_mel_json+'.train', args.win_label_json+'.train' ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = args.ph_mode)
    testset = INAWinData(args.win_mel_json+'.test', args.win_label_json+'.test' ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = args.ph_mode)
    
    train_dataloader = DataLoader(trainset, batch_size= 1,shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=1)

    args.input_size = trainset.nfilters
    args.nclass = trainset.nclass
    args.train_possw = trainset.pos_weight
    
    print(f"Possitive examples: {trainset.y.count(1)}")
    print(f"Negative examples: {trainset.y.count(0)}")
    network = model_init(args, True)
    
    #create the folder for the model if it doesn't exist
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    
    #build the feedforward model
    #load it first in case there is an error from an early script and you don't waste time loading data
    if args.start_epo != 0:
        load_model_dir = os.path.join(args.save_model_dir, args.model_name +'_'+str(args.start_epo)+'.pkl')

        if not os.path.exists(load_model_dir):
            print(f"Model file does not exist:\n:{load_model_dir}")
        else:
            print(f"Model Loaded:\n {load_model_dir}")
            
            network = load_model(load_model_dir, network, True)
          
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
    
#    # learning rate decay
#    if LR_schedule == "StepLR":
#        scheduler = torch.optim.lr_scheduler.StepLR(network[-1], step_size=LR_size, gamma=LR_factor)     # optimizer
#    elif LR_schedule == "ReduceLROnPlateau":
#        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(network[-1], 'min', patience=LR_size, factor=LR_factor)
    
    train(train_dataloader,test_dataloader,network, stats_dict, critetion )
