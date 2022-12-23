#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Thur 11 16:16:00 2021
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
from models.attention_network import LstmNet
from models.attention_network import Add_N_Norm
from models.attention_network import Predictor
from models.attention_network import CNNPredictor
from models.attention_network import BahdanauAttention
from welford import Welford
import pandas as pd
import numpy as np
import load_txt
import save_txt
import intersection
##sklearn things
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
#import libraries for graphics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
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
parser.add_argument('--roc_dir',type=str,default='./roc_curves/')
parser.add_argument('--output',type=str,default='./output/model1.lr')
parser.add_argument('--clas_rep_dir',type=str,default='./performance_reports/model1.lr')
parser.add_argument('--misp_dir',type=str,default='./performance_reports/model1.lr')
parser.add_argument('--model_name',type=str,default='VAE')
parser.add_argument('--ref_mode',type=str,default='a1')
parser.add_argument('--ph_mode',type=str,default='ax')
parser.add_argument('--cl_tr',type=str,default='auto')
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
parser.add_argument('--model_epoch',type=int,default=0)
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
    
args.model_name = args.model_name + DETAIL_NAME

args.mvn_pa = args.mvn_pa +f".w_{args.wind}.str_{args.str}"

if args.ctf > 0:
    args.mvn_pa = args.mvn_pa +f".ctf_{args.ctf}"
    args.win_samp_json = args.win_samp_json +f".ctf_{args.ctf}"
    args.win_label_json = args.win_label_json +f".ctf_{args.ctf}"
    
args.misp_dir = args.misp_dir + DETAIL_NAME
args.clas_rep_dir = args.clas_rep_dir + DETAIL_NAME
args.output = args.output + DETAIL_NAME

print(args)

### ----------------------------------------- seed
seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


### ----------------------------------------- Save dataframe as a wiki table format
def df2wiki(dataframe, save_path, float_format = '.4f'):
	#this writes a table in a format that can be pasted into the wiki
	#save_path (string) : the directory to save the file into
	
		
	if '.wiki' not in save_path:
		if save_path[-1] == '.':
			dataframe_name = save_path + 'wiki'
		else:
			dataframe_name = save_path + '.wiki'
	
	headers_list = list(dataframe)
	
	headers_list = ["||" + header for header in headers_list]
	headers_list = [header+" " for header in headers_list]
	headers_list[-1] = headers_list[-1]+"||"
	headers = np.array(headers_list)
	values = dataframe.values
	
	
	float_format = "{:"+float_format+"}"
	
	list_of_lists = list()
	for i in range(values.shape[0]):
		row_list = list()
		for j in range(values.shape[1]):
			if isinstance(values[i][j], float):
				row_list.append( "||"+float_format.format(values[i][j])+" " )
			else:
				row_list.append( "||"+str(values[i][j])+" ")
			if j == range(values.shape[1])[-1]:
				row_list[-1] = row_list[-1]+"||"
		
		list_of_lists.append(row_list)		
	
	#The none is for fixing the dimensions and make them able to concatenate
	values = np.array(list_of_lists)
	values = np.concatenate((headers[None,:], values), axis = 0)
	
	np.savetxt(dataframe_name, values, fmt='%s')
	
	print("DataFrame saved in a wiki-tabular format as:")
	print(dataframe_name)

### ----------------------------------------- load model
def load_model(pretrained_model, network, TRAIN_MODE):
#    [encoder, attention, predictor, domainclassifier, optimizer] = network
    encoder, attention, add_n_norm, predictor, optimizer = network

    if use_cuda:
        checkpoint = torch.load(pretrained_model)
    else:
        checkpoint = torch.load(pretrained_model, map_location=lambda storage, location: storage)
        
    epoch = checkpoint['epoch']
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
    

def print_single_ROC(x_tr, y_tr, n_tr, title, ROC_DIR, class_name ):
# this method prints labeled ROC
# the ROC prints the false positive rate vs true positive rate

        intx, inty = intersection.intersection(x_tr, y_tr, np.linspace(0, 1, 100), np.linspace(1, 0, 100))

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 1.1, 0.1)

        savename = os.path.join(ROC_DIR,title+'.png')

        fig, ax = plt.subplots()
        #ax.scatter(x, y)
        ax.plot(x_tr, y_tr, label=class_name)

        ax.set(xlim=(0, 1), ylim=(0, 1))
        
        #select the closest n 
        difx = [abs(intx - i ) for i in x_tr ]
        argmin = difx.index(min(difx))
        
        ax.annotate("{:.2f}".format(float(n_tr[argmin])), (x_tr[argmin], y_tr[argmin]))

        # for i, txt in enumerate(n):
            # ax.annotate("{:.2f}".format(float(txt)), (x[i], y[i]))
        
        ax.legend()
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(which='major', alpha=0.5)
        plt.xlabel("FPR")
        #plt.plot(x,y, 'o-')
        plt.plot([0,1],[1,0], '--')
        
        if inty.size != 0:
            plt.plot(intx, inty, '*k')
            plt.plot([0, intx], [inty, inty], '-.')
            plt.plot([intx, intx], [0, inty], '-.')
            
        plt.ylabel("TPR")
        plt.title('ROC CURVE')
        #plt.grid()
        plt.savefig(savename)
        
        print("ROC curve printed as:")
        print(savename)
        
        return n_tr[argmin]   
    
def print_ROC(x_tr, x_te, y_tr, y_te, n_tr, n_te, title, ROC_DIR ):
# this method prints labeled ROC
# the ROC prints the false positive rate vs true positive rate

        intx, inty = intersection.intersection(x_tr, y_tr, np.linspace(0, 1, 100), np.linspace(1, 0, 100))

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 1.1, 0.1)

        savename = os.path.join(ROC_DIR,title+'.png')

        fig, ax = plt.subplots()
        #ax.scatter(x, y)
        ax.plot(x_tr, y_tr, label='train')
        ax.plot(x_te, y_te, label='test')
        ax.set(xlim=(0, 1), ylim=(0, 1))
        
        #select the closest n 
        difx = [abs(intx - i ) for i in x_tr ]
        argmin = difx.index(min(difx))
        
        ax.annotate("{:.2f}".format(float(n_tr[argmin])), (x_tr[argmin], y_tr[argmin]))

        # for i, txt in enumerate(n):
            # ax.annotate("{:.2f}".format(float(txt)), (x[i], y[i]))
        
        ax.legend()
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(which='major', alpha=0.5)
        plt.xlabel("FPR")
        #plt.plot(x,y, 'o-')
        plt.plot([0,1],[1,0], '--')
        
        if inty.size != 0:
            plt.plot(intx, inty, '*k')
            plt.plot([0, intx], [inty, inty], '-.')
            plt.plot([intx, intx], [0, inty], '-.')
            
        plt.ylabel("TPR")
        plt.title('ROC CURVE')
        #plt.grid()
        plt.savefig(savename)
        
        print("ROC curve printed as:")
        print(savename)
        
        return n_tr[argmin]

### ----------------------------------------- model initialisation
def model_init(args, TRAIN_MODE):
    # model
    encoder = LstmNet(args.input_size, args.lstm_hid, args.lstm_layers )
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
    
    
### ----------------------------------------- get the classification thresholds
def collect_thresholds(df_target,df_output, nclass, args,phlabels):
#collect the per class classification thresholds

    cl_thres = []
    
    #collect the thresholds
    if args.cl_tr == 'auto':
        for c, ph in zip(range(nclass), phlabels):
            fpr, tpr, thres = roc_curve(df_target[:,c], df_output[:,c], pos_label=1)
            treshold = print_single_ROC(fpr, tpr, thres, 'ROC_'+ph, args.roc_dir, ph )
            fnr = 1 - tpr
            eer_thres = thres[np.nanargmin(np.absolute((fnr - fpr)))]
                
            cl_thres.append(eer_thres)
            print(f"{ph} thres: {eer_thres}")
            #print(f"{ph} graph thres: {treshold}")
                
            
    else:
        cl_thres = [ float(args.cl_tr) ] * nclass
            
    return cl_thres
        
        
        
### ----------------------------------------- label the examples using a list of thresholds
def label_posteriors(df_output, cl_thres):   
#for every class column, label the example using the corresponding
#class thresshold from a ordered list -> nclass == phone class_i
  
                #score the probabilities
    for col, thres in enumerate(cl_thres):
        df_output[:,col][df_output[:,col] >= thres ] = 1.0
        df_output[:,col][df_output[:,col] < thres ] = 0.0
        
    return df_output    
    
def gen_outputdf(ph_mode, output_path, subset, args):
#generates the label dataframe for a subset
    dataset = INAWinData(args.win_samp_json+'.'+ subset, args.win_label_json+'.'+ subset ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = ph_mode, subset = subset, ref_mode = args.ref_mode )
    
    dataloader = DataLoader(dataset, batch_size=256)
    target_list = []
    
    for step, (feat, target) in enumerate(dataloader):
        tar = target.squeeze(1)
        target_list = target_list + tar.detach().cpu().numpy().tolist()
    
    df_sample = pd.DataFrame.from_records(dataset.samples)
    df_target = pd.DataFrame(target_list)
    df_target2save = pd.concat([df_sample, df_target], axis=1)
    df_target2save.to_csv(output_path, sep = ' ', index = False, header = False)
    print(f"Target df saved as: {output_path}")
    
    
### ----------------------------------------- get posterior for a subset 
def get_post(dataset,network, output_path, subset):
#generates the label outputs for each subset
    encoder, attention, add_n_norm, predictor, optimizer = network
    
    dataloader = DataLoader(dataset, batch_size=256)
    
    output_list = []
    target_list = []
    
    
    for step, (feat, target) in enumerate(dataloader):
        hyp = encoder(feat)
        context, bahd_w = attention(hyp, hyp)
        emb = add_n_norm(hyp, context)
        output = predictor(emb)
        tar = target.squeeze(1)
        
        output_list = output_list + np.round( output.sigmoid().detach().cpu().numpy(), 4).tolist()
        target_list = target_list + tar.detach().cpu().numpy().tolist()
        

    df_sample = pd.DataFrame.from_records(dataset.samples)
    
    df_output = pd.DataFrame(output_list)
    df_target = pd.DataFrame(target_list)
    
    df_target2save = pd.concat([df_sample, df_target], axis=1)
    df_ouput2save = pd.concat([df_sample, df_output], axis=1)
    
    df_target2save.to_csv(f"{output_path[:output_path.rfind('_ep')]}.target.{subset}", sep = ' ', index = False, header = False)
    df_ouput2save.to_csv(f"{output_path}.output.{subset}", sep = ' ', index = False, header = False)
    print(f"Output file saved as: {output_path}.output.{subset}")
    print(f"Target file saved as: {output_path[:output_path.rfind('_ep')]}.target.{subset}") 

    return f"{output_path}.output.{subset}", f"{output_path[:output_path.rfind('_ep')]}.target.{subset}"
    
    
    
### ----------------------------------------- print the classification report
def print_classification_report(df_target, df_output, phlabels, output_path ):
    class_output = classification_report(df_target.astype(int), df_output.astype(int), target_names=phlabels, output_dict = True)
    
    df = pd.DataFrame(class_output).transpose()
    classlist = phlabels + [ 'micro avg' , 'macro avg', 'weighted avg', 'samples avg' ]
    df['classes']=classlist

    df= df[['classes', 'precision', 'recall', 'f1-score'] ]

    df_avg = df.iloc[-4:,:] 
    df_class = df.iloc[:-4,:] 
    #save class accuracy report
    df2wiki(df_class, output_path, float_format = '.4f')
    
    return df_avg
    
    
    
### ----------------------------------------- infers p(at least one error) = 1 - p(all phones are correct) = 1 - prod_i p(ph_i is correct)
def prod_pherror_prob( output_matrix, miss_target_matrix, phlabels_miss):  
#the output matrix is on the 'all' format. designed to predict the probability of a correct phone

    #the output matrix has the probability of a phone being correct. and zero for the phones not considered.
    #hence for a product of probabilities, replace the zeros with ones so they have no effect on the
    #product
    output_matrix[output_matrix == 0.0] = 1.0
    
    #1 - P(all phones are correct)
    prob_atleast_one_error = 1- np.prod(output_matrix, 1)
    
    if np.isnan(prob_atleast_one_error).any():
        print("There is a nan in prob_atleast_one_error")
    
        #prob_atleast_one_error[np.isnan(prob_atleast_one_error)] = 0.0
    
    #select the columnds from the target matrix that correspond to a mispronunciation    
    error_columns = [ i for i,ph in enumerate(phlabels_miss) if '_0' in ph ] 
    
    #to make it faster, if any flag for phone_0 = 1, label the segment as 1
    error_target = np.minimum(miss_target_matrix[:,error_columns].sum(1), np.ones(miss_target_matrix.shape[0]))
    
    return prob_atleast_one_error, error_target
    
### ----------------------------------------- sum the probabilities of detecting ph_error in phmiss mode model
def sum_pherror_prob( miss_output_matrix, miss_target_matrix, phlabels_miss):  
    #for every example. sum the probability that an error occur. Try first with normalisation
    error_columns = [ i for i,ph in enumerate(phlabels_miss) if '_0' in ph ] 
    
    error_sum = miss_output_matrix[:,error_columns].sum(1)
    total_sum = miss_output_matrix.sum(1)
    
    error_output = error_sum/total_sum
    
    if np.isnan(error_output).any():
        error_output[np.isnan(error_output)] = 0.0
    
    #to make it faster, if any flag for phone_0 = 1, label the segment as 1
    error_target = np.minimum(miss_target_matrix[:,error_columns].sum(1), np.ones(miss_target_matrix.shape[0]))
    
    return error_output, error_target
    
    
### ----------------------------------------- generates a phone class target matrix where 1= the phone is a mispronunciation. This for the 'all' mode (detect correct phone)
def gen_error_target(matrix_output_path,matrix_target_path, nclass_miss, nclass_all, phlabels_all, phlabels_miss):

    target_matrix = pd.read_csv(matrix_target_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_miss)).to_numpy()
    
    output_matrix = pd.read_csv(matrix_output_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_all)).to_numpy()
    error_target_matrix = np.zeros_like(output_matrix)
    del output_matrix
    
    #generate the indices for th ephone classes
    dictOfPh = { phlabels_all[i]: i for i in range(0, len(phlabels_all) ) }
    miss_dict = {}
    for i, ph in enumerate(phlabels_miss):
        if '_0' in ph:
            miss_dict[i]=[dictOfPh[ph.split('_')[0]], i]
            
    #generate the mask for the phone ocurrences
    for ph in miss_dict:
        error_target_matrix[:,miss_dict[ph][0]] += target_matrix[:,miss_dict[ph][1]]    
             
    return error_target_matrix

    
    
### ----------------------------------------- apply phone ocurrence mask
def apply_ph_occurence_mask(matrix_output_path,matrix_target_path, nclass_miss, nclass_all, phlabels_all, phlabels_miss, ph_mode):
#apply a mask for the output where output is zero if we dont expect a phone to occurr
            
    #once we know it exists. load the target dataframes
    target_matrix = pd.read_csv(matrix_target_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_miss)).to_numpy()
    
    if ph_mode == 'miss':
    
        output_matrix = pd.read_csv(matrix_output_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_miss)).to_numpy()
        ph_mask_matrix = target_matrix.astype(float).copy()
        #generate the indices for th ephone classes
        dictOfPh = {}
        
        for i, ph in enumerate(phlabels_miss):
            if ph[:-1] in dictOfPh: #last digit is the quality of pronunciation
                dictOfPh[ph[:-1]].append(i)
            else:
                dictOfPh[ph[:-1]] = [i]  
                
        #generate the mask for the phone ocurrences
        for i, ph in enumerate(phlabels_miss):
        
            ph_mask_matrix[:,dictOfPh[ph[:-1]][0]] += ph_mask_matrix[:,dictOfPh[ph[:-1]][1]]
            ph_mask_matrix[:,dictOfPh[ph[:-1]][1]] += ph_mask_matrix[:,dictOfPh[ph[:-1]][0]]
        

        ph_mask_matrix[ph_mask_matrix>1.0] = 1.0 
        
    else: #'all'
    
        output_matrix = pd.read_csv(matrix_output_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_all)).to_numpy()
        
        #2 masks. one with the precense of the phone and one with the prescence of a mispronunciation
        #generate the mask for the prescence of the phone
        ph_mask_matrix = np.zeros_like(output_matrix)
        #generate the indices for th ephone classes
        dictOfPh = { phlabels_all[i]: i for i in range(0, len(phlabels_all) ) }
        miss_dict = {}
        
        #since the labels just indicate a correct phoneme event, we need to check the error label as well to know
        #where to look for an expected phoneme
        for i, ph in enumerate(phlabels_miss):
            miss_dict[i]=[dictOfPh[ph.split('_')[0]], i]
        
        #generate the mask for the phone ocurrences
        for ph in miss_dict:
            ph_mask_matrix[:,miss_dict[ph][0]] += target_matrix[:,miss_dict[ph][1]]
            
        ph_mask_matrix[ph_mask_matrix>1.0] = 1.0 
    
    #multiply the mask with the outputs.
    return output_matrix * ph_mask_matrix

if __name__ == '__main__':


    #check if the required model exist or not. to save gpu time.
    if args.model_epoch != 0:
        load_model_dir = os.path.join(args.save_model_dir, args.model_name +'_'+str(args.model_epoch)+'.pkl')
        if not os.path.exists(load_model_dir):
            print(f"Model file does not exist:\n{load_model_dir}")
            sys.exit()
        else:
            print(f"Model file found:\n {load_model_dir}")
    
    test_output_path = f"{args.output}_ep{args.model_epoch}.output.test"
    test_target_path = f"{args.output}.target.test"
    train_output_path = f"{args.output}_ep{args.model_epoch}.output.train"
    train_target_path = f"{args.output}.target.train"
    
    if not( os.path.exists(test_output_path) and os.path.exists(test_target_path) ): 
    
        trainset = INAWinData(args.win_samp_json+'.train', args.win_label_json+'.train' ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = args.ph_mode, subset = 'train', ref_mode = args.ref_mode )
        testset = INAWinData(args.win_samp_json+'.test', args.win_label_json+'.test' ,args.pre_data_h5, mvn_pa = args.mvn_pa, ref_file = args.ref_file, ph_mode = args.ph_mode, subset = 'test', ref_mode = args.ref_mode )
    
        #get data details for formatting the model
        args.input_size = trainset.nfilters
        args.nclass = trainset.nclass
        args.train_possw = trainset.pos_weight
        args.seglen = trainset.seglen
    
        network = model_init(args, False)
        network = load_model(load_model_dir, network, False)
        print(f"Model Loaded:\n {load_model_dir}") 
    
        total_params = 0
        for model in network[:-1]:
            print(model)
            total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
    	
        print(f"Trainable Params: {total_params}")
        
        train_output_path,train_target_path = get_post(trainset,network, args.output + f"_ep{args.model_epoch}" ,'train')
        test_output_path,test_target_path = get_post(testset,network, args.output + f"_ep{args.model_epoch}" ,'test')
        
    #quick score the    outputs
    #GET THE NUMBER OF CLASSES
    if args.ph_mode == 'all' or args.ph_mode == 'miss':
        with open(f"{args.mvn_pa}.{args.ref_mode}.{args.ph_mode}.nclass.pkl","rb") as f:
            nclass = pickle.load(f) 
        with open(f"{args.mvn_pa}.{args.ref_mode}.{args.ph_mode}.phlabels.pkl","rb") as f:
            phlabels = pickle.load(f) 
            
        print("this is phlabels")
        print(phlabels)
        print(f"len: {len(phlabels)}")
        
        
        with open(f"{args.mvn_pa}.{args.ref_mode}.all.nclass.pkl","rb") as f:
            nclass_all = pickle.load(f) 
        with open(f"{args.mvn_pa}.{args.ref_mode}.all.phlabels.pkl","rb") as f:
            phlabels_all = pickle.load(f) 
        with open(f"{args.mvn_pa}.{args.ref_mode}.miss.nclass.pkl","rb") as f:
            nclass_miss = pickle.load(f) 
        with open(f"{args.mvn_pa}.{args.ref_mode}.miss.phlabels.pkl","rb") as f:
            phlabels_miss = pickle.load(f) 
      
        
        #if 'all' test on the segments where we expect the phone to ocurr 
        if args.ph_mode == 'all':
            #start here
            #for every class, list the segments with the occurrence of the phone
            train_miss_target_path = train_target_path.replace('phall', 'phmiss')
            test_miss_target_path = test_target_path.replace('phall', 'phmiss')
            if not( os.path.exists(train_miss_target_path) ): 
                #generate the output dataframe
                gen_outputdf('miss', train_miss_target_path, 'train', args) ##generate the target dataframe. YOu dont need a model output for this
            if not( os.path.exists(test_miss_target_path) ): 
                #generate the output dataframe
                gen_outputdf('miss', test_miss_target_path, 'test', args) 
                
            #select the segments that we know contain an expected phone
            #for ph_mode='all', the method output the probability of the expected phone being correct
            train_all_output_matrix =  apply_ph_occurence_mask(train_output_path,train_miss_target_path, nclass_miss, nclass_all, phlabels_all, phlabels_miss, args.ph_mode)
            test_all_output_matrix =  apply_ph_occurence_mask(test_output_path,test_miss_target_path, nclass_miss, nclass_all, phlabels_all, phlabels_miss, args.ph_mode)
            
            #use the miss target matrix to detect which segments contain a phone
            train_miss_target_matrix = pd.read_csv(train_miss_target_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_miss)).to_numpy()   
            test_miss_target_matrix = pd.read_csv(test_miss_target_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_miss)).to_numpy()

            #to obtain p(at least one error) = 1 - p(all phones are correct)
            train_atleast_1_error_output, train_error_target = prod_pherror_prob( train_all_output_matrix, train_miss_target_matrix, phlabels_miss)
            test_atleast_1_error_output, test_error_target = prod_pherror_prob( test_all_output_matrix, test_miss_target_matrix, phlabels_miss)
            
            fpr_tr, tpr_tr, thresholds_tr = roc_curve(train_error_target, train_atleast_1_error_output, pos_label=1)  
            fpr_te, tpr_te, thresholds_te = roc_curve(test_error_target, test_atleast_1_error_output, pos_label=1)
            
            treshold = print_ROC(fpr_tr, fpr_te, tpr_tr, tpr_te , thresholds_tr, thresholds_te, args.model_name+'_all_1-prodcorrect', args.roc_dir )
            
            output_tr = train_atleast_1_error_output > treshold
            output_tr = output_tr.astype(int)
            target_tr = train_error_target.astype(int)  
        
            output_te = test_atleast_1_error_output > treshold
            output_te = output_te.astype(int)
            target_te = test_error_target.astype(int)
            
            pr_tr, rcll_tr, F1_tr, _ = precision_recall_fscore_support(target_tr, output_tr, pos_label=1, average='binary')
            pr_te, rcll_te, F1_te, _ = precision_recall_fscore_support(target_te, output_te, pos_label=1, average='binary')
        
            df = { 'set': ['train', 'test'], 'precision' : [pr_tr, pr_te], 'recall' : [rcll_tr, rcll_te], 'f1' : [F1_tr, F1_te]}
            df = pd.DataFrame (df, columns = ['set','precision','recall', 'f1'])
            df2wiki(df, f"{args.misp_dir}_ep{args.model_epoch}._all_1-prodcorrect_class_report", float_format = '.4f')

            
        #if 'miss' we test on predicting both correct and incorrect pronunciation 
        if args.ph_mode == 'miss':  
        
            #implement the mask for the condition of expected phonemes
            train_miss_output_matrix =  apply_ph_occurence_mask(train_output_path,train_target_path, nclass_miss, nclass_all, phlabels_all, phlabels_miss, args.ph_mode)
            test_miss_output_matrix =  apply_ph_occurence_mask(test_output_path,test_target_path, nclass_miss, nclass_all, phlabels_all, phlabels_miss, args.ph_mode)
            
            #get the outputs as binary vectors
            train_miss_target_matrix = pd.read_csv(train_target_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_miss)).to_numpy()   
            test_miss_target_matrix = pd.read_csv(test_target_path, sep = ' ', header = None, usecols = np.arange(3, 3 + nclass_miss)).to_numpy()
            
            #sum the probabilities of detecting ph_0
            train_error_output, train_error_target = sum_pherror_prob( train_miss_output_matrix, train_miss_target_matrix, phlabels_miss)
            test_error_output, test_error_target = sum_pherror_prob( test_miss_output_matrix, test_miss_target_matrix, phlabels_miss)
            
            #compute roc curves
            fpr_tr, tpr_tr, thresholds_tr = roc_curve(train_error_target, train_error_output, pos_label=1)  
            fpr_te, tpr_te, thresholds_te = roc_curve(test_error_target, test_error_output, pos_label=1)
            
            #print ROC graph
            treshold = print_ROC(fpr_tr, fpr_te, tpr_tr, tpr_te , thresholds_tr, thresholds_te, args.model_name+'_miss_ph0sum', args.roc_dir )
        
            output_tr = train_error_output > treshold
            output_tr = output_tr.astype(int)
            target_tr = train_error_target.astype(int)  
        
            output_te = test_error_output > treshold
            output_te = output_te.astype(int)
            target_te = test_error_target.astype(int)
            
            print("train out - train tgt")
            for i in range(100):
                print(f"{output_tr[i]} - {target_tr[i]}")
            print("\n test out - test tgt")
            for i in range(100):
                print(f"{output_te[i]} - {target_te[i]}")
            
            pr_tr, rcll_tr, F1_tr, _ = precision_recall_fscore_support(target_tr, output_tr, pos_label=1, average='binary')
            pr_te, rcll_te, F1_te, _ = precision_recall_fscore_support(target_te, output_te, pos_label=1, average='binary')
        
            df = { 'set': ['train', 'test'], 'precision' : [pr_tr, pr_te], 'recall' : [rcll_tr, rcll_te], 'f1' : [F1_tr, F1_te]}
            df = pd.DataFrame (df, columns = ['set','precision','recall', 'f1'])
            df2wiki(df, f"{args.misp_dir}_ep{args.model_epoch}.miss_ph0sum_class_report", float_format = '.4f')

            
    else:
    #the error detection case
        nclass = 1
        output_tr = pd.read_csv(train_output_path, sep = ' ', header = None, usecols = [3]).to_numpy()
        target_tr = pd.read_csv(train_target_path, sep = ' ', header = None, usecols = [3]).to_numpy()
        fpr_te, tpr_te, thresholds_te = roc_curve(target_tr, output_tr, pos_label=1)
        
        output_te = pd.read_csv(test_output_path, sep = ' ', header = None, usecols = [3]).to_numpy()
        target_te = pd.read_csv(test_target_path, sep = ' ', header = None, usecols = [3]).to_numpy()
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(target_tr, output_tr, pos_label=1)  
    
        treshold = print_ROC(fpr_tr, fpr_te, tpr_tr, tpr_te , thresholds_tr, thresholds_te, args.model_name, args.roc_dir )
        
        output_tr = output_tr > treshold
        output_tr = output_tr.astype(int)
        target_tr = target_tr.astype(int)  
        
        output_te = output_te > treshold
        output_te = output_te.astype(int)
        target_te = target_te.astype(int)  
        
        pr_tr, rcll_tr, F1_tr, _ = precision_recall_fscore_support(target_tr, output_tr, pos_label=1, average='binary')
        pr_te, rcll_te, F1_te, _ = precision_recall_fscore_support(target_te, output_te, pos_label=1, average='binary')
        
        df = { 'set': ['train', 'test'], 'precision' : [pr_tr, pr_te], 'recall' : [rcll_tr, rcll_te], 'f1' : [F1_tr, F1_te]}
        df = pd.DataFrame (df, columns = ['set','precision','recall', 'f1'])
        df2wiki(df, f"{args.misp_dir}_ep{args.model_epoch}.class_report.test", float_format = '.4f')
        
