#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri December 04 15:46:00 2020

@author: Jose Antonio Lopez @ The University of Sheffield

generates histograms for the posteriors

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
from matplotlib.ticker import PercentFormatter

import argparse
import os,time, sys
sys.path.append('/aa/ad/tools')
import load_txt
import save_txt

#import libraries for graphics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


from da_env import VAE_IN_DIM, VAE_HID, VAE_SEG_LEN, VAE_EPOCH, VAE_MODEL
from da_env import TRAIN_LDAPOST, TEST_LDAPOST, LDA_TOPICS, LDA_COMP, LDA_EPOCHS
from da_env import REF_FILE

use_cuda = torch.cuda.is_available()


parser = argparse.ArgumentParser("Assessor's Decision")
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--epst',type=int,default=0)
parser.add_argument('--epend',type=int,default=100)
parser.add_argument('--step',type=int,default=5)
parser.add_argument('--ref_mode',type=str,default='a1')
parser.add_argument('--output',type=str,default='./output/model1.lr')
parser.add_argument('--postfile',type=str,default='./hist/model1.lr')
parser.add_argument('--gop', default=False,action='store_true')
parser.add_argument('--vae', default=False,action='store_true')
parser.add_argument('--lda', default=False,action='store_true')
parser.add_argument('--phseg', default=False,action='store_true')
parser.add_argument('--phone', default=False,action='store_true')
parser.add_argument('--cntxt', default=False,action='store_true')
parser.add_argument('--word', default=False,action='store_true')
parser.add_argument('--preref', default=False,action='store_true')
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
    

args.output = args.output + DETAIL_NAME# + f"_ep{args.epoch}"
args.postfile = args.postfile + DETAIL_NAME

print(args)


def plot_histogram_trvste(output, scorestr, scoreste):
#this method reads the ref file.
    #plot the score histogram
    
    min_ = min(scorestr + scoreste)
    max_ = max(scorestr + scoreste)
    
    histdir = os.path.join( os.path.dirname(output), 'trvste')
    if not os.path.exists(histdir):
        os.makedirs(histdir)
        
    output = os.path.join( histdir , os.path.basename(output))
        
    mean = np.mean(scorestr + scoreste)
    var = np.var(scorestr + scoreste)
    
    fig, ax = plt.subplots()
    plt.title(output[output.rfind('/')+1:])
    ax.hist(scorestr, bins=30, density = True, color= 'r', alpha=0.5, label='tr')
    ax.hist(scoreste,  bins=30, density = True, color= 'b', alpha=0.5, label='te')
    plt.legend(loc='upper right')
    #plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
            
    #plt.text(xlim[1] * 4/5.0, ylim[1]*3/4.0, r'$\mu=$'+str(round(mean,2))+'\n'+r'$\sigma=$'+str(round(var,2)))
    
    plt.ylabel('Freq')
    plt.xlabel('segment posterior')
    plt.savefig(output+'_trte.png')
    plt.close()
    
    
def plot_histogram_target(output, targettr, scorestr, targette, scoreste):
#this method reads the ref file.
    #plot the score histogram
    
    histdir = os.path.join( os.path.dirname(output), 'target')
    if not os.path.exists(histdir):
        os.makedirs(histdir)
    
    output = os.path.join( histdir , os.path.basename(output))
    
    #get the indeces for the posteriors of errors, target = 1
    idx1tr = [ i for i in range(len(targettr)) if targettr[i] == 1 ]
    idx0tr = [ i for i in range(len(targettr)) if targettr[i] == 0 ]
    del targettr
    idx1te = [ i for i in range(len(targette)) if targette[i] == 1 ]
    idx0te = [ i for i in range(len(targette)) if targette[i] == 0 ]
    del targette
    
    
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(output[output.rfind('/')+1:])
    

    #normalizing weights for the train set
    oneweights = 100 * np.ones_like(np.array( idx1tr ) ) / np.array( idx1tr ).size
    zeroweights = 100 * np.ones_like(np.array( idx0tr ) ) / np.array( idx0tr ).size
    #oneweights = 100 * np.ones_like(np.array( [scorestr[i] for i in idx1tr]) ) / np.array( idx1tr + idx0tr).size
    #zeroweights = 100 * np.ones_like(np.array( [scorestr[i] for i in idx0tr]) ) / np.array(  idx1tr + idx0tr).size 
    
    #firstcorrects
    axes[0].hist(np.array( [scorestr[i] for i in idx1tr]), weights=oneweights  ,color='r', alpha=0.5, label='err')
    axes[0].hist(np.array( [scorestr[i] for i in idx0tr]), weights=zeroweights ,color='b', alpha=0.5, label='ok')
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[0].set_title('train')
    axes[0].set(xlabel='segment posterior', ylabel='% Freq')
    
    #test set
    oneweights = 100 * np.ones_like(np.array( idx1te) ) / np.array( idx1te).size
    zeroweights = 100 * np.ones_like(np.array(idx0te ) ) / np.array( idx0te ).size
    #oneweights = 100 * np.ones_like(np.array( [scoreste[i] for i in idx1te]) ) / np.array( idx1te + idx0te ).size
    #zeroweights = 100 * np.ones_like(np.array( [scoreste[i] for i in idx0te]) ) / np.array( idx1te + idx0te).size
    #firstcorrects
    axes[1].hist(np.array( [scoreste[i] for i in idx1te]), weights=oneweights , color='r', alpha=0.5, label='err')
    axes[1].hist(np.array( [scoreste[i] for i in idx0te]), weights=zeroweights, color='b', alpha=0.5, label='ok')
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[1].set_title('test')
    axes[1].set(xlabel='segment posterior', ylabel='% Freq')
    
    plt.legend(loc='upper right')
    plt.savefig(output+'_target.png')
    plt.close()
    

if __name__ == '__main__':


    if args.epst == args.epend:
        args.epend = args.epst + 1
        args.step = 1
    
    for epoch in np.arange(args.epst, args.epend, args.step):
    
        #load the posterior file
        post_te = pd.read_csv(args.postfile + f"_ep{epoch}" +'.test', sep = ' ', header=None)
        post_te.columns = ['name', 'phidx', 'st', 'end', 'ph', 'wd', 'phtag','target', 'pos']
        post_tr = pd.read_csv(args.postfile + f"_ep{epoch}" +'.train', sep = ' ', header=None)
        post_tr.columns = ['name', 'phidx', 'st', 'end', 'ph', 'wd', 'phtag','target', 'pos']
        
        
        #plot_histogram_trvste(args.output + f"_ep{epoch}" +'.hist',  [ round(elem, 4) for elem in post_tr['pos'].tolist() ],  [ round(elem, 4) for elem in post_te['pos'].tolist() ])
        plot_histogram_target(args.output + f"_ep{epoch}" +'.hist', post_tr['target'].tolist(), [ round(elem, 4) for elem in post_tr['pos'].tolist() ], post_te['target'].tolist(),[ round(elem, 4) for elem in post_te['pos'].tolist() ])
        

