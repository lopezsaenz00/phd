#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs November 05 12:59:00 2020

@author: Jose Antonio Lopez @ The University of Sheffield (House arrest)

"""

import os, sys
sys.path.append('/aa/tools')

import argparse
import numpy as np
import load_txt

#import libraries for graphics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main function
def main():

    

    # -----------------------------------------------------------------------------
    # Process command line
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Load a pickle file which is a training log dictionary and plots useful curves.')
    
    parser.add_argument('-dic', metavar='string', nargs=1, help='Pickle dictionary')
    parser.add_argument('-d', metavar='string', nargs=1, help='subset')
    parser.add_argument('-ste', metavar='int', nargs=1, default = 0, help='Start Epoch')
    parser.add_argument('-ende', metavar='int', nargs=1, default = -1, help='End Epoch')
    
    
    
    args = parser.parse_args();     # parsing input comand line
    print(args)
    
    PICKLE_DIC = args.dic[0]    # input gop scp filename train
    START_EPOCH = int(args.ste) # start epoch
    END_EPOCH = int(args.ende) # end epoch
    SUBSET = args.d[0] # parsing SUBSET name
    
    
    #load training dictionary
    #dic keys
    #['epo']
    #['train_loss']
    #['train_mae_loss']
    #['train_kl_loss']
    #['eval_loss']
    #['eval_mae_loss']
    #['eval_kl_loss']
    if os.path.isfile(PICKLE_DIC):
        stats_dict = load_txt.dict_from_txtfile(PICKLE_DIC)
    else:
        raise Exception("Dictionary file not found.\n"+PICKLE_DIC)
        
    #in the case that the dictionary has multiple runs, we need to check how many trainings are actually saved there.
    epochs = stats_dict['epo']
    zeroidx = np.where(np.array(epochs).astype(int) == 0)[0]
    idx = (zeroidx[-1],len( epochs ))
    
    print("The idx are: "+str(idx[0])+':'+str(idx[1]))
    
    ##get the last training log
    train_loss = stats_dict['train_loss'][idx[0]:idx[1]]
    eval_loss = stats_dict['eval_loss'][idx[0]:idx[1]]
    epochs = epochs[idx[0]:idx[1]]
    
    lowest_epoch = train_loss.index(min(train_loss)) if SUBSET=='train' else eval_loss.index(min(eval_loss))
    
    print(f"Set: {SUBSET}")
    print(f"Lowest loss epoch: {epochs[lowest_epoch]}")
    print(f"Test loss: {eval_loss[lowest_epoch]}")
    print(f"TRain loss: {train_loss[lowest_epoch]}")
    
    
if __name__ == '__main__':
    main()
