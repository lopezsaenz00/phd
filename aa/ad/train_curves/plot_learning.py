#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs November 05 12:59:00 2020

@author: Jose Antonio Lopez @ The University of Sheffield (Oh pandemic days)

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
    parser.add_argument('-ste', metavar='int', nargs=1, default = 0, help='Start Epoch')
    parser.add_argument('-ende', metavar='int', nargs=1, default = -1, help='End Epoch')
    parser.add_argument('-savedir', metavar='string', nargs=1, default ='./',help='Where to sabe the plot.')
    
    
    
    args = parser.parse_args();     # parsing input comand line
    print(args)
    
    PICKLE_DIC = args.dic[0]    # input gop scp filename train
    START_EPOCH = int(args.ste) # start epoch
    END_EPOCH = int(args.ende) # end epoch
    SAVE_DIR = args.savedir[0]    # output save directory
    
    
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
        
        
    # Plot the train and test loss
    plt.figure(1, figsize=(5,4))
    plt.plot(stats_dict['epo'][idx[0]:], stats_dict['train_loss'][idx[0]:], label = 'Loss_train')
    plt.plot(stats_dict['epo'][idx[0]:], stats_dict['eval_loss'][idx[0]:], label = 'Loss_eval')
    plt.ylim(bottom=0) 
    basename = os.path.basename(PICKLE_DIC)
    plt.title('Total Loss\n'+ basename[:basename.rfind('_VAEh')]+'\n'+ basename[basename.rfind('_VAEh')+1:] )
    
    plt.legend(loc='upper right')
    plt.subplots_adjust(hspace = 0.2)
    plt.savefig(SAVE_DIR+'/'+PICKLE_DIC[PICKLE_DIC.rfind('/')+1:]+'.png', dpi = 300)

    
    
if __name__ == '__main__':
    main()
