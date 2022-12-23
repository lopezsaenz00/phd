#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed August 27 16:12:00 2020

@author: Jose Antonio Lopez @ The University of Sheffield

this script plots ROC curves

"""

import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.append('/aa/ad/tools')
sys.path.append('/aa/ad')
import load_txt
from save_txt import df2wiki

from da_env import REF_FILE
import pandas as pd
import argparse
import numpy as np
import load_txt
import intersection
from welford import Welford
import pandas as pd
import seaborn as sns
import json
import math


#import libraries for graphics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
    
def print_ROC(x, y, n, subset, args ):
# this method prints labeled ROC
# the ROC prints the false positive rate vs true positive rate
# n is a dictionary with the (fpr, tpr) coordinate for every trigger

    intx, inty = intersection.intersection(x, y, np.linspace(0, 1, 100), np.linspace(1, 0, 100))

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 1.1, 0.1)

    MODEL_NAME = os.path.basename(args.label_f) + subset
    plot_title = 'ROC\n'+MODEL_NAME[:MODEL_NAME.rfind('_VAEh')] + '\n'+ MODEL_NAME[MODEL_NAME.rfind('VAEh'):]
    new_title= 'ROC\n'+MODEL_NAME
    savename = os.path.join(args.graph_dir, 'ROC_'+MODEL_NAME+'.png' )

    fig, ax = plt.subplots()
    
    for key in list(n.keys()):
        ax.annotate("{:.2f}".format(float(key)), n[key])
    
    ax.scatter(x, y, label = subset)
    ax.set(xlim=(0, 1), ylim=(0, 1))

    # for i, txt in enumerate(n):
        # ax.annotate("{:.2f}".format(float(txt)), (x[i], y[i]))
        
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(which='major', alpha=0.5)
    plt.xlabel("FPR")
    plt.plot(x,y, 'o-')
    plt.plot([0,1],[1,0], '--')
        
    if inty.size != 0:
        plt.plot(intx, inty, '*k')
        plt.plot([0, intx], [inty, inty], '-.')
        plt.plot([intx, intx], [0, inty], '-.')
            
    plt.ylabel("TPR")
    plt.title(plot_title)
    plt.savefig(savename)
        
    print("ROC curve printed as:")
    print(savename)   
    
    return np.array([intx, inty]) #return the point of equal error rate
    
###################################
        
def gen_ref(ref_file, REF_MODE, samples):
#LOADS THE PRONUNCIATION ANNOTATION AND APPLIES THE INVERSE OF THE
#CORRECTNESS LABEL TO MAKE IT MATCH WITH DE DNN OUTPUT [ 0 = GOOD, 1 = MISPRONOUNCED ]
#REF_FILE the reference file annotated for mispronunciation
#REF_MODE wether is a1, a2, a3, and, maj, or
    
    
    #this method reads the ref file.
    ref = {}
    dref = []
    
    with open(ref_file, 'r') as myfile:
        lines = myfile.read().splitlines()
            
        for line in lines:
            
            if REF_MODE == 'a1' or REF_MODE == 'a2' or REF_MODE == 'a3' :
                line2append = line.split('\t')
            else:
                line2append = line.split(' ')
            init = line2append[0]
                
            if 'ina-' in init:
                
                filename = init
                seg_score = []  
                    
            if len(line2append) == 4 and not( '---' in init):
                seg_score.append(line2append)
                    
            if init == '.':
                ref[filename] = seg_score
	            
        # collect the annotators decision
        for sample in samples:
            samp = sample[0]
            filename = os.path.basename(samp[0])
            file_idx = samp[1]
            #the ref file
            ref_seg = ref[filename][file_idx]
	    
            dref.append( 1 - int(ref_seg[-1]))
            

    return dref
    
    
def test_trigger(scores, annotation, tr):
###
# Build the confusion matrix using a given trigger
###

    #conf matrix
    TP = 0
    TN = 0
    FP = 0
    FN = 0
        
    y_hat = [0 if sc < tr else 1 for sc in scores]
    
    ################################
    su = np.array(annotation) + np.array(y_hat)*2
    su = su.astype(int).tolist()
    
    #assemble the confusion matrix
    TP = su.count(3)
    FP = su.count(2)
    TN = su.count(0)
    FN = su.count(1)
    
    ################################
    
            
    #true positivity rate
    TPR = TP *1.0 / ( TP + FN )*1.0
    #false positive rate
    FPR = FP *1.0 / ( FP + TN  )*1.0
    #precision
    if ( TP + FP ) == 0:
        PR = 0
    else:
        PR = TP*1.0 / (TP + FP)*1.0
    
    #recall
    RCL = TP*1.0 / (TP + FN)*1.0
    #accuracy
    ACC = (TP + TN )*1.0/ (TP + TN + FP + FN)*1.0
    #f1 score
    F1 = 2.0*TP / (2.0*TP + FP*1.0 + FN*1.0 )
    
    return TPR, FPR, PR, RCL, ACC, F1

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    
def gen_roc(scores, annotation, trigger, args):
    ####
    #test the scores with different triggers, build and plot the ROC curves
    ###
    
    TPR_LIST = []
    FPR_LIST = []
    PR_LIST = {}
    RCL_LIST = {}
    ACC_LIST = {}
    F1_LIST = {}
    LEGENDS = {}
    
    for tr in trigger:
        
        TPR, FPR, PR, RCL, ACC, F1 = test_trigger(scores, annotation, tr)
        TPR_LIST.append(TPR)
        FPR_LIST.append(FPR)
        PR_LIST["{:.2f}".format(float(tr))] = PR
        RCL_LIST["{:.2f}".format(float(tr))] = RCL
        ACC_LIST["{:.2f}".format(float(tr))] = ACC
        F1_LIST["{:.2f}".format(float(tr))] = F1
        LEGENDS["{:.2f}".format(float(tr))] = [ FPR, TPR ]
        
        
    return FPR_LIST, TPR_LIST, PR_LIST, RCL_LIST, ACC_LIST, F1_LIST, LEGENDS
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    
def pick_trigger(EER, legends):
#from the true and false positive rates , select the trigger that generate
#the rates closer to the equal error rate in order to obtain the performance
#metris (accuracy, recall, F1)

    dist = 99999
    
    for key in list(legends.keys()):
        x =  (EER[0] - legends[key][0] ) 
        y =  (EER[1] - legends[key][1] ) 
        d = math.sqrt( (x*x) + (y*y) )
        
        if d < dist:
            dist = d
            pick = key

    return pick
    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main function
if __name__ == '__main__':

    # -----------------------------------------------------------------------------
    # Process command line
    # -----------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Load the GOP scores and find the triggers.')
    
    
    parser.add_argument('-ph_json',type=str,default='json/sample_mfcc.json.train')
    parser.add_argument('-ref_mode',type=str,default='a1') 
    parser.add_argument('-label_f',type=str,default='../output/FFWD1_mel80')
    parser.add_argument('-graph_dir',type=str,default='./plots')
    parser.add_argument('-stats_dir',type=str,default='./reports')
    parser.add_argument('-filt', default=False,action='store_true')
    
    args = parser.parse_args();     # parsing input comand line
    
    print(args)
    
    
    #define the annotation file with the ref mode.
    ref_file = REF_FILE.replace('REF_MODE', args.ref_mode)
    
    trigger=np.linspace(0,1, 100) 
    
    if args.filt:
        #setlist = ['filttrain', 'train', 'test'] #you have to sort out the ph_json list for the filtered data
        setlist = ['train', 'test']
    else:
        setlist = ['train', 'test']
    
    for SUBSET in setlist:
    #for SUBSET in [ 'test']:
    
        #the phone sample lists
        SAMPLES_LIST = json.load( open(args.ph_json+SUBSET,'r') )
        #load the assessor's annotation for every phone sample in the list 
        REF = gen_ref(ref_file, args.ref_mode, SAMPLES_LIST)
    
        #load the output file from the dnn
        df = pd.read_csv(args.label_f+SUBSET, sep = ' ', header = None, usecols = [0, 1, 7])
    
        #get the name and segment if for every phone sample
        SAMPLES_LIST = [ x[0] for x in SAMPLES_LIST]
    
        #just to make sure that the scores match the sample
        annotation = []
        for index, row in df.iterrows():
            annotation.append( REF[ SAMPLES_LIST.index( [row[0], row[1]] ) ] )
    
        scores = df[7].to_list()
        #now both scores and annotation are aligned.
    
    
        if SUBSET == 'train':
            fpr_tr, tpr_tr, pr_tr, rcl_tr, acc_tr, f1_tr, legends = gen_roc(scores, annotation, trigger, args)
            EER = print_ROC(fpr_tr, tpr_tr, legends, SUBSET, args )
            #select the trigger which gets closer to the EER coordinate
            pick = pick_trigger(EER, legends)
            
            
        if SUBSET == 'test':
            fpr_te, tpr_te, pr_te, rcl_te, acc_te, f1_te, legends = gen_roc(scores, annotation, trigger, args)
            _ = print_ROC(fpr_te, tpr_te, legends, SUBSET, args )
            
            
    #generate the dataframe to plot
    d =  { 'Trigger': [ str(round(float(pick), 4)), '', '', ''], 'stat': ['Precision', 'Recall', 'F1', 'Accuracy'], 'Train': [pr_tr[pick], rcl_tr[pick], f1_tr[pick], acc_tr[pick]], 'Test': [pr_te[pick], rcl_te[pick], f1_te[pick], acc_te[pick]]}  
    df = pd.DataFrame(data=d)
    
    
    MODEL_NAME = os.path.basename(args.label_f)
    savename = os.path.join(args.stats_dir, MODEL_NAME )
       
    #plot the wikitable with the performance stats
    df2wiki(df, savename, float_format = '.4f')
