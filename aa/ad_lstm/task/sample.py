#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sat October 24 18:56:00 2020
From Mingjie Chen

@author: Jose Antonio Lopez @ The University of Sheffield
Generates short segments of the INA set with a window size and a stride.

"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import json
import numpy as np
import argparse
import copy


#################################################################################
def time2frame(time, win_time = 0.025, hop_time = 0.01, samplerate = 16000):
#from : http://www1.icsi.berkeley.edu/Speech/faq/time-to-frames.html
#win=25ms, hop=10ms and samplerate=8000Hz.
#Then the window is 200 samples
#the hop is 80 samples.
#example. 3.5 hours at 16k are 201600000 samples

	win_samp = int(win_time * samplerate)
	hop_samp = int(hop_time * samplerate)

	#Round (don't truncate) the time to the nearest sample frame
	t_samp = int(round(time * samplerate))

	#Offset the first (win-hop)/2 samples which is the extra bit at
	#the beginning that comes from using wholly-enclosed frames
	t_samp_adj = t_samp - int((win_samp - hop_samp)/2)

	#Divide by the number of samples per hop
	return int(t_samp_adj / hop_samp)
#################################################################################



def gen_samples(subset_ph_seg, audiolen, SEG_LEN, STRIDE, cutoff_frame_edge):
#uses a moving window to generate the spectogram indices and the correpsonding phone labels

    subset_labels = []
    subset_samples = []
    phcounter = []

    #grab a list of the recordings in each subset
    setrec = [ line[0] for line in subset_ph_seg]
    
    
    for audio_item in audiolen:
        rec = audio_item[0]

    
        #grab the segments for said recording
        reclist = [i for i,x in enumerate(setrec) if x == rec]
        reclist = [ subset_ph_seg[i] for i in reclist  ]
        
        
        last_fr = int(audio_item[1])
        last_samp_fr_end = 0
        last_samp_fr_st = 0 - STRIDE
        
        sample_fr_list = []
        

        
        while last_samp_fr_end < last_fr:  
            
            last_samp_fr_st += STRIDE
            last_samp_fr_end = last_samp_fr_st + SEG_LEN
        
            sample_fr_list.append( [rec, last_samp_fr_st, last_samp_fr_end] )
            

        
        #cut down the last window sample to the actual last frame of the recording
        sample_fr_copy =copy.deepcopy(sample_fr_list)
        sample_fr_list = []
        for lol in sample_fr_copy:
            if lol[-1] <= last_fr:
                sample_fr_list.append(lol)
            else:
                sample_fr_list.append(lol[:2]+[last_fr])
        
        #if the length of the any window segment is not the window len, scrap it out
        sample_fr_copy =copy.deepcopy(sample_fr_list)
        sample_fr_list = []
        for lol in sample_fr_copy:
            if (lol[-1]-lol[-2])*1.0 == 1.0*SEG_LEN:
                sample_fr_list.append(lol)
            
        
        ##if no segments, skip
        if sample_fr_list:
            
            sample_label_list = []  #the matching list for the sample_fr_list
                    
            #get the phones present
            for count, sample in enumerate(sample_fr_list):
                st = int(sample[1])
                end = int(sample[2])
                
                sample_window_list = []
                

                
                for segidx, segment in enumerate(reclist):
                

                    
                    phstart =  int(segment[2])
                    phend = int(segment[3])
                    phlen = phend -  phstart
                    

                    
                    #see how much of the phone segment appears inside the sample
                    
                    #if the phone is contained inside the sample
                    if cutoff_frame_edge > 0:
                        if phstart >= (st + cutoff_frame_edge) and phend <= (end - cutoff_frame_edge) :
                            sample_window_list.append([segment[4], segidx, max( st, phstart ), min(end, phend)])
                    
                    
                    else:
                        #if the phone ends inside the sample (dont start in the sample)
                        if phend <= end and phend > st and phstart < st:
                            #if at least 75% the len of the phone is contained in the sample
                            if phlen*0.75 < float(phend - st):
                                #print(f"phlen/2 : {phlen*0.75} < {float(phend - st)}")
                                #add the segment
                                #print("It ends inside the sample")
                                sample_window_list.append([segment[4], segidx, max( st, phstart ), min(end, phend) ])
                        
                        #if the phone starts inside the sample and dont end in the sample
                        elif phstart < end and phstart >= st and phend > end:
                            if phlen*0.75 < float( end - phstart ):
                                #print(f"phlen/2 : {phlen*0.75} < {float( end - phstart )}")
                                #add the segment
                                #print("It starts inside the sample")
                                sample_window_list.append([segment[4], segidx, max( st, phstart ), min(end, phend)])
                    
                        #if the phone is contained inside the sample
                        elif phstart >= st and phend <= end :
                            #print("it is IN!")
                            sample_window_list.append([segment[4], segidx, max( st, phstart ), min(end, phend)])
                        
                        #if the whole segment is smaller than the phone segment 
                        elif phstart <= st and phend >= end:
                            #print("it contains the sample!")
                            sample_window_list.append([segment[4], segidx, max( st, phstart ), min(end, phend)])
                        
                        else:
                            chamo =0
                        
                        
                sample_label_list.append(sample_window_list)
                  
            
            #check each sample to see if the labels are empty
            emptywindows = []
            for winn, lol in enumerate(sample_label_list):
                if not lol:
                    emptywindows.append(winn)
               
            #if any segment has no labels , skip it   
            if emptywindows:
                #copy the lists
                sample_label_list_copy = copy.deepcopy(sample_label_list)
                sample_fr_list_copy = copy.deepcopy(sample_fr_list)
                sample_label_list = []
                sample_fr_list = []
                for winn, lol in enumerate(sample_label_list_copy):
                    if winn not in emptywindows:
                        sample_label_list.append(lol)
                        sample_fr_list.append(sample_fr_list_copy[winn])
            
                
            #collect the number of phones detected in a segment           
            for jo in sample_label_list:
                phcounter.append(len(jo))
                        
            subset_labels.extend( sample_label_list) #every item on _samples corresponding to the sample_fr_list
            subset_samples.extend(sample_fr_list)
            
        
    return subset_samples, subset_labels, np.array(phcounter)

# Main function
def main():

    ##########
    # argument
    ##########
    parser = argparse.ArgumentParser('Sampling')
    parser.add_argument('--win_samp_json',type=str,default='json/win_mel.json.x')
    parser.add_argument('--win_label_json',type=str,default='json/win_label.json.x')
    parser.add_argument('--tr_sam_json',type=str,default='json/sample_mfcc.json.train')
    parser.add_argument('--te_sam_json',type=str,default='json/sample_mfcc.json.test')
    parser.add_argument('--tr_data_phlen_json',type=str,default='json/data_length_mfcc.json.train')
    parser.add_argument('--te_data_phlen_json',type=str,default='json/data_length_mfcc.json.test') 
    parser.add_argument('--tr_data_len_json',type=str,default='json/data_length_mel80.json.train')
    parser.add_argument('--te_data_len_json',type=str,default='json/data_length_mel80.json.test')
    parser.add_argument('--mode',type=str,default='ph')
    parser.add_argument('--seg_len',type=float,default=1)
    parser.add_argument('--stride',type=float,default=0.5)
    parser.add_argument('--cutoff_frame_edge',type=int,default=0)
    args = parser.parse_args()
    print(args)
    
    if args.cutoff_frame_edge > 0:
        args.win_samp_json = f"{args.win_samp_json}.ctf_{args.cutoff_frame_edge}"
        args.win_label_json = f"{args.win_label_json}.ctf_{args.cutoff_frame_edge}"

    SEG_LEN = time2frame(args.seg_len)
    STRIDE = time2frame(args.stride,)
    
    print(f"seg len: {SEG_LEN}")
    print(f"stride: {STRIDE}")

    train_json_path = args.tr_data_phlen_json
    test_json_path = args.te_data_phlen_json

    if not os.path.exists(args.tr_sam_json):
        os.makedirs(args.tr_sam_json)
    if not os.path.exists(args.te_sam_json):
        os.makedirs(args.te_sam_json)


    with open(train_json_path,'r') as json_f_1:
        train_ph_seg = json.load(json_f_1)
    with open(test_json_path,'r') as json_f_2:
        test_ph_seg = json.load(json_f_2)
      
      #load the frame length of the audiofiles  
    with open(args.tr_data_len_json,'r') as json_f_1:
        train_audiolen = json.load(json_f_1)
    with open(args.te_data_len_json,'r') as json_f_1:
        test_audiolen = json.load(json_f_1)

    train_fr_list, train_labels, train_phcounter = gen_samples(train_ph_seg, train_audiolen, SEG_LEN, STRIDE, args.cutoff_frame_edge)
    test_fr_list, test_labels, test_phcounter = gen_samples(test_ph_seg, test_audiolen, SEG_LEN, STRIDE, args.cutoff_frame_edge)


    with open(args.win_samp_json+'.train','w') as sp_f:
        json.dump(train_fr_list,sp_f,indent=4)
    with open(args.win_label_json+'.train','w') as sp_f:
        json.dump(train_labels,sp_f,indent=4)

    with open(args.win_samp_json+'.test','w') as sp_f:
        json.dump(test_fr_list,sp_f,indent=4)
    with open(args.win_label_json+'.test','w') as sp_f:
        json.dump(test_labels,sp_f,indent=4)


    print(f"N feat train: {len(train_fr_list)}",flush=True)
    print(f"N label train: {len(train_labels)}",flush=True)
    print(f"N feat test: {len(test_fr_list)}",flush=True)
    print(f"N label test: {len(test_labels)}",flush=True)
    
    print(f"Train feat segments: {args.win_samp_json+'.train'}")
    print(f"Train labels: {args.win_label_json+'.train'}")
    print(f"AVG PH labels/segment: {train_phcounter.mean()} \t STD PH labels/segment: {train_phcounter.std()}")
    print(f"Test feat segments: {args.win_samp_json+'.test'}")
    print(f"Test labels: {args.win_label_json+'.test'}")
    print(f"AVG PH labels/segment: {test_phcounter.mean()} \t STD PH labels/segment: {test_phcounter.std()}")
    


if __name__ == '__main__':
    main()
