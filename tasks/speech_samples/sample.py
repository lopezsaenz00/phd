#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sat October 24 18:56:00 2020
From Mingjie Chen

@author: Jose Antonio Lopez @ The University of Sheffield

"""

import json
import numpy as np
import argparse
import os

##########
# argument
##########
parser = argparse.ArgumentParser('Sampling')
parser.add_argument('--tr_sam_json',type=str,default='json/sample_mfcc.json.train')
parser.add_argument('--te_sam_json',type=str,default='json/sample_mfcc.json.test')
parser.add_argument('--tr_data_len_json',type=str,default='json/data_length_mfcc.json.train')
parser.add_argument('--te_data_len_json',type=str,default='json/data_length_mfcc.json.test')
parser.add_argument('--seg_len',type=int,default=30)
parser.add_argument('--stride',type=int,default=10)
args = parser.parse_args()
print(args)


SEGMENT_LEN = args.seg_len
STRIDE = args.stride

train_json_path = args.tr_data_len_json
test_json_path = args.te_data_len_json
train_sample_out_path = args.tr_sam_json+f'sample_mel80.json.train.seg{args.seg_len}.str{args.stride}'
test_sample_out_path = args.te_sam_json+f'sample_mel80.json.test.seg{args.seg_len}.str{args.stride}'

if not os.path.exists(args.tr_sam_json):
    os.makedirs(args.tr_sam_json)
if not os.path.exists(args.te_sam_json):
    os.makedirs(args.te_sam_json)


with open(train_json_path,'r') as json_f_1:
    train_data_length = json.load(json_f_1)
with open(test_json_path,'r') as json_f_2:
    test_data_length = json.load(json_f_2)


train_data_size = len(train_data_length)
print(f"train size {train_data_size}",flush=True)
train_samples = []
test_data_size = len(test_data_length)
print(f"test size {test_data_size}",flush=True)
test_samples = []
#for _ in range(SAMPLE_N):
#    ind = np.random.choice(data_size)
#    h5_path = data_length[ind][0]
#    length = data_length[ind][1]
#    start = np.random.choice(length-SEGMENT_LEN)
#    samples.append((h5_path,start))


for h5_path, sample_length in train_data_length:
    print(f"sample_length {sample_length}")
    for start in range(0,sample_length-SEGMENT_LEN,STRIDE):
        
        print(f"{h5_path}  {start}")
        train_samples.append((h5_path,start) )
for h5_path, sample_length in test_data_length:
    print(f"sample_length {sample_length}")
    for start in range(0,sample_length-SEGMENT_LEN,STRIDE):
        print(f"{h5_path}  {start}")
        test_samples.append((h5_path,start) )
    
with open(train_sample_out_path,'w') as sample_f_1:
    json.dump(train_samples,sample_f_1) 
with open(test_sample_out_path,'w') as sample_f_2:
    json.dump(test_samples,sample_f_2) 
print(f"finish train {len(train_samples)} test {len(test_samples)}",flush=True)


