#!/bin/bash
#@author: Jose Antonio Lopez @ The University of Sheffield

#generate samples of window size with a stride for the INA set. it also alocates the phoneme labels given the original alignment


WD=/share/mini1/res/t/asr/call/childread-nl/its/aa/ad_lstm

export  HDF5_USE_FILE_LOCKING=FALSE

FEAT=plp  #{fbk, plp, mel80}

window=0.5  #in seconds
stride=0.05  #in seconds

cutoff_frame_edge=5  #the number of frames within the edge a phone label has to be located  to be recognised as present in the segment
						#if zero, 75% of the phone has to be present in the segment to be recognised as present

ENVIRONMENT=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/aa_py3
source /share/mini1/sw/std/python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

python  $WD/task/sample.py  \
                        --seg_len $window \
                        --stride $stride \
                        --tr_sam_json $WD/json/INA_$FEAT/\
                        --te_sam_json $WD/json/INA_$FEAT/\
                        --tr_data_phlen_json $WD/json/INA_$FEAT/ph_samp_$FEAT.json.train \
                        --te_data_phlen_json $WD/json/INA_$FEAT/ph_samp_$FEAT.json.test \
                        --tr_data_len_json $WD/json/INA_$FEAT/data_length_$FEAT.json.train \
                        --te_data_len_json $WD/json/INA_$FEAT/data_length_$FEAT.json.test \
                        --win_samp_json $WD/json/INA_$FEAT/win_samp.json.w_$window.str_$stride \
                        --win_label_json $WD/json/INA_$FEAT/win_label.json.w_$window.str_$stride \
						--cutoff_frame_edge $cutoff_frame_edge


