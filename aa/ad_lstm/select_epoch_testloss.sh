#!/bin/bash

#This method loads the train_stats dictionary and returns the model epoch with the lowest test loss shown during training
#for the lstm+attn models

# Antonio Lopez 05/11/2020

DATE=`date '+%Y-%m-%d %H:%M:%S'`

DATASET=test

FEAT=plp  #{fbk, plp, mel80}

ref_mode=a3   #{a1, a2, a3, maj, or, and}

PYMODEL=LSTM+ATTN+FWD # {LSTM+ATTN+FWD, LSTM+ATTN+CNN}

window=0.5  #in seconds
stride=0.05   #in seconds
cutoff_frame_edge=2  #the number of frames within the edge a phone label has to be located  to be recognised as present in the segment
						#if zero, 75% of the phone has to be present in the segment to be recognised as present

lr=1e-5  #learning rate
batch_size=128  #batch processing.
ph_mode=miss   #ax  #a single phone label or 'all' for all the phones, 'error' for detecting an error regardless of the phone class

#Possitive class weights
cw='false'   #{float, 'auto' , 'false'} float (number) defines the weight for the positive clas, 'auto' computes the ratio from the real class counts and 'false'  just doesn't use it

#LSTM PARAMS
LSTM_HID=64
LSTM_NLAYERS=6
#ATTENTION
ATT_HID=128
ATT_DIM=-1   #1 for softmax across time, 2 for softmax across lstm components, -1 for a 2dSoftmax
#CNN PARAMS
OUTPUT_CHANNELS=4-2 #USE COMMAS TO ADD MORE THAN ONE CNN LAYER, IT'S UNDERSTOOD AS A LIST
CNN_K_SIZE=3  #KERNEL SIZE
CNN_STR=1 #stride
#FULLY CONNECTED PART OF THE PREDICTOR
PRED_NLAYERS=6
PRED_HIDDEN=1024

NO_WEIGHTS=false
if [ $cw = "false" ]; then
NO_WEIGHTS=true
fi

#LSTM+ATTN+FWD_train_lr1e-5_refa1_plp_pherror_bs128_LSTMl1_LSTMh64_ATTh128_ATTd-1_PREDl4_PREDh1024_nw.win_0.5.str_0.05.ctf_2
DIC_NAME=$PYMODEL'_train_lr'$lr'_ref'$ref_mode'_'$FEAT'_ph'$ph_mode'_bs'$batch_size'_LSTMl'$LSTM_NLAYERS'_LSTMh'$LSTM_HID'_ATTh'$ATT_HID'_ATTd'$ATT_DIM

if [[ $PYMODEL == *"CNN"* ]]; then
DIC_NAME=$DIC_NAME'_CNNO'$OUTPUT_CHANNELS'_CNNK'$CNN_K_SIZE'_CNNSTR'$CNN_STR
fi

if [ "$PRED_NLAYERS" -gt "0" ]; then
DIC_NAME=$DIC_NAME'_PREDl'$PRED_NLAYERS'_PREDh'$PRED_HIDDEN
fi

if [ $NO_WEIGHTS = "true" ]; then
DIC_NAME=$DIC_NAME'_nw'
else
DIC_NAME=$DIC_NAME'_n'$cw
fi

DIC_NAME=$DIC_NAME'.win_'$window'.str_'$stride

if [ "$cutoff_frame_edge" -gt "0" ]; then
	DIC_NAME=$DIC_NAME'.ctf_'$cutoff_frame_edge
fi

WD=/aa/ad_lstm


TRAIN_STAT_DIR=$WD/json/train_stats
SELECT_SCRIPT=$WD/select_epoch_testloss.py


##this lines have to be added to bash scripts running python script. it activates the correct conda and python libraries
#activate python (this is just a random python 3 thas works fine for plotting
ENVIRONMENT=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/aa_py3
source /share/mini1/sw/std/python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT

python $SELECT_SCRIPT -dic $TRAIN_STAT_DIR/$DIC_NAME -d $DATASET



