#!/bin/bash

#This method loads the train_stats dictionary and returns the model epoch with the lowest test loss shown during training

# Antonio Lopez 05/11/2020

DATE=`date '+%Y-%m-%d %H:%M:%S'`

##feedforward model
lr=1e-3
PYMODEL=FFWD1
ref_mode=a1   #{a1, a2, a3, maj, or, and}

dataset=test

#Possitive class weights
cw=32.0    #{float, 'auto' , 'false'} float (number) defines the weight for the positive clas, 'auto' computes the ratio from the real class counts and 'false'  just doesn't use it

#posterior filtered training set?
FILTERED=false  #used the filtered dataset
cutofflo=0.4
cutoffhi=0.6
cutofepoch=100

#INPUTS TO INCLUDE
GOP=true
VAE=true
LDA=true
PHSEG_IDX=false
PHONE=true
CNTXT=false
WORD=false
PREREF=false  #we might implement this later
NO_WEIGHTS=false
if [ $cw = "false" ]; then
NO_WEIGHTS=true
fi

## VAE parameters
VAE_HID=64   #The embedidng layer
VAE_LR='1e-3'   #Learning rate used for training the VAE
VAE_BETA='1e-4' #the beta factor used for training the VAE
VAE_EPOCH=100  #The number of epochs the VAE was trained

##LDA parameters
LDA_TOPICS=16
LDA_COMP=512
LDA_EPOCHS=20

GOP=${GOP,,}
VAE=${VAE,,}
LDA=${LDA,,}
PHSEG_IDX=${PHSEG_IDX,,}
PHONE=${PHONE,,}
CNTXT=${CNTXT,,}
WORD=${WORD,,}
PREREF=${PREREF,,}
FILTERED=${FILTERED,,}
dataset=${dataset,,}
NO_WEIGHTS=${NO_WEIGHTS,,}

DIC_NAME=$PYMODEL'_train_ina_mel80_lr'$lr'_ref'$ref_mode

if [ $PHONE = "true" ]; then
DIC_NAME=$DIC_NAME'_ph'
fi
if [ $PHSEG_IDX = "true" ]; then
DIC_NAME=$DIC_NAME'_phseg'
fi
if [ $WORD = "true" ]; then
DIC_NAME=$DIC_NAME'_wd'
fi
if [ $CNTXT = "true" ]; then
DIC_NAME=$DIC_NAME'_ctx'
fi
if [ $PREREF = "true" ]; then
DIC_NAME=$DIC_NAME'_preref'
fi
if [ $GOP = "true" ]; then
DIC_NAME=$DIC_NAME'_gop'
fi
if [ $VAE = "true" ]; then
DIC_NAME=$DIC_NAME'_VAEh'$VAE_HID'_VAEep'$VAE_EPOCH
fi
if [ $LDA = "true" ]; then
DIC_NAME=$DIC_NAME'_LDAt'$LDA_TOPICS'_LDAd'$LDA_COMP'_LDAep'$LDA_EPOCHS
fi
if [ $FILTERED = "true" ]; then
DIC_NAME=$DIC_NAME'_filt_ep'$cutofepoch'_lo'$cutofflo'_hi'$cutoffhi
fi
if [ $NO_WEIGHTS = "true" ]; then
DIC_NAME=$DIC_NAME'_nw'
else
DIC_NAME=$DIC_NAME'_n'$cw
fi

WD=/share/mini1/res/t/asr/call/childread-nl/its/aa/ad


TRAIN_STAT_DIR=$WD/json/train_stats
SELECT_SCRIPT=$WD/select_epoch_testloss.py


##this lines have to be added to bash scripts running python script. it activates the correct conda and python libraries
#activate python (this is just a random python 3 thas works fine for plotting
ENVIRONMENT=/l2arctic/lda/tools/conda_env/lid_lda
source /python/anaconda3/v3.7/etc/profile.d/conda.sh
conda activate

python $SELECT_SCRIPT -dic $TRAIN_STAT_DIR/$DIC_NAME -d $dataset



