#!/bin/bash

# this script models the triggers for the different phones
#this script is just for the post mode
#run first the label script
# Antonio Lopez 09/08/2020

DATE=`date '+%Y-%m-%d %H:%M:%S'`

MODE=post #this script is just for the post mode


##feedforward model
lr=1e-3
PYMODEL=FFWD1
ref_mode=a1   #{a1, a2, a3, maj, or, and}
epoch=16

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
NO_WEIGHTS=false
if [ $cw = "false" ]; then
NO_WEIGHTS=true
fi

WD=/aa/ad
DATA_JSON=/aa/vae_ph/json/INA_wav_mel80

#outputed labels
OUTPUT_FILE=$WD/output/$PYMODEL'_mel80_lr'$lr'_ref'$ref_mode


if [ $PHONE = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_ph'
fi
if [ $PHSEG_IDX = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_phseg'
fi
if [ $WORD = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_wd'
fi
if [ $CNTXT = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_ctx'
fi
if [ $PREREF = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_preref'
fi
if [ $GOP = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_gop'
fi
if [ $VAE = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_VAEh'$VAE_HID'_VAEep'$VAE_EPOCH
fi
if [ $LDA = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_LDAt'$LDA_TOPICS'_LDAd'$LDA_COMP'_LDAep'$LDA_EPOCHS
fi
if [ $FILTERED = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_filt_ep'$cutofepoch'_lo'$cutofflo'_hi'$cutoffhi
fi
if [ $NO_WEIGHTS = "true" ]; then
OUTPUT_FILE=$OUTPUT_FILE'_nw'
else
OUTPUT_FILE=$OUTPUT_FILE'_n'$cw
fi


OUTPUT_FILE=$OUTPUT_FILE'_ep'$epoch'.'  #just add train and test


TR_PH=$DATA_JSON/ph_length_mel80.json. #just add train and test
GRAPH=$WD/roc_curves/plots
REPORT_DIR=$WD/performance_reports

REF_FILE=/aa/task/ref/INA.v1.cut1-6.phone.$ref_mode.ref
ROC_SCRIPT=$WD/roc_curves/roc_curves.py


##this lines have to be added to bash scripts running python script. it activates the correct conda and python libraries
#activate python
ENVIRONMENT=/lda/tools/conda_env/lid_lda
source /python/anaconda3/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

if [ $FILTERED = "true" ]; then
python $ROC_SCRIPT -ph_json $TR_PH -ref_mode $ref_mode -label_f $OUTPUT_FILE -graph_dir $GRAPH -stats_dir $REPORT_DIR -filt
else
python $ROC_SCRIPT -ph_json $TR_PH -ref_mode $ref_mode -label_f $OUTPUT_FILE -graph_dir $GRAPH -stats_dir $REPORT_DIR
fi


