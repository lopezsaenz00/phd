# encoding: utf-8
"""
Created on 26 Nov 2020

@author: Jose Antonio Lopez @ The University of Sheffield, 26 Nov 2020

This files sets main data configurations for itslfat.

"""

import os

SUBSET = 'train'

WAV_SCP='/aa/lib/INA.v1.cut1-6.phone.'+SUBSET.lower()

# ----------------------------------------------------------------------------
# GOP SCORE or equivalent. Say 'post' for example which is the absolute phone posterior 
# ----------------------------------------------------------------------------

#the different methods used for computing the GOP score. This have already been obtained. It doesnt do it on the fly.
#from /aa/gop
# ./itslfat : the original itslfat tool . Can't really remember if the GOP scoring works at 100%. This performs alignment from its dictionary rather than the
#               reference annotation
#./itslfat_dic : This computes the GOP as the original witt & young paper.  However, the phone sequence for the alignment comes from the reference annotation
#./itslfat_post : Instead of the basic GOP score this just uses the absolute phone logposterior.

SCORE_TYPE='post'

GOP_SCP='/aa/lib/INA.v1.cut1-6.phone.'+SCORE_TYPE.lower()+'gop.'+SUBSET.lower()

# ----------------------------------------------------------------------------
# VAE embeddings. The phone embeddings obtained via VAE
# ----------------------------------------------------------------------------

#The mel spectogram data used to train the VAE. Also, the dataset contains alignment information and GOP scores. 
#The dataset was assembled using:
#aa/vae_ph/task/run_pre_mel.sh
#The data is a h5py file structure.
#MEL_DATA=/aa/vae_ph/h5/INA_wav_mel80.h5



#the torch VAE model
VAE_IN_DIM=80
VAE_HID=64   #The embedidng layer
VAE_LR='1e-3'   #Learning rate used for training the VAE
VAE_BETA='1e-4' #the beta factor used for training the VAE
NORMALIZED=True  #mean and var normalization for the VAE
VAE_SEG_LEN=20  #The frame lenght of the segments feed to the VAE
VAE_EPOCH=100  #The number of epochs the VAE was trained

VAE_MVN='/aa/vae_ph/mvn/wav_mel80_train'  #path to the mean and v 

VAE_MODEL=f"/aa/vae_ph/ckpt/vae/VAE_mel80_hid{VAE_HID}_lr{VAE_LR}_beta{VAE_BETA}_Z/VAE_INA_MEL80_hid{VAE_HID}_lr{VAE_LR}_beta{VAE_BETA}_Z_{VAE_EPOCH}.pkl"
    

# ----------------------------------------------------------------------------
# LDA topic posteriors
# ----------------------------------------------------------------------------

#a compressed csv file that holds a pandas dataframe with the topic posteriors of every utterance

LDA_TOPICS=16
LDA_COMP=512
LDA_EPOCHS=20


TRAIN_LDAPOST=f"/aa/lda/train-lda/pz/INA.train_lda.t{LDA_TOPICS}.ep{LDA_EPOCHS}.gmm.ITSLang_{LDA_COMP}.train.csv.gz"
TEST_LDAPOST=f"/aa/lda/train-lda/pz/INA.test_lda.t{LDA_TOPICS}.ep{LDA_EPOCHS}.gmm.ITSLang_{LDA_COMP}.test.csv.gz"

# ----------------------------------------------------------------------------
# Annotation REF
# ----------------------------------------------------------------------------

#REF_MODE="a1" #{a1, a2, a3, maj, or, and}
REF_FILE="/aa/task/ref/INA.v1.cut1-6.phone.REF_MODE.ref"  #the word REF_MODE act as a token for selecting the ref mode via run_ffwd.sh
