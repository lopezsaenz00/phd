#!/bin/bash
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/mini1/sw/std/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib

MODE=dic  #{dic, dic2} dic2 uses a stricter acostic model for gop

SCP_TRAIN=/share/mini1/res/t/asr/call/childread-nl/its/aa/lib/INA.v1.cut1-6.phone.train
SCP_TEST=/share/mini1/res/t/asr/call/childread-nl/its/aa/lib/INA.v1.cut1-6.phone.test
LAB_SCP_TRAIN=/share/mini1/res/t/asr/call/childread-nl/its/aa/lib/INA.v1.cut1-6.phone.$MODE'post.train'
LAB_SCP_TEST=/share/mini1/res/t/asr/call/childread-nl/its/aa/lib/INA.v1.cut1-6.phone.$MODE'post.test'

ENVIRONMENT=/share/mini1/res/t/lid/accentl2/studio-en/l2arctic/lda/tools/conda_env/lid_lda
source /share/mini1/sw/std/python/anaconda3/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

WD=/share/mini1/res/t/asr/call/childread-nl/its/aa/vae_ph
export  HDF5_USE_FILE_LOCKING=FALSE

python  $WD/task/preprocess_mfcc.py  --dataset_output $WD/h5/INA_wav_mfcc39.h5 \
                        --data_length_output $WD/json/data_length_mfcc39.json \
                        --speaker2index_output $WD/json/INAspeaker2index.json \
                        --wav_scp_train  $SCP_TRAIN \
                        --wav_scp_test  $SCP_TEST \
                        --lab_scp_train  $LAB_SCP_TRAIN \
                        --lab_scp_test  $LAB_SCP_TEST > pre_INA_mfcc.log
#$PYTHON preprocessing/make_dataset_zerospeech2017.py  --mode sample
