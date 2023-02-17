#!/bin/bash
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib

MODE=dic  #{dic, dic2} dic2 uses a stricter acostic model for gop
MEL_FILTERS=80

SCP_TRAIN=/aa/lib/phone.train
SCP_TEST=/aa/lib/phone.test
#whe use the gop files as lab files since it is useful to retain the score in the h5py 
LAB_SCP_TRAIN=/aa/lib/phone.$MODE'gop.train'
LAB_SCP_TEST=/aa/lib/phone.$MODE'gop.test'

ENVIRONMENT=/tools/conda_env/lid_lda
source /python/anaconda3/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

WD=/share/mini1/res/t/asr/call/childread-nl/its/aa/vae_ph
export  HDF5_USE_FILE_LOCKING=FALSE

python  $WD/task/preprocess_mel.py  --dataset_output $WD/h5/wav_mel80.h5 \
                        --data_length_output $WD/json/wav_mel80/data_length_mel80.json \
                        --ph_length_output $WD/json/wav_mel80/ph_length_mel80.json \
                        --ph_samp_output $WD/json/wav_mel80/ph_samp_mel80.json \
                        --speaker2index_output $WD/json/wav_mel80/speaker2index.json \
                        --wav_scp_train  $SCP_TRAIN \
                        --wav_scp_test  $SCP_TEST \
                        --lab_scp_train  $LAB_SCP_TRAIN \
                        --lab_scp_test  $LAB_SCP_TEST \
						--num_filters $MEL_FILTERS > pre_mel.log

