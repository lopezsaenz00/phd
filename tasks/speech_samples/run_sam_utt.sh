#!/bin/bash

#similar to run_sam.sh but instead of creating list with segment pointers it's just a list for whole utterances. This for testing the reconstruction of the
#vae

##this lines have to be added to bash scripts running python script. it activates the correct conda and python libraries
#activate python
ENVIRONMENT=/share/mini1/res/t/lid/accentl2/studio-en/l2arctic/lda/tools/conda_env/lid_lda
source /share/mini1/sw/std/python/anaconda3/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

WD=/share/mini1/res/t/asr/call/childread-nl/its/aa/vae_ph

export  HDF5_USE_FILE_LOCKING=FALSE

PYTHON=/share/mini1/sw/std/python/anaconda3-5.1.0/v5.1.0/bin/python
export  HDF5_USE_FILE_LOCKING=FALSE

python  $WD/task/sample_utt.py         --tr_sam_json $WD/json/INA_wav_mel80/\
                        --te_sam_json $WD/json/INA_wav_mel80/ \
                        --tr_data_len_json $WD/json/data_length_mel80.json.train \
                        --te_data_len_json $WD/json/data_length_mel80.json.test

#/home/acq18mc/.conda/envs/PYTORCH/bin/python  train.py
#/home/acq18mc/.conda/envs/PYTORCH/bin/python  sample_sp.py
#/home/acq18mc/.conda/envs/PYTORCH/bin/python  timit_phone.py

