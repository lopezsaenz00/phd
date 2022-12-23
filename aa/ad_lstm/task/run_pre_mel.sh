#!/bin/bash
#@author: Jose Antonio Lopez @ The University of Sheffield

#submits the Mel spectogram generator for the INA set



MODE=post  #{dic, dic2} dic2 uses a stricter acostic model for gop
MEL_FILTERS=80

SCP_TRAIN=/aa/lib/INA.v1.cut1-6.phone.train
SCP_TEST=/aa/lib/INA.v1.cut1-6.phone.test
#whe use the gop files as lab files since it is useful to retain the score in the h5py 
LAB_SCP_TRAIN=/aa/lib/INA.v1.cut1-6.phone.$MODE'gop.train'
LAB_SCP_TEST=/aa/lib/INA.v1.cut1-6.phone.$MODE'gop.test'

ENVIRONMENT=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/aa_py3
source /share/mini1/sw/std/python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

WD=/aa/ad_lstm
export  HDF5_USE_FILE_LOCKING=FALSE

python  $WD/task/preprocess_mel.py  --dataset_output $WD/h5/INA_mel80.h5 \
                        --data_length_output $WD/json/INA_mel80/data_length_mel80.json \
                        --ph_length_output $WD/json/INA_mel80/ph_length_mel80.json \
                        --ph_samp_output $WD/json/INA_mel80/ph_samp_mel80.json \
                        --speaker2index_output $WD/json/INA_mel80/INAspeaker2index.json \
                        --wav_scp_train  $SCP_TRAIN \
                        --wav_scp_test  $SCP_TEST \
                        --lab_scp_train  $LAB_SCP_TRAIN \
                        --lab_scp_test  $LAB_SCP_TEST \
						--num_filters $MEL_FILTERS > pre_INA_mel.log
#$PYTHON preprocessing/make_dataset_zerospeech2017.py  --mode sample
