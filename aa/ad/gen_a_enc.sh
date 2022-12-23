#!/bin/bash

#the scripts generates the embeddings for every assessor model. technically the same as the label

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib
source activate new-tf-gpu

#SUBMITJOB='submitjob -q GPU -o -l hostname="node24|node25|node20|node21|node22|node23" -eo'
SUBMITJOB='submitjob -q GPU'
#SUBMITJOB='submitjob -q GPU -o -l hostname=node25 -eo'  #node26 works fine. it's just busy
#WAITJOB='-w 5756052'

##feedforward model
lr=1e-3
PYMODEL=FFWD4
ref_mode=a3   #{a1, a2, a3, maj, or, and}

#INPUTS TO INCLUDE
GOP=true
VAE=true
LDA=true

epoch=80


PYTHON=/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/bin/python
H5DATA_DIR=/aa/vae_ph/h5
DATA_JSON=/aa/vae_ph/json/INA_wav_mel80

GOP=${GOP,,}
VAE=${VAE,,}
LDA=${LDA,,}

WD=/aa/ad
LOG=$WD/log
LOG=$LOG/$PYMODEL'_'ina_mel80_lr$lr

#make directory if doesn't exist
mkdir -p $LOG


f=$LOG/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode # '_sep'$sstart'_'eep$end.sh
l=$LOG/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode # '_sep'$sstart'_'eep$end.log

if [ $GOP = "true" ]; then
	f=$f'_gop'
	l=$l'_gop'
fi	
if [ $VAE = "true" ]; then
	f=$f'_vae'
	l=$l'_vae'
fi
if [ $LDA = "true" ]; then
	f=$f'_lda'
	l=$l'_lda'
fi

f=$f'_ep'$epoch'_enc.sh'
l=$l'_ep'$epoch'_enc.log'


echo '#!/bin/bash' > $f
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib' >> $f
echo 'source activate new-tf-gpu' >> $f
echo "PYTHON=$PYTHON" >> $f
echo "H5DATA_DIR=$H5DATA_DIR" >> $f
echo "DATA_JSON=$DATA_JSON" >> $f
echo "REF_MODE=$ref_mode" >> $f
echo "WD=$WD" >> $f
echo "lr=$lr" >> $f
echo "epoch=$epoch" >> $f
echo "PYMODEL=$PYMODEL" >> $f 

echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
echo '$PYTHON  $WD/ffwd_enc.py  --use_cuda   \' >> $f
echo '					   --pre_data_h5 $H5DATA_DIR/INA_wav_mel80.h5 \' >> $f
echo '					   --mvn_pa $WD/mvn/wav_mel80_train \' >> $f
echo '					   --tr_ph_json $DATA_JSON/ph_samp_mel80.json.train \' >> $f
echo '					   --te_ph_json $DATA_JSON/ph_samp_mel80.json.test \' >> $f
echo '					   --tr_ph_len_json $DATA_JSON/ph_length_mel80.json.train \' >> $f
echo '					   --te_ph_len_json $DATA_JSON/ph_length_mel80.json.test \' >> $f
echo '					   --ref_mode $REF_MODE \' >> $f
echo '					   --lr $lr \' >> $f
echo '					   --epoch $epoch \' >> $f
echo "					   --model $PYMODEL"' \' >> $f
echo "					   --model_name $PYMODEL"'_INA_MEL80_lr'$lr'_ref'$ref_mode' \' >> $f
echo "					   --model_dir $WD/ckpt/ffwd/$PYMODEL"'_mel80_lr'$lr'_ref'$ref_mode' \' >> $f
echo "					   --output $WD/enc/$PYMODEL"'_mel80_lr'$lr'_ref'$ref_mode' \' >> $f

if [ $GOP = "true" ]; then
	echo '					   --gop \' >> $f
fi
if [ $VAE = "true" ]; then
	echo '					   --vae \' >> $f
fi
if [ $LDA = "true" ]; then
	echo '					   --lda \' >> $f
fi


chmod +x $f


jid=$($SUBMITJOB $WAITJOB $l $f | grep -E [0-9]+)
echo $f
echo "Submitted as: ${jid}"


