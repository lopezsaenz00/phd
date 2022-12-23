#!/bin/bash

#the scripts loads the dataset and labels it using the specified ffwd model.
#it generates an output text file with the phone segment and the decision of the model. 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib
source activate new-tf-gpu

#SUBMITJOB='submitjob -q GPU -o -l hostname="node24|node25|node20|node21|node22|node23|node26" -eo'
SUBMITJOB='submitjob -q GPU -o -l hostname="node24|node23|node22|node20|node25|node26" -eo'
#SUBMITJOB='submitjob -q GPU -o -l hostname=node25 -eo'  #node26 works fine. it's just busy
#WAITJOB='-w 5756052'

##feedforward model
lr=1e-3
PYMODEL=FFWD1
ref_mode=a3   #{a1, a2, a3, maj, or, and}

#posterior filtered training set?
FILTERED=true  #used the filtered dataset
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

#if you need the posterior from a single model epoch, make epst = epend.
#not the best way to do it, but this originally was made for a range of model epochs.
epst=100
epend=100
estep=5  #the step for obtaining the posterior


PYTHON=/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/bin/python
H5DATA_DIR=/aa/vae_ph/h5
DATA_JSON=/aa/vae_ph/json/INA_wav_mel80
FILTERED_DATA_DIR=/aa/ad/filterdata

GOP=${GOP,,}
VAE=${VAE,,}
LDA=${LDA,,}
PHSEG_IDX=${PHSEG_IDX,,}
PHONE=${PHONE,,}
CNTXT=${CNTXT,,}
WORD=${WORD,,}
PREREF=${PREREF,,}
FILTERED=${FILTERED,,}

WD=/share/mini1/res/t/asr/call/childread-nl/its/aa/ad
LOG=$WD/log
LOG=$LOG/$PYMODEL'_'ina_mel80_lr$lr

#make directory if doesn't exist
mkdir -p $LOG


f=$LOG/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode # '_sep'$sstart'_'eep$end.sh
l=$LOG/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode # '_sep'$sstart'_'eep$end.log

if [ $PHONE = "true" ]; then
	f=$f'_ph'
	l=$l'_ph'
fi
if [ $PHSEG_IDX = "true" ]; then
	f=$f'_phseg'
	l=$l'_phseg'
fi
if [ $WORD = "true" ]; then
	f=$f'_wd'
	l=$l'_wd'
fi
if [ $CNTXT = "true" ]; then
	f=$f'_ctx'
	l=$l'_ctx'
fi
if [ $PREREF = "true" ]; then
	f=$f'_preref'
	l=$l'_preref'
fi
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
if [ $FILTERED = "true" ]; then
	f=$f'_filtep'$cutofepoch'_lo'$cutofflo'_hi'$cutoffhi
	l=$l'_filtep'$cutofepoch'_lo'$cutofflo'_hi'$cutoffhi
fi

f=$f'_epst'$epst'_epend'$epend'_step'$estep'_post.sh'
l=$l'_epst'$epst'_epend'$epend'_step'$estep'_post.log'


echo '#!/bin/bash' > $f
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/mini1/sw/std/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib' >> $f
echo 'source activate new-tf-gpu' >> $f
echo "PYTHON=$PYTHON" >> $f
echo "H5DATA_DIR=$H5DATA_DIR" >> $f
echo "DATA_JSON=$DATA_JSON" >> $f
echo "REF_MODE=$ref_mode" >> $f
echo "WD=$WD" >> $f
echo "lr=$lr" >> $f
echo "epst=$epst" >> $f
echo "epend=$epend" >> $f
echo "estep=$estep" >> $f
echo "PYMODEL=$PYMODEL" >> $f 
if [ $FILTERED = "true" ]; then
	echo "ctlo=$cutofflo" >> $f 
	echo "cthi=$cutoffhi" >> $f 
	echo "ctep=$cutofepoch" >> $f 	
	echo "FILTDATA=$FILTERED_DATA_DIR/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode" >> $f 
fi

echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
echo '$PYTHON  $WD/ffwd_post.py  --use_cuda   \' >> $f
echo '					   --pre_data_h5 $H5DATA_DIR/INA_wav_mel80.h5 \' >> $f
echo '					   --mvn_pa $WD/mvn/wav_mel80_train \' >> $f
echo '					   --tr_ph_json $DATA_JSON/ph_samp_mel80.json.train \' >> $f
echo '					   --te_ph_json $DATA_JSON/ph_samp_mel80.json.test \' >> $f
echo '					   --tr_ph_len_json $DATA_JSON/ph_length_mel80.json.train \' >> $f
echo '					   --te_ph_len_json $DATA_JSON/ph_length_mel80.json.test \' >> $f
echo '					   --ref_mode $REF_MODE \' >> $f
echo '					   --lr $lr \' >> $f
echo '					   --epst $epst \' >> $f
echo '					   --epend $epend \' >> $f
echo '					   --step $estep \' >> $f
echo "					   --model $PYMODEL"' \' >> $f
echo "					   --model_name $PYMODEL"'_INA_MEL80_lr'$lr'_ref'$ref_mode' \' >> $f
echo "					   --model_dir $WD/ckpt/ffwd/$PYMODEL"'_mel80_lr'$lr'_ref'$ref_mode' \' >> $f
echo "					   --output $WD/post/$PYMODEL"'_mel80_lr'$lr'_ref'$ref_mode' \' >> $f
if [ $GOP = "true" ]; then
	echo '					   --gop \' >> $f
fi
if [ $VAE = "true" ]; then
	echo '					   --vae \' >> $f
fi
if [ $LDA = "true" ]; then
	echo '					   --lda \' >> $f
fi
if [ $PHSEG_IDX = "true" ]; then
	echo '					   --phseg \' >> $f
fi
if [ $PHONE = "true" ]; then
	echo '					   --phone \' >> $f
fi
if [ $CNTXT = "true" ]; then
	echo '					   --cntxt \' >> $f
fi
if [ $WORD = "true" ]; then
	echo '					   --word \' >> $f
fi
if [ $PREREF = "true" ]; then
	echo '					   --preref \' >> $f
fi
if [ $FILTERED = "true" ]; then
	echo '					   --filt \' >> $f
	echo '					   --filtdata $FILTDATA \' >> $f
	echo '					   --ctep $ctep \' >> $f		
	echo '					   --ctlo $ctlo \' >> $f
	echo '					   --cthi $cthi \' >> $f
fi

chmod +x $f


jid=$($SUBMITJOB $WAITJOB $l $f | grep -E [0-9]+)
echo $f
echo "Submitted as: ${jid}"
