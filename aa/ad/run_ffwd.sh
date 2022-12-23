#!/bin/bash

#trains a ffwd forthe assessors decision.

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib
source activate new-tf-gpu

SUBMITJOB='submitjob -q GPU -o -l hostname="node24|node23|node20|node26" -eo'  #node21 works fine. it's just busy
SUBMITJOB='submitjob -q GPU -o -l hostname="node24|node23|node20|node25|node26" -eo'
WAITJOB='-w 5741290'

FILTERED=false  #used the filtered dataset
cutofflo=0.4
cutoffhi=0.6
cutofepoch=100

##feedforward model
lr=1e-3
PYMODEL=FFWD1
ref_mode=a1   #{a1, a2, a3, maj, or, and}

#Possitive class weights
cw=32.0   #{float, 'auto' , 'false'} float (number) defines the weight for the positive clas, 'auto' computes the ratio from the real class counts and 'false'  just doesn't use it


#INPUTS TO INCLUDE
GOP=true
VAE=true
LDA=true
PHSEG_IDX=false
PHONE=true
CNTXT=false
WORD=false
PREREF=false  #we might implement this later


#training specs
startepoc=0
endepoc=50
start=$startepoc
rune=50 #how many epochs per sh file
end=`expr $rune + $start`

PYTHON=/conda_env/lid_lda/bin/python
H5DATA_DIR=/aa/vae_ph/h5
DATA_JSON=/aa/vae_ph/json/INA_wav_mel80
FILTERED_DATA_DIR=/aa/ad/filterdata

WD=/aa/ad
LOG=$WD/log
LOG=$LOG/$PYMODEL'_'ina_mel80_lr$lr

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

mkdir -p $LOG


for counter in $( seq $startepoc $endepoc )
do

   if [[ $end -ge `expr $endepoc + $rune` ]];
   then
      break
   fi

	sstart=$start
	if [[ "$start" == 1 ]]; then
		sstart=0
	fi
	
	
	f=$LOG/$PYMODEL'_train_ina_mel80_lr'$lr'_ref'$ref_mode # '_sep'$sstart'_'eep$end.sh
	l=$LOG/$PYMODEL'_train_ina_mel80_lr'$lr'_ref'$ref_mode # '_sep'$sstart'_'eep$end.log
	
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
	if [ $NO_WEIGHTS = "true" ]; then
		f=$f'_nw'
		l=$l'_nw'
	else
		f=$f'_w'$cw
		l=$l'_w'$cw
	fi
	
	
		
	f=$f'_sep'$sstart'_'eep$end.sh
	l=$l'_sep'$sstart'_'eep$end.log

	
	echo '#!/bin/bash' > $f
	echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib' >> $f
	echo 'source activate new-tf-gpu' >> $f
	echo "PYTHON=$PYTHON" >> $f
	echo "H5DATA_DIR=$H5DATA_DIR" >> $f
	echo "DATA_JSON=$DATA_JSON" >> $f
	echo "REF_MODE=$ref_mode" >> $f
	echo "WD=$WD" >> $f
	echo "lr=$lr" >> $f
	echo "PYMODEL=$PYMODEL" >> $f 
	if [ $FILTERED = "true" ]; then
		echo "ctlo=$cutofflo" >> $f 
		echo "cthi=$cutoffhi" >> $f 
		echo "ctep=$cutofepoch" >> $f 	
		echo "FILTDATA=$FILTERED_DATA_DIR/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode" >> $f 
	fi
	if [ ! $NO_WEIGHTS = "true" ]; then
		echo "cw=$cw" >> $f
	fi
	
	echo 'train_stats_dic=$WD/json/train_stats/$PYMODEL'_'train_ina_mel80_lr$lr"'"_ref"'"$ref_mode' >> $f
	
	echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
	echo '$PYTHON  $WD/ffwd_train.py  --use_cuda   \' >> $f
	echo '					   --pre_data_h5 $H5DATA_DIR/INA_wav_mel80.h5 \' >> $f
	echo '					   --mvn_pa $WD/mvn/wav_mel80_train \' >> $f
	echo '					   --tr_ph_json $DATA_JSON/ph_samp_mel80.json.train \' >> $f
	echo '					   --te_ph_json $DATA_JSON/ph_samp_mel80.json.test \' >> $f
	echo '					   --tr_ph_len_json $DATA_JSON/ph_length_mel80.json.train \' >> $f
	echo '					   --te_ph_len_json $DATA_JSON/ph_length_mel80.json.test \' >> $f
	echo "					   --log_dir $WD/runs/ffwd/$PYMODEL"'_mel80_lr'$lr'_ref'$ref_mode' \' >> $f
	echo '					   --ref_mode $REF_MODE \' >> $f
	echo '					   --lr $lr \' >> $f
	echo "					   --model $PYMODEL"' \' >> $f
	echo '					   --save_model \' >> $f
	echo "					   --model_name $PYMODEL"'_INA_MEL80_lr'$lr'_ref'$ref_mode' \' >> $f
	echo '					   --start_epo '$sstart' \' >> $f
	echo '					   --end_epo '$end' \' >> $f
	echo "					   --stats_dict $WD/json/train_stats/$PYMODEL"'_train_ina_mel80_lr'$lr'_ref'$ref_mode' \' >> $f
	echo "					   --save_model_dir $WD/ckpt/ffwd/$PYMODEL"'_mel80_lr'$lr'_ref'$ref_mode' \' >> $f
	if [ $GOP = "true" ]; then
		echo '					   --gop \' >> $f
	fi
	if [ $VAE = "true" ]; then
		echo '					   --vae \' >> $f
	fi
	if [ $LDA = "true" ]; then
		echo '					   --lda \' >> $f
	fi
	#
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
	if [ $NO_WEIGHTS = "true" ]; then
		echo '					   --nw \' >> $f
	else
		echo '					   --cw  $cw\' >> $f
	fi


	
	chmod +x $f
	
	jid=$($SUBMITJOB $WAITJOB $l $f | grep -E [0-9]+)
	WAITJOB="-w $jid"
	echo $counter. "\$start = " $sstart " and \$end = " $end
	echo $f
	echo "Submitted as: ${jid}"
   
   
   if [[ $start -eq 1 ]];
   then
      start=0
   fi
   
   start=$(($start+$rune))
   end=$(($end+$rune))
   
   
done
