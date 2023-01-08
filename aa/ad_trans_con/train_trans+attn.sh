#!/bin/bash


SUBMITJOB='submitjob -g1 -M4 -o -l gputype="GeForceGTXTITANX|GeForceGTX1080Ti|" -eo'
WAITJOB='-w 5890741'

#PYTHON=
export  HDF5_USE_FILE_LOCKING=FALSE


#ENVIRONMENT=
#source /python/anaconda3/v3.7/etc/profile.d/conda.sh
ENVIRONMENT=/python/anaconda3-2019.07/v3.7/envs/aa_py3
source /python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

FEAT=plp  #{fbk, plp, mel80}

PYMODEL=TRANS+ATTN+FWD # {LSTM+ATTN+FWD, LSTM+ATTN+CNN}
window=0.5  #in seconds
stride=0.05   #in seconds
cutoff_frame_edge=2  #the number of frames within the edge a phone label has to be located  to be recognised as present in the segment
						#if zero, 75% of the phone has to be present in the segment to be recognised as present

ref_mode=max   #{and0, and1, max}
lr=1e-5  #learning rate
batch_size=128  #batch processing.
ph_mode=error   #ax  #a single phone label or 'all' for all the phones, 'error' for detecting an error regardless of the phone class, 'miss' for detecting correct and incorrect pronunciations

#Possitive class weights
cw='false'   #{float, 'auto' , 'false'} float (number) defines the weight for the positive clas, 'auto' computes the ratio from the real class counts and 'false'  just doesn't use it

REF_FILE=

##TRANS PARAMS
##trnsformer Encoder layer
TR_HEADS=5     #NUMBER OF HEADS FOR THE MULTIATTENTION
TR_FFWD_DIM=128	#DIMENSION OF THE FEEDFORWARD MODEL (eq. attn_hid=128)
##trnsformer encoder
TR_LAYERS=1		#ENCODER LAYERS
TR_NORM=true # layer normalization component 

#FULLY CONNECTED PART OF THE PREDICTOR
PRED_NLAYERS=4
#PRED_HIDDEN=1024  #use '-' to submit a list of layer sizes: 256-64 for 2 hidden layers
PRED_HIDDEN=1024

#training loop specs
startepoc=0
endepoc=50
rune=10 #how many epochs per sh file
start=$startepoc
end=`expr $rune + $start`


NO_WEIGHTS=false
if [ $cw = "false" ]; then
NO_WEIGHTS=true
fi


WD=/aa/ad_trans_con
LOG=$WD/log
LOG=$LOG/$PYMODEL.$FEAT'_lr'$lr'_ref'$ref_mode


mkdir -p $LOG
#create the sh and log files
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
	
	f=$LOG/$PYMODEL'_train_'$FEAT'_ref'$ref_mode'_lr'$lr'_ph'$ph_mode'_bs'$batch_size'_TRH'$TR_HEADS'_TRh'$TR_FFWD_DIM'_TRl'$TR_LAYERS'_TRnorm'$TR_NORM
	
	if [ "$PRED_NLAYERS" -gt "0" ]; then
		f=$f'_PREDl'$PRED_NLAYERS'_PREDh'$PRED_HIDDEN
	fi
	
	f=$f'_win'$window'_str'$stride #	
	
	if [ "$cutoff_frame_edge" -gt "0" ]; then
		f=$f'_ctf'$cutoff_frame_edge
	fi
	

	if [ $NO_WEIGHTS = "true" ]; then
		f=$f'_nw'
	else
		f=$f'_w'$cw
	fi
		
	l=$f'_sep'$sstart'_'eep$end.log
	f=$f'_sep'$sstart'_'eep$end.sh
	
	#fill out the .sh file
	echo '#!/bin/bash' > $f
	
	echo 'ENVIRONMENT=' >> $f
	echo 'source /python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh' >> $f
	echo 'conda activate $ENVIRONMENT' >> $f
	echo 'nvidia-smi' >> $f
	echo 'env' >> $f
	#echo 'echo $CONDA_PREFIX' >> $f
	echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
	echo 'export OPENBLAS_NUM_THREADS=1' >> $f
	echo 'export GOTO_NUM_THREADS=1' >> $f
	echo 'export OMP_NUM_THREADS=1' >> $f
	echo "WD=$WD" >> $f
	echo "PYMODEL=$PYMODEL" >> $f
	echo "FEAT=$FEAT" >> $f
	echo "window=$window" >> $f
	echo "stride=$stride" >> $f
	echo "ref_mode=$ref_mode" >> $f
	echo "lr=$lr" >> $f
	echo "batch_size=$batch_size" >> $f
	echo "ph_mode=$ph_mode" >> $f
	echo "cutoff_frame_edge=$cutoff_frame_edge" >> $f
	if [ ! $NO_WEIGHTS = "true" ]; then
		echo "cw=$cw" >> $f
	fi
	
	echo "REF_FILE=$REF_FILE" >> $f
	echo "TR_HEADS=$TR_HEADS" >> $f
	echo "TR_FFWD_DIM=$TR_FFWD_DIM" >> $f
	echo "TR_LAYERS=$TR_LAYERS" >> $f
	echo "TR_NORM=$TR_NORM" >> $f
	
	if [ "$PRED_NLAYERS" -gt "0" ]; then
		echo "PRED_NLAYERS=$PRED_NLAYERS" >> $f
		echo "PRED_HIDDEN=$PRED_HIDDEN" >> $f
	fi

	echo 'OPENBLAS_NUM_THREADS=1' >> $f
	echo 'MAGICK_THREAD_LIMIT=1' >> $f

	echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
	
	echo 'python $WD/train_trans+attn.py --use_cuda   \' >> $f
	echo '					   --feat $FEAT \' >> $f
	echo '					   --pre_data_h5 $WD/h5/INA_$FEAT.h5 \' >> $f
	echo "					   --mvn_pa $WD/mvn/INA_$FEAT"'_train \' >> $f
	echo '					   --win_samp_json $WD/json/INA_$FEAT/win_samp.json.w_$window.str_$stride \' >> $f
	echo '					   --win_label_json $WD/json/INA_$FEAT/win_label.json.w_$window.str_$stride \' >> $f
	echo '					   --save_model_dir $WD/ckpt/$PYMODEL'"'"_lr"'"$lr"'"_"'"$FEAT"'"_ref"'"$ref_mode '\' >> $f
	echo '					   --model_name $PYMODEL'"'"_lr"'"$lr '\' >> $f
	echo '					   --save_model \' >> $f
	echo '					   --ref_file $REF_FILE \' >> $f
	echo '					   --ref_mode $ref_mode \' >> $f
	echo '					   --ph_mode ${ph_mode,,} \' >> $f
	echo '					   --stats_dict $WD/json/train_stats/$PYMODEL'"'"_train_lr"'"$lr '\' >> $f
	echo '					   --wind $window \' >> $f
	echo '					   --str $stride \' >> $f
	echo '					   --ctf $cutoff_frame_edge \' >> $f
	echo '					   --tr_heads $TR_HEADS \' >> $f
	echo '					   --tr_fwd_dim $TR_FFWD_DIM \' >> $f
	echo '					   --tr_layers $TR_LAYERS \' >> $f
	echo '					   --tr_norm $TR_NORM \' >> $f
	
	if [ "$PRED_NLAYERS" -gt "0" ]; then
		echo '					   --pred_layers $PRED_NLAYERS \' >> $f
		echo '					   --pred_hid $PRED_HIDDEN \' >> $f
	fi
	
	echo '					   --lr $lr \' >> $f
	echo '					   --batch_size $batch_size \' >> $f
	echo '					   --start_epo '$sstart' \' >> $f
	echo '					   --end_epo '$end' \' >> $f
	if [ $NO_WEIGHTS = "true" ]; then
		echo '					   --nw \' >> $f
	else
		echo '					   --cw  $cw\' >> $f
	fi
	
	chmod +x $f
	
	rm -f $l
	
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
