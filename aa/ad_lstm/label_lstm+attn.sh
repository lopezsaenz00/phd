#!/bin/bash


SUBMITJOB='submitjob -q GPU -o -l hostname="node24|node23|node25|node26" -eo'
WAITJOB='-w 5890741'

export  HDF5_USE_FILE_LOCKING=FALSE

ENVIRONMENT=/lda/tools/conda_env/lid_lda
source /python/anaconda3/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

PYMODEL=LSTM+ATTN
window=0.5  #in seconds
stride=0.1   #in seconds

epoch=16

ref_mode=a1   #{a1, a2, a3, maj, or, and}
lr=1e-3  #learning rate
batch_size=256  #batch processing.
ph_mode=error   #ax  #a single phone label or 'all' for all the phones

#Possitive class weights
cw='false'   #{float, 'auto' , 'false'} float (number) defines the weight for the positive clas, 'auto' computes the ratio from the real class counts and 'false'  just doesn't use it

WD=/aa/ad_lstm
LOG=$WD/log

REF_FILE=/aa/task/ref/INA.v1.cut1-6.phone.$ref_mode.ref

#LSTM PARAMS
LSTM_HID=512
LSTM_NLAYERS=1
LSTM_OUTSIZE=1024
#DUAL ATTN NET
ATT_HID=128
#DAN_NLAYERS=1

#training loop specs
startepoc=0
endepoc=1
start=$startepoc
rune=1 #how many epochs per sh file
end=`expr $rune + $start`


NO_WEIGHTS=false
if [ $cw = "false" ]; then
NO_WEIGHTS=true
fi


WD=/aa/ad_lstm
LOG=$WD/log
LOG=$LOG/$PYMODEL'_lr'$lr'_ref'$ref_mode


f=$LOG/$PYMODEL'_lr'$lr'_ref'$ref_mode'_ph'$ph_mode'_bs'$batch_size'_LSTMh'$LSTM_HID'_LSTMl'$LSTM_NLAYERS'_LSTMo'$LSTM_OUTSIZE'_DANh'$ATT_HID  # '_sep'$sstart'_'eep$end.sh

if [ $NO_WEIGHTS = "true" ]; then
	f=$f'_nw'
else
	f=$f'_w'$cw
fi

f=$f'_ep'$epoch'_label.sh'
l=$f'_ep'$epoch'_label.log'

#fill out the .sh file
echo '#!/bin/bash' > $f
echo 'ENVIRONMENT=/lda/tools/conda_env/lid_lda' >> $f
echo 'source /share/mini1/sw/std/python/anaconda3/v3.7/etc/profile.d/conda.sh' >> $f
echo 'conda activate $ENVIRONMENT' >> $f
echo 'echo $CONDA_PREFIX' >> $f
echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
echo 'export OPENBLAS_NUM_THREADS=1' >> $f
echo 'export GOTO_NUM_THREADS=1' >> $f
echo 'export OMP_NUM_THREADS=1' >> $f
echo "WD=/aa/ad_lstm" >> $f
echo "PYMODEL=$PYMODEL" >> $f
echo "window=$window" >> $f
echo "stride=$stride" >> $f
echo "ref_mode=$ref_mode" >> $f
echo "lr=$lr" >> $f
echo "batch_size=$batch_size" >> $f
echo "ph_mode=$ph_mode" >> $f
if [ ! $NO_WEIGHTS = "true" ]; then
	echo "cw=$cw" >> $f
fi
echo "REF_FILE=$REF_FILE" >> $f
echo "LSTM_HID=$LSTM_HID" >> $f
echo "LSTM_NLAYERS=$LSTM_NLAYERS" >> $f
echo "LSTM_OUTSIZE=$LSTM_OUTSIZE" >> $f
echo "ATT_HID=$ATT_HID" >> $f
echo 'OPENBLAS_NUM_THREADS=1' >> $f
echo 'MAGICK_THREAD_LIMIT=1' >> $f

echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f

echo 'python $WD/train_lstm+attn.py --use_cuda   \' >> $f
echo '					   --pre_data_h5 $WD/h5/INA_wav_mel80.h5 \' >> $f
echo '					   --mvn_pa $WD/mvn/wav_mel80_train \' >> $f
echo '					   --win_mel_json $WD/json/INA_wav_mel80/win_mel.json.w_$window.str_$stride \' >> $f
echo '					   --win_label_json $WD/json/INA_wav_mel80/win_label.json.w_$window.str_$stride \' >> $f
echo '					   --save_model_dir $WD/ckpt/$PYMODEL'_lr'$lr'_ref'$ref_mode \' >> $f
echo '					   --model_name $PYMODEL'_lr'$lr'_ref'$ref_mode \' >> $f
echo '					   --output $WD/output/$PYMODEL'_lr'$lr'_ref'$ref_mode \' >> $f
echo '					   --ref_file $REF_FILE \' >> $f
echo '					   --ref_mode $ref_mode \' >> $f
echo '					   --ph_mode ${ph_mode,,} \' >> $f
echo '					   --mvn_pa $WD/mvn/wav_mel80_train \' >> $f
echo '					   --wind $window \' >> $f
echo '					   --str $stride \' >> $f
echo '					   --lstm_hid $LSTM_HID \' >> $f
echo '					   --lstm_layers $LSTM_NLAYERS \' >> $f
echo '					   --lstm_out $LSTM_OUTSIZE \' >> $f
echo '					   --att_hid $ATT_HID \' >> $f
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
	
	f=$LOG/$PYMODEL'_lr'$lr'_ref'$ref_mode'_ph'$ph_mode'_bs'$batch_size'_LSTMh'$LSTM_HID'_LSTMl'$LSTM_NLAYERS'_LSTMo'$LSTM_OUTSIZE'_DANh'$ATT_HID  # '_sep'$sstart'_'eep$end.sh

	if [ $NO_WEIGHTS = "true" ]; then
		f=$f'_nw'
	else
		f=$f'_w'$cw
	fi
	
	f=$f'_ep'$epoch'_label.sh'
	l=$f'_ep'$epoch'_label.log'
	
	#fill out the .sh file
	echo '#!/bin/bash' > $f
	echo 'ENVIRONMENT=/lda/tools/conda_env/lid_lda' >> $f
	echo 'source /python/anaconda3/v3.7/etc/profile.d/conda.sh' >> $f
	echo 'conda activate $ENVIRONMENT' >> $f
	echo 'echo $CONDA_PREFIX' >> $f
	echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
	echo 'export OPENBLAS_NUM_THREADS=1' >> $f
	echo 'export GOTO_NUM_THREADS=1' >> $f
	echo 'export OMP_NUM_THREADS=1' >> $f
	echo "WD=/share/mini1/res/t/asr/call/childread-nl/its/aa/ad_lstm" >> $f
	echo "PYMODEL=$PYMODEL" >> $f
	echo "window=$window" >> $f
	echo "stride=$stride" >> $f
	echo "ref_mode=$ref_mode" >> $f
	echo "lr=$lr" >> $f
	echo "batch_size=$batch_size" >> $f
	echo "ph_mode=$ph_mode" >> $f
	if [ ! $NO_WEIGHTS = "true" ]; then
		echo "cw=$cw" >> $f
	fi
	echo "REF_FILE=$REF_FILE" >> $f
	echo "LSTM_HID=$LSTM_HID" >> $f
	echo "LSTM_NLAYERS=$LSTM_NLAYERS" >> $f
	echo "LSTM_OUTSIZE=$LSTM_OUTSIZE" >> $f
	echo "ATT_HID=$ATT_HID" >> $f
	echo 'OPENBLAS_NUM_THREADS=1' >> $f
	echo 'MAGICK_THREAD_LIMIT=1' >> $f

	echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
	
	echo 'python $WD/train_lstm+attn.py --use_cuda   \' >> $f
	echo '					   --pre_data_h5 $WD/h5/INA_wav_mel80.h5 \' >> $f
	echo '					   --mvn_pa $WD/mvn/wav_mel80_train \' >> $f
	echo '					   --win_mel_json $WD/json/INA_wav_mel80/win_mel.json.w_$window.str_$stride \' >> $f
	echo '					   --win_label_json $WD/json/INA_wav_mel80/win_label.json.w_$window.str_$stride \' >> $f
	echo '					   --save_model_dir $WD/ckpt/$PYMODEL'_lr'$lr'_ref'$ref_mode \' >> $f
	echo '					   --model_name $PYMODEL'_lr'$lr'_ref'$ref_mode \' >> $f
	echo '					   --save_model \' >> $f
	echo '					   --ref_file $REF_FILE \' >> $f
	echo '					   --ref_mode $ref_mode \' >> $f
	echo '					   --ph_mode ${ph_mode,,} \' >> $f
	echo '					   --stats_dict $WD/json/train_stats/$PYMODEL'_train_lr'$lr'_ref'$ref_mode \' >> $f
	echo '					   --mvn_pa $WD/mvn/wav_mel80_train \' >> $f
	echo '					   --wind $window \' >> $f
	echo '					   --str $stride \' >> $f
	echo '					   --lstm_hid $LSTM_HID \' >> $f
	echo '					   --lstm_layers $LSTM_NLAYERS \' >> $f
	echo '					   --lstm_out $LSTM_OUTSIZE \' >> $f
	echo '					   --att_hid $ATT_HID \' >> $f
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
	
	#jid=$($SUBMITJOB $WAITJOB $l $f | grep -E [0-9]+)
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
