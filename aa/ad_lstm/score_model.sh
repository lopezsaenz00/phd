#!/bin/bash
#Created on Fri December 04 15:46:00 2020

#@author: Jose Antonio Lopez @ The University of Sheffield

#this script submits the generator of posterior probabilities to score the performance of the models


SUBMITJOB='submitjob -g1 -M4 -o -l gputype="GeForceGTXTITANX|GeForceGTX1080Ti|" -eo'
WAITJOB='-w 5952044'

export  HDF5_USE_FILE_LOCKING=FALSE

ENVIRONMENT=/python/anaconda3-2019.07/v3.7/envs/aa_py3
source /python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX


FEAT=plp  #{fbk, plp, mel80}

PYMODEL=LSTM+ATTN+FWD # {LSTM+ATTN+FWD, LSTM+ATTN+CNN}
MODEL_EPOCH=33  #which training epoch to use
window=0.5  #in seconds
stride=0.05   #in seconds
cutoff_frame_edge=2  #the number of frames within the edge a phone label has to be located  to be recognised as present in the segment
						#if zero, 75% of the phone has to be present in the segment to be recognised as present

ref_mode=a3   #{a1, a2, a3, maj, or, and}
lr=1e-5  #learning rate
batch_size=128  #batch processing.
ph_mode=miss   #ax  #a single phone label or 'all' for all the phones, 'error' for detecting an error regardless of the phone class
class_threshold='auto'   #a floating number between 0 and 1 or 'auto'

#Possitive class weights
cw='false'   #{float, 'auto' , 'false'} float (number) defines the weight for the positive clas, 'auto' computes the ratio from the real class counts and 'false'  just doesn't use it

REF_FILE=/aa/task/ref/INA.v1.cut1-6.phone.$ref_mode.ref

#LSTM PARAMS
LSTM_HID=64
LSTM_NLAYERS=6
#ATTENTION HID
ATT_HID=128
ATT_DIM=-1   #1 for softmax across time, 2 for softmax across lstm components, -1 for a 2dSoftmax
#CNN PARAMS
OUTPUT_CHANNELS=4-2 #USE COMMAS TO ADD MORE THAN ONE CNN LAYER, IT'S UNDERSTOOD AS A LIST
CNN_K_SIZE=3  #KERNEL SIZE
CNN_STR=1 #stride
#FULLY CONNECTED PART OF THE PREDICTOR
PRED_NLAYERS=6
PRED_HIDDEN=1024

NO_WEIGHTS=false
if [ $cw = "false" ]; then
NO_WEIGHTS=true
fi

WD=/aa/ad_lstm
CLASS_REP_DIR=$WD/performance_reports/class_rep
MISP_DETECTION_DIR=$WD/performance_reports/misp_detection

LOG=$WD/log
LOG=$LOG/$PYMODEL.$FEAT'_lr'$lr'_ref'$ref_mode

#name the sh file to submit
f=$LOG/$PYMODEL'_train_'$FEAT'_ref'$ref_mode'_lr'$lr'_ph'$ph_mode'_bs'$batch_size'_LSTMl'$LSTM_NLAYERS'_LSTMh'$LSTM_HID'_ATTh'$ATT_HID'_ATTd'$ATT_DIM

if [[ $PYMODEL == *"CNN"* ]]; then
	f=$f'_CNNO'$OUTPUT_CHANNELS'_CNNK'$CNN_K_SIZE'_CNNSTR'$CNN_STR	
fi

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
	
l=$f'_ep'$MODEL_EPOCH'_score.log'
f=$f'_ep'$MODEL_EPOCH'_score.sh'


#fill out the .sh file
echo '#!/bin/bash' > $f

echo 'ENVIRONMENT=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/aa_py3' >> $f
echo 'source /share/mini1/sw/std/python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh' >> $f
echo 'conda activate $ENVIRONMENT' >> $f
echo 'nvidia-smi' >> $f
echo 'env' >> $f
echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
echo 'export OPENBLAS_NUM_THREADS=1' >> $f
echo 'export GOTO_NUM_THREADS=1' >> $f
echo 'export OMP_NUM_THREADS=1' >> $f
echo "WD=$WD" >> $f
echo "CLASS_REP_DIR=$CLASS_REP_DIR" >> $f
echo "MISP_DETECTION_DIR=$MISP_DETECTION_DIR" >> $f
echo "PYMODEL=$PYMODEL" >> $f
echo "FEAT=$FEAT" >> $f
echo "window=$window" >> $f
echo "stride=$stride" >> $f
echo "ref_mode=$ref_mode" >> $f
echo "lr=$lr" >> $f
echo "batch_size=$batch_size" >> $f
echo "class_threshold=$class_threshold" >> $f
echo "ph_mode=$ph_mode" >> $f
echo "cutoff_frame_edge=$cutoff_frame_edge" >> $f
if [ ! $NO_WEIGHTS = "true" ]; then
	echo "cw=$cw" >> $f
fi
echo "REF_FILE=$REF_FILE" >> $f
echo "LSTM_HID=$LSTM_HID" >> $f
echo "LSTM_NLAYERS=$LSTM_NLAYERS" >> $f
echo "ATT_HID=$ATT_HID" >> $f
echo "ATT_DIM=$ATT_DIM" >> $f

if [[ $PYMODEL == *"CNN"* ]]; then
	echo "OUTPUT_CHANNELS=$OUTPUT_CHANNELS" >> $f
	echo "CNN_K_SIZE=$CNN_K_SIZE" >> $f
	echo "CNN_STR=$CNN_STR" >> $f

fi

if [ "$PRED_NLAYERS" -gt "0" ]; then
	echo "PRED_NLAYERS=$PRED_NLAYERS" >> $f
	echo "PRED_HIDDEN=$PRED_HIDDEN" >> $f
fi

echo 'OPENBLAS_NUM_THREADS=1' >> $f
echo 'MAGICK_THREAD_LIMIT=1' >> $f

echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f

echo 'python $WD/score_model.py --use_cuda   \' >> $f
echo '					   --feat $FEAT \' >> $f
echo '					   --pre_data_h5 $WD/h5/INA_$FEAT.h5 \' >> $f
echo "					   --mvn_pa $WD/mvn/INA_$FEAT"'_train \' >> $f
echo '					   --win_samp_json $WD/json/INA_$FEAT/win_samp.json.w_$window.str_$stride \' >> $f
echo '					   --win_label_json $WD/json/INA_$FEAT/win_label.json.w_$window.str_$stride \' >> $f
echo '					   --save_model_dir $WD/ckpt/$PYMODEL'"'"_lr"'"$lr"'"_"'"$FEAT"'"_ref"'"$ref_mode '\' >> $f
echo '					   --output $WD/post/$PYMODEL'"'"_lr"'"$lr"'"_"'"$FEAT"'"_ref"'"$ref_mode '\' >> $f
echo '					   --clas_rep_dir $CLASS_REP_DIR/$PYMODEL'"'"_lr"'"$lr"'"_"'"$FEAT"'"_ref"'"$ref_mode '\' >> $f
echo '					   --misp_dir $MISP_DETECTION_DIR/$PYMODEL'"'"_lr"'"$lr"'"_"'"$FEAT"'"_ref"'"$ref_mode '\' >> $f
echo '					   --model_name $PYMODEL'"'"_lr"'"$lr '\' >> $f
echo '					   --roc_dir $WD/roc_curves \' >> $f
echo '					   --save_model \' >> $f
echo '					   --att_soft_dim $ATT_DIM \' >> $f
echo '					   --ref_file $REF_FILE \' >> $f
echo '					   --ref_mode $ref_mode \' >> $f
echo '					   --ph_mode ${ph_mode,,} \' >> $f
echo '					   --cl_tr $class_threshold \' >> $f
echo '					   --wind $window \' >> $f
echo '					   --str $stride \' >> $f
echo '					   --ctf $cutoff_frame_edge \' >> $f
echo '					   --lstm_hid $LSTM_HID \' >> $f
echo '					   --lstm_layers $LSTM_NLAYERS \' >> $f
echo '					   --att_hid $ATT_HID \' >> $f


if [[ $PYMODEL == *"CNN"* ]]; then
	echo '					   --o_channels $OUTPUT_CHANNELS \' >> $f
	echo '					   --cnv_k_size $CNN_K_SIZE \' >> $f
	echo '					   --cnv_stride $CNN_STR \' >> $f
fi	

if [ "$PRED_NLAYERS" -gt "0" ]; then
	echo '					   --pred_layers $PRED_NLAYERS \' >> $f
	echo '					   --pred_hid $PRED_HIDDEN \' >> $f
fi

echo '					   --lr $lr \' >> $f
echo '					   --batch_size $batch_size \' >> $f
echo '					   --model_epoch '$MODEL_EPOCH' \' >> $f

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
