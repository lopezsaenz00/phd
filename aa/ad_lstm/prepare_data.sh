#!/bin/bash
"""
Created on February 2021

@author: Jose Antonio Lopez @ The University of Sheffield

This script submits the creator of the dataset objects to train the assessor model

"""

SUBMITJOB='submitjob -m 5000'
WAITJOB='-w 6267593'

export  HDF5_USE_FILE_LOCKING=FALSE

ENVIRONMENT=/python/anaconda3-2019.07/v3.7/envs/aa_py3
source /python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh
conda activate $ENVIRONMENT
echo $CONDA_PREFIX

FEAT=plp  #{fbk, plp, mel80}

ref_mode=a1   #{a1, a2, a3, maj, or, and}
subset=test

window=0.5  #in seconds
stride=0.05   #in seconds
cutoff_frame_edge=2  #the number of frames within the edge a phone label has to be located  to be recognised as present in the segment
						#if zero, 75% of the phone has to be present in the segment to be recognised as present

ph_mode=miss   #ax  #a single phone label or 'all' for all the correclty pronounced phones, 'error' for detecting an error regardless of the phone class, 'miss' for correct and incorrect pronunciation


WD=/aa/ad_lstm


LOG=$WD/log
LOG=$LOG/prep_data.$FEAT.$ref_mode
REF_FILE=/aa/task/ref/INA.v1.cut1-6.phone.$ref_mode.ref

mkdir -p $LOG

f=$LOG/prep_data.$ref_mode'_'$FEAT'_'$subset'_ph'$ph_mode'_win'$window'_str'$stride

if [ "$cutoff_frame_edge" -gt "0" ]; then
	f=$f'_ctf'$cutoff_frame_edge
fi

l=$f.log
f=$f.sh
	
#fill out the .sh file
echo '#!/bin/bash' > $f

echo 'ENVIRONMENT=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/aa_py3' >> $f
echo 'source /share/mini1/sw/std/python/anaconda3-2019.07/v3.7/etc/profile.d/conda.sh' >> $f
echo 'conda activate $ENVIRONMENT' >> $f
echo 'nvidia-smi' >> $f
echo 'env' >> $f
#echo 'echo $CONDA_PREFIX' >> $f
echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
echo 'export OPENBLAS_NUM_THREADS=1' >> $f
echo 'export GOTO_NUM_THREADS=1' >> $f
echo 'export OMP_NUM_THREADS=1' >> $f
echo "ref_mode=$ref_mode" >> $f
echo "subset=$subset" >> $f
echo "FEAT=$FEAT" >> $f
echo "REF_FILE=$REF_FILE" >> $f
echo "WD=$WD" >> $f
echo "window=$window" >> $f
echo "stride=$stride" >> $f
echo "ph_mode=$ph_mode" >> $f
echo "cutoff_frame_edge=$cutoff_frame_edge" >> $f

echo 'OPENBLAS_NUM_THREADS=1' >> $f
echo 'MAGICK_THREAD_LIMIT=1' >> $f

echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f

echo 'python $WD/prepare_data.py --use_cuda   \' >> $f
echo '					   --feat $FEAT \' >> $f
echo '					   --pre_data_h5 $WD/h5/INA_$FEAT.h5 \' >> $f
echo '					   --mvn_pa $WD/mvn/INA_'$FEAT'_train \' >> $f
echo '					   --win_mel_json $WD/json/INA_'$FEAT'/win_samp.json.w_$window.str_$stride \' >> $f
echo '					   --win_label_json $WD/json/INA_'$FEAT'/win_label.json.w_$window.str_$stride \' >> $f
echo '					   --ref_mode $ref_mode \' >> $f
echo '					   --ref_file $REF_FILE \' >> $f
echo '					   --ph_mode ${ph_mode,,} \' >> $f
echo '					   --wind $window \' >> $f
echo '					   --str $stride \' >> $f
echo '					   --subset $subset \' >> $f
echo '					   --ctf $cutoff_frame_edge \' >> $f


chmod +x $f

rm -f $l

jid=$($SUBMITJOB $WAITJOB $l $f | grep -E [0-9]+)
WAITJOB="-w $jid"
echo $f
echo "Submitted as: ${jid}"
   
