#!/bin/bash

#the scripts generates histograms of the posterior probailities

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/mini1/sw/std/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib
source activate new-tf-gpu


SUBMITJOB='submitjob -q NORMAL -m 6000'
WAITJOB='-w 5834547'

##feedforward model
lr=1e-3
PYMODEL=FFWD1
ref_mode=a1   #{a1, a2, a3, maj, or, and}

#posterior filtered training set?
FILTERED=false  #used the filtered dataset
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
FILTERED=${FILTERED,,}

epst=100
epend=100
estep=5  #the step for obtaining the posterior


PYTHON=/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/bin/python
H5DATA_DIR=/aa/vae_ph/h5
DATA_JSON=/aa/vae_ph/json/INA_wav_mel80

GOP=${GOP,,}
VAE=${VAE,,}
LDA=${LDA,,}
PHSEG_IDX=${PHSEG_IDX,,}
PHONE=${PHONE,,}
CNTXT=${CNTXT,,}
WORD=${WORD,,}
PREREF=${PREREF,,}

WD=/aa/ad
LOG=$WD/log
LOG=$LOG/$PYMODEL'_'ina_mel80_lr$lr

#make directory if doesn't exist
mkdir -p $LOG


f=$LOG/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode # '_sep'$sstart'_'eep$end.sh
l=$LOG/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode # '_sep'$sstart'_'eep$end.log
hitsidr=$WD/hist/$PYMODEL'_ina_mel80_lr'$lr'_ref'$ref_mode

if [ $PHONE = "true" ]; then
	f=$f'_ph'
	l=$l'_ph'
	hitsidr=$hitsidr'_ph'	
fi
if [ $PHSEG_IDX = "true" ]; then
	f=$f'_phseg'
	l=$l'_phseg'
	hitsidr=$hitsidr'_phseg'	
fi
if [ $WORD = "true" ]; then
	f=$f'_wd'
	l=$l'_wd'
	hitsidr=$hitsidr'_wd'	
fi
if [ $CNTXT = "true" ]; then
	f=$f'_ctx'
	l=$l'_ctx'
	hitsidr=$hitsidr'_ctx'	
fi
if [ $PREREF = "true" ]; then
	f=$f'_preref'
	l=$l'_preref'
	hitsidr=$hitsidr'_preref'	
fi
if [ $GOP = "true" ]; then
	f=$f'_gop'
	l=$l'_gop'
	hitsidr=$hitsidr'_gop'
fi	
if [ $VAE = "true" ]; then
	f=$f'_vae'
	l=$l'_vae'
	hitsidr=$hitsidr'_vae'	
fi
if [ $LDA = "true" ]; then
	f=$f'_lda'
	l=$l'_lda'
	hitsidr=$hitsidr'_lda'	
fi

f=$f'_epst'$epst'_epend'$epend'_step'$estep'_plotpost.sh'
l=$l'_epst'$epst'_epend'$epend'_step'$estep'_plotpost.log'
hitsidr=$hitsidr'_epst'$epst'_epend'$epend'_step'$estep

mkdir -p $hitsidr


echo '#!/bin/bash' > $f
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/python/anaconda3-5.1.0/v5.1.0/envs/new-tf-gpu/lib' >> $f
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

echo 'export  HDF5_USE_FILE_LOCKING=FALSE' >> $f
echo '$PYTHON  $WD/plot_post.py   \' >> $f
echo '					   --ref_mode $REF_MODE \' >> $f
echo '					   --lr $lr \' >> $f
echo '					   --epst $epst \' >> $f
echo '					   --epend $epend \' >> $f
echo '					   --step $estep \' >> $f
echo "					   --postfile $WD/post/$PYMODEL"'_mel80_lr'$lr'_ref'$ref_mode' \' >> $f
echo "					   --output $hitsidr/$PYMODEL"'_mel80_lr'$lr'_ref'$ref_mode' \' >> $f
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


chmod +x $f


jid=$($SUBMITJOB $WAITJOB $l $f | grep -E [0-9]+)
echo $f
echo "Submitted as: ${jid}"
