#!/bin/bash
#@author: Jose Antonio Lopez @ The University of Sheffield

#submits the lda posterior generator. To obtain posteriors across speakers.



DATE=`date '+%Y-%m-%d %H:%M:%S'`

SUBSET=test #{TRAIN;TEST; suitcase}
waitjob="-w 5728206"

N_TOPICS=16
N_COMP=512
EPOCHS=20

SUBSET=$(echo ${SUBSET,,})
WAV_SCP=/aa/lib/INA.v1.cut1-6.phone.$SUBSET

gmm_basename=gmm.ITSLang_
LDA_MODEL=lda.t$N_TOPICS.ep$EPOCHS.$gmm_basename$N_COMP

FEATURES_SCP=l2arctic.base.plp0_d_a_z.cmn.all-data_norm

FEATURES_SCP=$EXP_DIR/scpfiles/train_lda.$gmm_basename$N_COMP-$SUBSET.scp

#required paths
EXP_DIR=/aa/lda/train-lda
GRAPHS_DIR=$EXP_DIR/graphs
LDA_MODEL_DIR=$EXP_DIR/models #save the gensim lda model
VQ_DIR=/aa/lda/gmm-vq/vq-data
TOOLS_DIR=/aa/lda/tools
PZ_DIR=$EXP_DIR/pz


DATA_SCP=$EXP_DIR/scpfiles/lda_post.$gmm_basename$N_COMP-$SUBSET.scp
VQ_FILELIST=$EXP_DIR/scpfiles/$gmm_basename$N_COMP.vq.$SUBSET



#############################################################

rm -f $DATA_SCP
cp $WAV_SCP $DATA_SCP

#get the vq lists
rm -f $VQ_FILELIST
cp $VQ_DIR/$gmm_basename$N_COMP.vq.$SUBSET $VQ_FILELIST


LDA_POST_SCRIPT=$EXP_DIR/get-lda-posteriors.py

##this lines have to be added to bash scripts running python script. it activates the correct conda and python libraries
#activate python
ENVIRONMENT=/aa/lda/tools/conda_env/lid_lda
source_conda='source /python/anaconda3/v3.7/etc/profile.d/conda.sh'
conda_activate='conda activate $ENVIRONMENT'
echo_conda_env='echo $CONDA_PREFIX'
conda_deactivate='conda deactivate'

#modify the feature list file path if context is required.

BASE_NAME=INA.$SUBSET"_"$LDA_MODEL

MODEL_NAME=$LDA_MODEL_DIR/$LDA_MODEL

submitjob=''


f="$EXP_DIR/LOG/lda-post.$BASE_NAME.$SUBSET.sh"
l="$EXP_DIR/LOG/lda-post.$BASE_NAME.$SUBSET.log"

echo '#!/bin/bash' > $f
echo $source_conda >> $f
echo $conda_activate >> $f
echo $echo_conda_env >> $f

echo "export N_COMP=$N_COMP" >> $f
echo "export N_TOPICS=$N_TOPICS" >> $f
echo "export GRAPHS_DIR=$GRAPHS_DIR" >> $f
echo "export BASE_NAME=$BASE_NAME" >> $f
echo "export TOOLS_DIR=$TOOLS_DIR" >> $f
echo "export PZ_DIR=$PZ_DIR" >> $f
echo "export LDA_POST_SCRIPT=$LDA_POST_SCRIPT" >> $f
echo "export VQ_FILELIST=$VQ_FILELIST" >> $f
echo "export MODEL_NAME=$MODEL_NAME" >> $f
echo "export DATA_SCP=$DATA_SCP" >> $f
echo "export SUBSET=${SUBSET,,}" >> $f
echo "export EPOCHS=$EPOCHS" >> $f

echo "python $LDA_POST_SCRIPT" >> $f

echo "conda deactivate">> $f

chmod +x $f

jid=$($submitjob $waitjob $l $f | grep -E [0-9]+)

echo "LDA train file $f"
echo "With log: $l"
echo "Submitted as: ${jid}"
