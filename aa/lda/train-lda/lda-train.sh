#!/bin/bash
# created on Nov 16 2020
# @author: Jose Antonio Lopez @ Infirmary Rd
# submits the LDA trainer.


DATE=`date '+%Y-%m-%d %H:%M:%S'`

##this lines have to be added to bash scripts running python script. it activates the correct conda and python libraries
ENVIRONMENT=/aa/lda/tools/conda_env/lid_lda
source_conda='source /python/anaconda3/v3.7/etc/profile.d/conda.sh'
conda_activate='conda activate $ENVIRONMENT'
echo_conda_env='echo $CONDA_PREFIX'
conda_deactivate='conda deactivate'


#features settings
#waitjob=''

N_COMP=512  #the size of the gmm used for the quantization

#LDA settings
N_TOPICS=16 #4,8,16,32,64
EPOCHS=20

ITERATIONS=50 #50 #The pases for the sampling on the E step
ALPHA=1 #it declares a uniform distribution for the dirichlet prior. (higher means scattered clusters)
ETA=0.0001 #

gmm_basename=gmm.ITSLang_


#required paths
WAV_SCP_TRAIN=/aa/lib/INA.v1.cut1-6.phone.train
WAV_SCP_TEST=/aa/lib/INA.v1.cut1-6.phone.test
EXP_DIR=/aa/lda/train-lda
SCP_DIR=
GRAPHS_DIR=$EXP_DIR/graphs
LDA_MODEL_DIR=$EXP_DIR/models #save the gensim lda model
VQ_DIR=/aa/lda/gmm-vq/vq-data
TOOLS_DIR=/aa/lda/tools
PZ_DIR=$EXP_DIR/pz

TRAIN_SCP=$EXP_DIR/scpfiles/train_lda.$gmm_basename$N_COMP-train.scp
TEST_SCP=$EXP_DIR/scpfiles/train_lda.$gmm_basename$N_COMP-test.scp
#get the vq lists
VQ_TRAIN_FILELIST=$VQ_DIR/$gmm_basename$N_COMP.vq.train
VQ_TEST_FILELIST=$VQ_DIR/$gmm_basename$N_COMP.vq.test

#############################################################

rm -f $TRAIN_SCP
rm -f $TEST_SCP

cp $WAV_SCP_TRAIN $TRAIN_SCP
cp $WAV_SCP_TEST $TEST_SCP


############################################################

LDA_TRAIN_SCRIPT=$EXP_DIR/lda-train.py

#modify the feature list file path if context is required.

BASE_NAME=lda.t$N_TOPICS.ep$EPOCHS.$gmm_basename$N_COMP


submitjob=''

f="$EXP_DIR/LOG/lda-train.$BASE_NAME.sh"
l="$EXP_DIR/LOG/lda-train.$BASE_NAME.log"


echo '#!/bin/bash' > $f
echo $environment_name >> $f
echo $source_conda >> $f
echo $conda_activate >> $f
echo $echo_conda_env >> $f

echo "export N_COMP=$N_COMP" >> $f
echo "export BASE_NAME=$BASE_NAME" >> $f
echo "export VQ_TRAIN_FILELIST=$VQ_TRAIN_FILELIST" >> $f
echo "export VQ_TEST_FILELIST=$VQ_TEST_FILELIST" >> $f
echo "export N_TOPICS=$N_TOPICS" >> $f
echo "export EPOCHS=$EPOCHS" >> $f
echo "export ITERATIONS=$ITERATIONS" >> $f
echo "export ALPHA=$ALPHA" >> $f
echo "export ETA=$ETA" >> $f
echo "export LDA_MODEL_DIR=$LDA_MODEL_DIR" >> $f
echo "export GRAPHS_DIR=$GRAPHS_DIR" >> $f
echo "export TOOLS_DIR=$TOOLS_DIR" >> $f
echo "export EXP_DIR=$EXP_DIR" >> $f
echo "export PZ_DIR=$PZ_DIR" >> $f
echo "export TRAIN_SCP=$TRAIN_SCP" >> $f
echo "export TEST_SCP=$TEST_SCP" >> $f

echo "echo $LDA_TRAIN_SCRIPT" >> $f
#echo "python" >> $f
echo "python $LDA_TRAIN_SCRIPT" >> $f
echo "conda deactivate">> $f

chmod +x $f

jid=$($submitjob $waitjob $l $f | grep -E [0-9]+)

echo "LDA train file $f"
echo "With log: $l"
echo "Submitted as: ${jid}"
