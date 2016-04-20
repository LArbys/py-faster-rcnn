#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB='rpn_uboone_train_'$4
TEST_IMDB='rpn_uboone_test_'$4
PT_DIR='rpn_uboone'

ITERS=400

LOG="experiments/logs/frcnn_alt_opt_ub_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_rpn.py --gpu ${GPU_ID} \
    --net_name ${NET} \
    --weights data/rpn_uboone_models/${NET}.caffemodel \
    --imdb ${TRAIN_IMDB} \
    --cfg experiments/cfgs/faster_rcnn_alt_opt_trimese_rms.yml \
    ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x
