#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
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
PT_DIR="rpn_uboone"

LOG="experiments/logs/ub_test_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/uboone_diag.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_alt_opt/fast_rcnn_test.pt \
  --net output/faster_rcnn_alt_opt/rpn_uboone_train_$4/google_$4_faster_rcnn_final.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_alt_opt_UB_google_$4.yml \
    ${EXTRA_ARGS}
