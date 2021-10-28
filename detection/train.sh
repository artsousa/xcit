#!/bin/bash

WK_DIR=$(pwd)

CFGS=$WK_DIR/configs/xcit/
TOOLS=$WK_DIR/tools
#BBONE=$WK_DIR/backbone

$TOOLS/dist_train.sh \
	$CFGS/rcnn_xcit_tiny12_p8.py 1 \
	--work-dir $WK_DIR/outputs \
	--seed 42 \
	--gpus 1 \
	--deterministic \
