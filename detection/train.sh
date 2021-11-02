#!/bin/bash

WK_DIR=$(pwd)

CFGS=$WK_DIR/configs/custom/
TOOLS=$WK_DIR/tools
#BBONE=$WK_DIR/backbone

$TOOLS/dist_train.sh \
	$CFGS/frcnn_r50_fpn.py 1 \
	--work-dir $WK_DIR/outputs \
	--seed 42 \
	--gpus 1 \
	--deterministic \
