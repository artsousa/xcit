#!/bin/bash

WK_DIR=$(pwd)

CFGS=$WK_DIR/configs/configs/
TOOLS=$WK_DIR/tools
#BBONE=$WK_DIR/backbone

$TOOLS/dist_train.sh \
	$CFGS/ssd300_r18.py 1 \
	--work-dir $WK_DIR/outputs \
	--seed 42 \
	--gpus 1 \
	--deterministic \
