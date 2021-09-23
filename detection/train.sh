#!/bin/bash

WK_DIR=$(pwd)

CFGS=$WK_DIR/configs/xcit
TOOLS=$WK_DIR/tools
BBONE=$WK_DIR/backbone

$TOOLS/dist_train.sh \
	$CFGS/__mask_rcnn_xcit_small_12_p8_3x_coco.py 1 \
	--work-dir $WK_DIR/outputs \
	--seed 42 \
	--gpus 1 \
	--deterministic \
