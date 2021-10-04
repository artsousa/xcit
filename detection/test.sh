#!/bin/bash

WK_DIR=$(pwd)

CONFIG=$WK_DIR/configs/xcit/
TOOLS=$WK_DIR/tools

$TOOLS/dist_test.sh \
	$CFGS/__mask_rcnn_xcit_small_12_p8_3x_coco.py \
	$WK_DIR/outputs/epoch_20.pth 1 \
#	--eval bbox \

