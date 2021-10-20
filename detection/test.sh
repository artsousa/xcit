#!/bin/bash

WK_DIR=$(pwd)

CONFIG=$WK_DIR/configs/xcit/faster
TOOLS=$WK_DIR/tools
BBONE=$WK_DIR/backbone

$TOOLS/dist_test.sh $CONFIG/retina_monitors.py \
	$WK_DIR/outputs/epoch_5.pth 1 \
       --eval bbox --show	

