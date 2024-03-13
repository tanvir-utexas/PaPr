#!/usr/bin/env bash

z=0.4

CONFIG=./configs/recognition/videomae_papr/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400.py
CHECKPOINT=./checkpoints/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth
WORK_DIR=./work_dirs/MAE_PaPr_Large/z_${z}

GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29509}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --cfg-options model.backbone.fraction=${z} model.backbone.proposal="x3ds"\
    --work-dir ${WORK_DIR} \
    --launcher pytorch ${@:4}