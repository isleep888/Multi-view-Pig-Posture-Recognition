DATA_DIR=#!/usr/bin/env bash
set -e

DATA_DIR="/data/yuehan/Swim_pose/multiview_pig_posture_recognition"

python pig_timm_resnet_attention_train.py \
  --data_dir "$DATA_DIR" \
  --out_dir "./pig_resnet_attention_outputs" \
  --models resnet34.a1_in1k resnet50.a1_in1k \
  --img_size 384 \
  --batch 96 \
  --workers 8 \
  --epochs 45 \
  --freeze_epochs 3 \
  --patience 9
