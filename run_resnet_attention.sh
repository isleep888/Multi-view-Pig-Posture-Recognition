#!/usr/bin/env bash
set -e

# Change this to your dataset root.
# The directory should contain:
#   train1.csv
#   train1_images/
#   train2.csv
#   train2_images/
#   test.csv
#   test_images/
#   sample_submission.csv
DATA_DIR="/path/to/multiview_pig_posture_recognition"

python pig_timm_resnet_improved.py \
  --data_dir "$DATA_DIR" \
  --out_dir "./pig_resnet_attention_outputs" \
  --models resnet34.a1_in1k resnet50.a1_in1k \
  --img_size 384 \
  --batch 96 \
  --workers 8 \
  --epochs 45 \
  --freeze_epochs 3 \
  --patience 9
