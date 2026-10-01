#!/bin/bash

CHUNKS=1

# Script for multi-goal navigation
MODEL_PATH="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
CONFIG_PATH="VLN_CE/vlnce_baselines/config/langmap_baselines/planavid_langmap_seq.yaml"
SAVE_PATH="tmp/eval_planavid_multi_7b"
CHECK=${CHECK:-1}  # override from the shell, e.g. CHECK=0 sh <script>.sh

echo "Starting evaluation at $(date)"
echo "Using $CHUNKS GPUs"
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Starting GPU $IDX (CUDA_DEVICE=$(( IDX % 8 )))"
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python run_seq.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --model-path $MODEL_PATH \
    --result-path $SAVE_PATH \
    --exp-save "data" \
    --vlm 'qwen2.5-vl-7b-instruct' \
    --check $CHECK

done


python analyze_langmap_results.py \
  --eval-type multi \
  --results-dir $SAVE_PATH

echo "All processes started. Waiting for completion..."
wait
echo "All processes completed at $(date)"

