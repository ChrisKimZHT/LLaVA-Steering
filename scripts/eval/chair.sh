#!/bin/bash
MODEL_PATH=""
MODEL_NAME="tinyllava_phi2-mores-r4-f4l5-v"
CONV_MODE="phi"

EVAL_DIR="/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/eval"

python -m tinyllava.eval.model_vqa_chair \
    --model-path $MODEL_PATH \
    --annotation-folder $EVAL_DIR/CHAIR/annotations \
    --image-folder $EVAL_DIR/CHAIR/val2014 \
    --answers-file $EVAL_DIR/CHAIR/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE \
    --max_new_tokens 512 \
    --sample_seed 42 \
    --sample_num 500

python -m tinyllava.eval.chair_utils \
    --cap_file $EVAL_DIR/CHAIR/answers/$MODEL_NAME.jsonl \
    --image_id_key image_id --caption_key caption \
    --coco_path $EVAL_DIR/CHAIR/annotations \
    --save_path temp_outputs/$MODEL_NAME.json
