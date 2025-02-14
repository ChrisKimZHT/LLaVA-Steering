#!/bin/bash

MODEL_PATH=""
MODEL_NAME="tinyllava_phi2-mores-r4-f4l5-v"
CONV_MODE="phi"

EVAL_DIR="/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/eval"

python -m tinyllava.eval.model_vqa_pope \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
