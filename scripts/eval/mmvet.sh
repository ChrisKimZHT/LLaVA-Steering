#!/bin/bash

MODEL_PATH=""
MODEL_NAME="tinyllava_phi2-mores-r4-f4l5-v"
CONV_MODE="phi"

EVAL_DIR="/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/eval"

python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mm-vet/llava-mm-vet.jsonl \
    --image-folder $EVAL_DIR/mm-vet/images \
    --answers-file $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

mkdir -p $EVAL_DIR/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --dst $EVAL_DIR/mm-vet/results/$MODEL_NAME.json
