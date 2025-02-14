#!/bin/bash

MODEL_PATH=""
MODEL_NAME="tinyllava_phi2-mores-r4-f4l5-v"
CONV_MODE="phi"

EVAL_DIR="/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/eval"

python -m tinyllava.eval.model_vqa_mmmu \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MMMU/anns_for_eval.json \
    --image-folder $EVAL_DIR/MMMU/all_images \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python scripts/convert_answer_to_mmmu.py \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --answers-output $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json

cd $EVAL_DIR/MMMU/eval

python main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json
