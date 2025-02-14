#!/bin/bash

python -m tinyllava.text_eval.model_text_eval_hellaswag.py \
    --model_path /home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/two_stage_pretrain/tiny-llava-phi-2-siglip-so400m-patch14-384-base-pretrain-loreft-r1-f4l5-v/end \
    --model_name mores-r1-test1 \
    --conv-mode phi