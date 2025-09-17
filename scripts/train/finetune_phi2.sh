#!/bin/bash

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
VERSION="$3"
TRAIN_RECIPE="$4"
TUNE_TYPE_LLM="$5"
OUTPUT_DIR="$6"
RUN_NAME="$7"

NUM_TRAIN_EPOCHS="${8:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${9:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${10:-4}"
GRADIENT_ACCUMULATION_STEPS="${11:-4}"
LEARNING_RATE="${12:-2e-4}"

MORES_CONFIG_PATH="${13:-None}"
INTERVENTION_POSITIONS="${14:-None}"
MORES_SHARE_WEIGHTS="${15:-None}"
INTERVENE_MODALITY="${16:-None}"

INTERVENTION_POSITIONS_2="${17:-None}"
MORES_SHARE_WEIGHTS_2="${18:-None}"
INTERVENE_MODALITY_2="${19:-None}"

TINYLLAVA_VERSION="${20:-None}"
PRETRAINED_MODEL_PATH="${21:-None}"

CONV_VERSION="${22:-None}"
LLM_VERSION="${23:-None}"
VT_VERSION="${24:-None}"
VT_VERSION2="${25:-None}"
CN_VERSION="${26:-None}"
MODEL_MAX_LENGTH="${27:-None}"

LORA_R="${28:-128}"
LORA_ALPHA="${29:-256}"

GIN_NUM_LAYERS="${30:-5}"
GIN_HIDDEN_DIM="${31:-300}"
GRAPH_DROP_RATIO="${32:-0.1}"
GRAPH_POOLING="${33:-mean}"
GRAPH_INIT_CHECKPOINT="${34:-None}"

TINYLLAVA_VERSION_NAME=$(echo $TINYLLAVA_VERSION | cut -d'/' -f2)

cmd="deepspeed --include localhost:7 --master_port 25565 tinyllava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --epoch_to_save 1 \
    --save_step 2000 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name $RUN_NAME \
    --gin_num_layers $GIN_NUM_LAYERS \
    --gin_hidden_dim $GIN_HIDDEN_DIM \
    --graph_drop_ratio $GRAPH_DROP_RATIO \
    --graph_pooling $GRAPH_POOLING \
    --graph_init_checkpoint $GRAPH_INIT_CHECKPOINT"

if [ "$MORES_CONFIG_PATH" != "None" ]; then
    cmd="$cmd --mores_config_path \"$MORES_CONFIG_PATH\""
fi

if [ "$INTERVENTION_POSITIONS" != "None" ]; then
    cmd="$cmd --intervention_positions \"$INTERVENTION_POSITIONS\""
fi
if [ "$MORES_SHARE_WEIGHTS" != "None" ]; then
    cmd="$cmd --mores_share_weights \"$MORES_SHARE_WEIGHTS\""
fi
if [ "$INTERVENE_MODALITY" != "None" ]; then
    cmd="$cmd --intervene_modality \"$INTERVENE_MODALITY\""
fi

if [ "$INTERVENTION_POSITIONS_2" != "None" ]; then
    cmd="$cmd --intervention_positions_2 \"$INTERVENTION_POSITIONS_2\""
fi
if [ "$MORES_SHARE_WEIGHTS_2" != "None" ]; then
    cmd="$cmd --mores_share_weights_2 \"$MORES_SHARE_WEIGHTS_2\""
fi
if [ "$INTERVENE_MODALITY_2" != "None" ]; then
    cmd="$cmd --intervene_modality_2 \"$INTERVENE_MODALITY_2\""
fi

if [ "$TINYLLAVA_VERSION" != "None" ]; then
    cmd="$cmd --tinyllava_version \"$TINYLLAVA_VERSION\""
fi

if [ "$PRETRAINED_MODEL_PATH" != "None" ]; then
    cmd="$cmd --pretrained_model_path \"$PRETRAINED_MODEL_PATH\""
fi

if [ "$CONV_VERSION" != "None" ]; then
    cmd="$cmd --conv_version \"$CONV_VERSION\""
fi
if [ "$LLM_VERSION" != "None" ]; then
    cmd="$cmd --model_name_or_path \"$LLM_VERSION\""
fi
if [ "$VT_VERSION" != "None" ]; then
    cmd="$cmd --vision_tower \"$VT_VERSION\""
fi
if [ "$VT_VERSION2" != "None" ]; then
    cmd="$cmd --vision_tower2 \"$VT_VERSION2\""
fi
if [ "$CN_VERSION" != "None" ]; then
    cmd="$cmd --connector_type \"$CN_VERSION\""
fi
if [ "$MODEL_MAX_LENGTH" != "None" ]; then
    cmd="$cmd --model_max_length \"$MODEL_MAX_LENGTH\""
fi

if [ "$TUNE_TYPE_LLM" != "None" ]; then
    cmd="$cmd --tune_type_llm \"$TUNE_TYPE_LLM\""
fi

if [ "$LORA_R" != "None" ]; then
    cmd="$cmd --lora_r \"$LORA_R\""
fi

if [ "$LORA_ALPHA" != "None" ]; then
    cmd="$cmd --lora_alpha \"$LORA_ALPHA\""
fi

eval $cmd