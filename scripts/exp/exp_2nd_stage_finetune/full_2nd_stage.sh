FINETUNE_DATA_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/text_files/llava_v1_5_mix665k.json
IMAGE_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train

VERSION=
FINETUNE_TRAIN_RECIPE=common
TUNE_TYPE_LLM=full
OUTPUT_DIR=/home/hk-project-starter-p0022188/tum_piz8108/data/checkpoints/llava_factory/two_stage_pretrain/tiny-llava-phi-2-siglip-so400m-patch14-384-base-pretrain-full
RUN_NAME=tiny-llava-phi-2-siglip-so400m-patch14-384-base-pretrain-full

# training hyperparameters
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5

CONV_VERSION=phi

PRETRAINED_MODEL_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/checkpoints/llava_factory/two_stage_pretrain/tiny-llava-phi-2-siglip-so400m-patch14-384-base-pretrain


bash scripts/train/finetune_phi2.sh "$FINETUNE_DATA_PATH" "$IMAGE_PATH" "$VERSION" "$FINETUNE_TRAIN_RECIPE" "$TUNE_TYPE_LLM" "$OUTPUT_DIR" "$RUN_NAME" "$NUM_TRAIN_EPOCHS" "$PER_DEVICE_TRAIN_BATCH_SIZE" "$PER_DEVICE_EVAL_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$LEARNING_RATE" "$MORES_CONFIG_PATH" "$INTERVENTION_POSITIONS" "$MORES_SHARE_WEIGHTS" "$INTERVENE_MODALITY" "$INTERVENTION_POSITIONS_2" "$MORES_SHARE_WEIGHTS_2" "$INTERVENE_MODALITY_2" "$TINYLLAVA_VERSION" "$PRETRAINED_MODEL_PATH" "$CONV_VERSION" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$MODEL_MAX_LENGTH" "$LORA_R" "$LORA_ALPHA"