DATA_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/text_files/blip_laion_cc_sbu_558k.json
IMAGE_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/llava/llava_pretrain/images

FINETUNE_TRAIN_RECIPE=common
TUNE_TYPE_LLM=frozen
OUTPUT_DIR=/home/hk-project-starter-p0022188/tum_piz8108/data/checkpoints/llava_factory/two_stage_pretrain/tinyllava_phi2-1st_stage
RUN_NAME=tinyllava_phi2-1st_stage

# training hyperparameters
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-3

CONV_VERSION=llama

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
MODEL_MAX_LENGTH=2048

bash scripts/train/finetune_phi2.sh "$DATA_PATH" "$IMAGE_PATH" "$VERSION" "$FINETUNE_TRAIN_RECIPE" "$TUNE_TYPE_LLM" "$OUTPUT_DIR" "$RUN_NAME" "$NUM_TRAIN_EPOCHS" "$PER_DEVICE_TRAIN_BATCH_SIZE" "$PER_DEVICE_EVAL_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$LEARNING_RATE" "$MORES_CONFIG_PATH" "$INTERVENTION_POSITIONS" "$MORES_SHARE_WEIGHTS" "$INTERVENE_MODALITY" "$INTERVENTION_POSITIONS_2" "$MORES_SHARE_WEIGHTS_2" "$INTERVENE_MODALITY_2" "$TINYLLAVA_VERSION" "$PRETRAINED_MODEL_PATH" "$CONV_VERSION" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$MODEL_MAX_LENGTH" "$LORA_R" "$LORA_ALPHA"