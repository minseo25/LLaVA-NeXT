# export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

LLM_VERSION="Qwen/Qwen2.5-1.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
# stage 2

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="kollava_onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_ko_caption1307k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint
MID_RUN_NAME="kollava_onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-finetune-v2"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="3" --nnodes="1" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path /vlm/LLaVA-NeXT/scripts/train/stage2.yaml \
    --image_folder /vlm/LLaVA-NeXT/images \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --pretrain_mm_mlp_adapter="/vlm/LLaVA-NeXT/ckpt/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_patch_merge_type spatial_meanpooling3x3 \
    --bf16 True \
    --run_name ${MID_RUN_NAME} \
    --output_dir "/vlm/LLaVA-NeXT/ckpt/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
# --model_max_length 32768 \
# --gradient_accumulation_steps 64 \