#!/bin/bash

# GPU config
USE_TPU=False

# Experiment (data/checkpoint/directory) config
DATA_PATH=./ASCAD.h5
DATASET=ASCAD
CKP_DIR_BASE=./checkpoints # Base directory for checkpoints
RESULT_PATH_BASE=./results # Base directory for results

# Optimization config
LEARNING_RATE=2.5e-4
CLIP=0.25
MIN_LR_RATIO=0.004
INPUT_LENGTH=10000 # or 40000
DATA_DESYNC=200 # 400 for input length 40K

# Training config
TRAIN_BSZ=16
EVAL_BSZ=16
TRAIN_STEPS=4000000
WARMUP_STEPS=1000000
ITERATIONS=20000 # Maybe reduce this for faster hyperparameter sweeps
SAVE_STEPS=40000 # Maybe save less frequently during sweeps

# Model config
N_LAYER=2
D_MODEL=128
D_HEAD=32
N_HEAD=8
D_INNER=256
N_HEAD_SM=8
D_HEAD_SM=16
DROPOUT=0.05
# This flag is present in your working script, so we keep it.
# It appears to be harmlessly ignored by train_trans.py
DROPATT=0.0
CONV_KERNEL_SIZE=3 # The kernel size of the first convolutional layer is set to 11
N_CONV_LAYER=2
POOL_SIZE=10 # Adjust based on dataset/paper, e.g., 8 for ASCADf
D_KERNEL_MAP=512 # Consider testing 256, 128 as well
MODEL_NORM='preLC'
HEAD_INIT='forward'
SM_ATTN=True

# Evaluation config
MAX_EVAL_BATCH=100
OUTPUT_ATTN=False

# --- Beta Optimization ---
# Define the range of BETA_HAT_2 values to test
BETA_VALUES=(150)


if [[ $1 == 'train' ]]; then
    echo "Starting Beta Optimization Training & Testing..."
    
    # Ensure base directories exist
    mkdir -p "${CKP_DIR_BASE}"
    mkdir -p "${RESULT_PATH_BASE}"

    for BETA in "${BETA_VALUES[@]}"; do
        echo "--- Training with BETA_HAT_2 = $BETA ---"

        # Create specific directories and paths for this beta value
        CURRENT_CKP_DIR="${CKP_DIR_BASE}/beta_${BETA}"
        CURRENT_RESULT_PATH_TRAIN="${RESULT_PATH_BASE}/beta_${BETA}_train" # Log for training
        CURRENT_RESULT_PATH_TEST="${RESULT_PATH_BASE}/beta_${BETA}_test"   # .txt file for test results
        
        mkdir -p "${CURRENT_CKP_DIR}"

        # --- RUN TRAINING ---
        # This command is copied from your "working" script, with paths/beta updated
        python train_trans.py \
            --use_tpu=${USE_TPU} \
            --data_path=${DATA_PATH} \
            --dataset=${DATASET} \
            --checkpoint_dir="${CURRENT_CKP_DIR}" \
            --warm_start=False \
            --result_path="${CURRENT_RESULT_PATH_TRAIN}" \
            --learning_rate=${LEARNING_RATE} \
            --clip=${CLIP} \
            --min_lr_ratio=${MIN_LR_RATIO} \
            --warmup_steps=${WARMUP_STEPS} \
            --input_length=${INPUT_LENGTH} \
            --data_desync=${DATA_DESYNC} \
            --train_batch_size=${TRAIN_BSZ} \
            --eval_batch_size=${EVAL_BSZ} \
            --train_steps=${TRAIN_STEPS} \
            --iterations=${ITERATIONS} \
            --save_steps=${SAVE_STEPS} \
            --n_layer=${N_LAYER} \
            --d_model=${D_MODEL} \
            --d_head=${D_HEAD} \
            --n_head=${N_HEAD} \
            --d_inner=${D_INNER} \
            --n_head_softmax=${N_HEAD_SM} \
            --d_head_softmax=${D_HEAD_SM} \
            --dropout=${DROPOUT} \
            --dropatt=${DROPATT} \
            --conv_kernel_size=${CONV_KERNEL_SIZE} \
            --n_conv_layer=${N_CONV_LAYER} \
            --pool_size=${POOL_SIZE} \
            --d_kernel_map=${D_KERNEL_MAP} \
            --beta_hat_2=${BETA} \
            --model_normalization=${MODEL_NORM} \
            --head_initialization=${HEAD_INIT} \
            --softmax_attn=${SM_ATTN} \
            --max_eval_batch=${MAX_EVAL_BATCH} \
            --do_train=True

        echo "--- Finished training. Now testing BETA_HAT_2 = $BETA ---"
    done

elif [[ $1 == 'test' ]]; then
    # --- RUN TESTING ---
    # This command is copied from your "working" script's test block
    # It will use the checkpoint we just saved in CURRENT_CKP_DIR
    # --checkpoint_idx=0 tells the script to use the *latest* checkpoint
    echo "--- Testing ---"

    for BETA in "${BETA_VALUES[@]}"; do
        CURRENT_CKP_DIR="${CKP_DIR_BASE}/beta_${BETA}"
        CURRENT_RESULT_PATH_TEST="${RESULT_PATH_BASE}/beta_${BETA}_test"   # .txt file for test results
        
        echo "--- Testing with BETA_HAT_2 = $BETA ---"
        python train_trans.py \
            --use_tpu=${USE_TPU} \
            --data_path=${DATA_PATH} \
            --dataset=${DATASET} \
            --checkpoint_dir="${CURRENT_CKP_DIR}" \
            --checkpoint_idx=0 \
            --warm_start=False \
            --result_path="${CURRENT_RESULT_PATH_TEST}" \
            --learning_rate=${LEARNING_RATE} \
            --clip=${CLIP} \
            --min_lr_ratio=${MIN_LR_RATIO} \
            --warmup_steps=${WARMUP_STEPS} \
            --input_length=${INPUT_LENGTH} \
            --train_batch_size=${TRAIN_BSZ} \
            --eval_batch_size=${EVAL_BSZ} \
            --train_steps=${TRAIN_STEPS} \
            --iterations=${ITERATIONS} \
            --save_steps=${SAVE_STEPS} \
            --n_layer=${N_LAYER} \
            --d_model=${D_MODEL} \
            --d_head=${D_HEAD} \
            --n_head=${N_HEAD} \
            --d_inner=${D_INNER} \
            --n_head_softmax=${N_HEAD_SM} \
            --d_head_softmax=${D_HEAD_SM} \
            --dropout=${DROPOUT} \
            --dropatt=${DROPATT} \
            --conv_kernel_size=${CONV_KERNEL_SIZE} \
            --n_conv_layer=${N_CONV_LAYER} \
            --pool_size=${POOL_SIZE} \
            --d_kernel_map=${D_KERNEL_MAP} \
            --beta_hat_2=${BETA} \
            --model_normalization=${MODEL_NORM} \
            --head_initialization=${HEAD_INIT} \
            --softmax_attn=${SM_ATTN} \
            --max_eval_batch=${MAX_EVAL_BATCH} \
            --output_attn=${OUTPUT_ATTN} \
            --do_train=False

        echo "--- Finished testing for BETA_HAT_2 = $BETA ---"
    done
    
    echo "=================================================="
    echo "Beta Optimization Training and Testing Complete."
    echo "--- Summary of Mean Key Ranks (Lower is better) ---"
    echo "Showing last line (mean rank) from each test result file:"
    
    # This will print the filename and the last line (mean key rank) for each test
    tail -n 1 ${RESULT_PATH_BASE}/beta_*_test.txt
else
    echo "Unknown argument: $1. Use 'train' to run the full optimization."
fi
