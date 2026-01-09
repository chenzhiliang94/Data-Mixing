#!/bin/bash
export TQDM_DISABLE=1

# ----------------------- #
# Shared configuration
# ----------------------- #
ITER=50
NUM_DATA=10000
EPOCHS=1
TRIALS=3
EXP_SETTING=in_dist 
TIME_LIMIT=1000
LORA_RANK=128
LIMIT=100 
RUN_ON=general 
TRAIN_BATCH=16
EVAL_BATCH=16
MODEL=llama-8b 
UCB_BETA=20
OPT_METHOD=mixed 
USE_JOBS=1 
ACQ_FUNC=ucb 

# ----------------------- #
# Task definitions
# ----------------------- #
# First 3 tasks
tasks_set_1=("arc_challenge" "truthfulqa_gen" "commonsense_qa")

# Last 3 tasks
tasks_set_2=("triviaqa" "gsm8k" "mmlu")

# ----------------------- #
# Function to run a job
# ----------------------- #
# Arguments: $1=task_name, $2=gpu_id, $3=eval_method
run_task() {
    local task=$1
    local gpu_id=$2
    local method=$3
    local seed=13549
    
    # Update INFO_PRINTOUT dynamically to avoid file overwrite collisions
    # e.g., "eval_loss_in_dist" or "performance_in_dist"
    local local_info="${method}_${EXP_SETTING}"

    echo "STARTING Task: $task | GPU: $gpu_id | Method: $method"
    echo "OUTPUT AT /home/alfred/Data-Mixing/printout_final/new/${MODEL}_${ACQ_FUNC}_${task}_${local_info}.out"

    # We set CUDA_VISIBLE_DEVICES locally for this command
    CUDA_VISIBLE_DEVICES=$gpu_id python3 -u BO_runs_LLM_joint_optimization.py \
        --iterations=$ITER \
        --num_data=$NUM_DATA \
        --epochs=$EPOCHS \
        --trials=$TRIALS \
        --eval_tasks=$task \
        --experiments_setting=$EXP_SETTING \
        --time_limit=$TIME_LIMIT \
        --lora_rank=$LORA_RANK \
        --limit=$LIMIT \
        --run_BO_on=$RUN_ON \
        --training_batch=$TRAIN_BATCH \
        --evaluation_batch=$EVAL_BATCH \
        --eval_method=$method \
        --seed=$seed \
        --acq_function=$ACQ_FUNC \
        --model=$MODEL \
        --JoBS=$USE_JOBS \
        --ucb_beta=$UCB_BETA \
        --optimize_method=$OPT_METHOD \
        --save_name="${MODEL}_${ACQ_FUNC}_${task}_${local_info}.json" \
        > "/home/alfred/Data-Mixing/printout_final/new/${MODEL}_${ACQ_FUNC}_${task}_${local_info}.out" 2>&1

    echo "DONE: $task on GPU $gpu_id ($method)"
    echo "-----------------------------------"
}

# ----------------------- #
# Execution Blocks
# ----------------------- #

# 1. First 3 tasks, eval_method=eval_loss on GPU4
(
    echo "Launching Set 1 on GPU 1 (eval_loss)..."
    for task in "${tasks_set_1[@]}"; do
        run_task "$task" 1 "eval_loss"
    done
) &

# 2. Last 3 tasks, eval_method=eval_loss on GPU5
(
    echo "Launching Set 2 on GPU 4 (eval_loss)..."
    for task in "${tasks_set_2[@]}"; do
        run_task "$task" 4 "eval_loss"
    done
) &

# 3. First 3 tasks, eval_method=performance on GPU6
(
    echo "Launching Set 1 on GPU 5 (performance)..."
    for task in "${tasks_set_1[@]}"; do
        run_task "$task" 5 "performance"
    done
) &

# 4. Last 3 tasks, eval_method=performance on GPU7
(
    echo "Launching Set 2 on GPU 6 (performance)..."
    for task in "${tasks_set_2[@]}"; do
        run_task "$task" 6 "performance"
    done
) &

# Wait for all background processes (the 4 groups) to finish
wait
echo "All configurations completed."