#!/bin/bash
export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=3

# ----------------------- #
# Shared configuration
# ----------------------- #
ITER=50
NUM_DATA=10000
EPOCHS=1
TRIALS=3
EXP_SETTING=ood # ood or in_dist
TIME_LIMIT=1000 # usually this is not a limiting factor because we will finish 1 epoch
LORA_RANK=128
LIMIT=100 # how many datapoints to evaluation in llm-harness. 0 means all points.
RUN_ON=general # optimize both components
TRAIN_BATCH=32
EVAL_BATCH=32
EVAL_METHOD=eval_loss # IMPORTANT. eval_loss or performance (either take loss or performance)
MODEL=llama-8b # IMPORTANT
UCB_BETA=20
OPT_METHOD=random # IMPORTANT related to BO for mixed problems. No need to change.
INFO_PRINTOUT=evaluate_on_loss_RANDOM_NO_BO # additional info to identify the experiment. Only affects the output file name.
ACQ_FUNC=ucb # EI or ucb

# ----------------------- #
# Task groups
# ----------------------- #
# group1=("triviaqa" "mmlu")
# group1=("arc_challenge")
# group1=("winogrande")
group1=("arc_challenge" "truthfulqa_gen" "gsm8k" "commonsense_qa" "mmlu" "triviaqa")
# ----------------------- #
# Function to run a job
# ----------------------- #
run_task() {
    local task=$1
    local seed=$RANDOM
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "EVAL_METHOD=$EVAL_METHOD"
    echo "MODEL=$MODEL"
    echo "ACQ=$ACQ_FUNC"
    echo "INFO=$INFO_PRINTOUT"
    echo "OUTPUT AT /home/chenzhil/printout/${MODEL}_${ACQ_FUNC}_${task}_${INFO_PRINTOUT}.out"
    echo "RESULTS WILL BE SAVED AT ${MODEL}_${ACQ_FUNC}_${task}_${INFO_PRINTOUT}.json"

    nohup python3 -u BO_runs_LLM_joint_optimization.py \
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
        --eval_method=$EVAL_METHOD \
        --seed=$seed \
        --acq_function=$ACQ_FUNC \
        --model=$MODEL \
        --ucb_beta=$UCB_BETA \
        --optimize_method=$OPT_METHOD \
        --save_name="${MODEL}_${ACQ_FUNC}_${task}_${INFO_PRINTOUT}.json" \
        > "/home/chenzhil/printout/${MODEL}_${ACQ_FUNC}_${task}_${INFO_PRINTOUT}.out" 2>&1
}

# ----------------------- #
# Run first group (3 tasks)
# ----------------------- #
for task in "${group1[@]}"; do
    echo "Running task in sequence: $task"
    run_task "$task"
done

wait