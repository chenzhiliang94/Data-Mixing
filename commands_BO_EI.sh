export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

tasks=("arc_challenge" "gsm8k" "truthfulqa_gen" "commonsense_qa" "mmlu" "triviaqa")
max_jobs=3
for task in "${tasks[@]}"; do
    echo "Running task: $task"

    nohup python3 -u BO_runs_LLM_joint_optimization.py \
        --iterations=50 \
        --num_data=10000 \
        --epochs=1 \
        --trials=3 \
        --eval_tasks=$task \
        --experiments_setting=ood \
        --time_limit=500 \
        --lora_rank=128 \
        --limit=100 \
        --run_BO_on=general \
        --seed=42 \
        --acq_function=EI \
        --model=llama-8b \
        --ucb_beta=5.0 \
        --optimize_method=mixed \
        --save_name="EI_${task}.json" \
        > ~/printout/EI_${task}.out 2>&1 &
    
    # Wait until fewer than max_jobs are running
    while (( $(jobs -r | wc -l) >= max_jobs )); do
        sleep 10
    done
done

wait