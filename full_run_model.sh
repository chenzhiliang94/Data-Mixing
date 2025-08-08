export WANDB_API_KEY=e6f1b6b6a8508afc0013c18e2956418fa49662b5

TASKS=("commonsense_qa" "gsm8k" "headqa_en" "rowan_hellaswag" "pubmedqa" "sciq" "triviaqa" "truthfulqa_gen" "wikitext" "mmlu" "ai2_arc")
MODES=("model")
GPUS=(5)

for task in "${TASKS[@]}"; do
  mkdir -p "printout_BO/full_run/${task}"
  for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    gpu="${GPUS[$i]}"
    echo "Running task=$task, mode=$mode on GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u BO_runs_LLM_joint_optimization.py \
      --contaminate=0 \
      --iterations=60 \
      --num_data=5000 \
      --epochs=1 \
      --trials=3 \
      --evaluation_cuda=0 \
      --sample_method=random \
      --eval_tasks="$task" \
      --experiments_setting=ood \
      --output_dir=results_full_run/ \
      --lora_rank=128 \
      --time_limit=100 \
      --ucb_beta=5 \
      --limit=100 \
      --run_BO_on="$mode" \
      >> "printout_BO/full_run/${task}/${mode}.out" \
      2>> "printout_BO/full_run/${task}/${mode}.err" &
    echo "Launched $task-$mode on GPU $gpu with PID=$!" >> pid_log.txt
  done
  wait
done
wait