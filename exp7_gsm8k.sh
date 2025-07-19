export WANDB_API_KEY=e6f1b6b6a8508afc0013c18e2956418fa49662b5

TASK="gsm8k"
MODES=("model" "data" "all")
GPUS=(0 1 2)

for i in "${!MODES[@]}"; do
  mode="${MODES[$i]}"
  gpu="${GPUS[$i]}"

  echo "Running task=$TASK, mode=$mode on GPU=$gpu"

  CUDA_VISIBLE_DEVICES=$gpu python3 -u BO_runs_LLM_joint_optimization.py \
    --contaminate=0 \
    --iterations=100 \
    --num_data=5000 \
    --epochs=1 \
    --trials=3 \
    --evaluation_cuda=0 \
    --sample_method=random \
    --eval_tasks="$TASK" \
    --experiments_setting=ood \
    --output_dir=output_joint_optimization/results_updated \
    --lora_rank=128 \
    --time_limit=1000 \
    --ucb_beta=5 \
    --limit=100 \
    --run_BO_on="$mode" \
    >> "printout_BO/experiment_7/${TASK}_${mode}.out" \
    2>> "printout_BO/experiment_7/${TASK}_${mode}.err" &

  echo "Launched $TASK-$mode on GPU $gpu with PID=$!" >> pid_log2.txt
done

wait
