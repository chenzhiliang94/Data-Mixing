
# run BO over everything, lora_rank is what's the max lora rank we can use (from 0 to lora_rank)
# --contaminate=0: unused
# --iterations=100 no. of BO steps
# --num_data=5000 no. of datapoints for fine-tuning
# --sample sample randomly from each dataset (don't need to change for now)
# --eval_tasks=gsm8k the evaluation task. ctrl-f the python file to find other task names
# --experiments_setting=ood We remove the eval_tasks from the pool of training datasets. alternative: in_dist
# --output_dir=output_joint_optimization/results_updated not so important
# --lora_rank=128 is what's the max lora rank we can use (from 0 to lora_rank)
# --time_limit=200 train for how long
# --ucb_beta=0.5 BO hyper param
# --limit=200 during evaluation, query over 200 eval questions; can save time for testing purposes.
# Use --limit=-1 for full results, but might be slow for certain eval domains.
# --run_BO_on=all run BO to optimize both data and model factors (mixing ratio, architecture etc) 
# --run_BO_on=all_fixed_features is similar to all, but we used some tricks to handle integer inputs.

# the difference between these two commands is just --run_BO_on=all_fixed_features and --run_BO_on=all
# --run_BO_on=all gives better results 
#CUDA_VISIBLE_DEVICES=2 python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=5 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=200 --ucb_beta=0.5 --limit=200 --run_BO_on=all_fixed_features >> printout_BO/BO_all_fixed_features.out

tasks=("commonsense_qa" "gsm8k" "headqa_en" "hellaswag" "pubmedqa" "sciq" "triviaqa" "truthfulqa_gen" "wikitext")

for task in "${tasks[@]}"; do
    echo "Running task: $task"

    CUDA_VISIBLE_DEVICES=3 nohup python3 -u BO_runs_LLM_joint_optimization.py \
        --contaminate=0 \
        --iterations=75 \
        --num_data=5000 \
        --epochs=1 \
        --trials=3 \
        --evaluation_cuda=0 \
        --sample_method=random \
        --eval_tasks="$task" \
        --experiments_setting=ood \
        --output_dir=output_joint_optimization/results_updated \
        --lora_rank=128 \
        --time_limit=200 \
        --ucb_beta=5 \
        --limit=200 \
        --run_BO_on=all \
        >> "output_joint_optimization/BO_final_${task}.out" 2>&1
    wait
done

# CUDA_VISIBLE_DEVICES=3 nohup python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=3 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=200 --ucb_beta=10 --limit=200 --run_BO_on=all >> printout_BO/BO_time_200_gsm8k_all_ucb_10.out

# CUDA_VISIBLE_DEVICES=3 nohup python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=3 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=200 --ucb_beta=15 --limit=200 --run_BO_on=all >> printout_BO/BO_time_200_gsm8k_all_ucb_15.out

# CUDA_VISIBLE_DEVICES=3 nohup python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=3 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=200 --ucb_beta=20 --limit=200 --run_BO_on=all >> printout_BO/BO_time_200_gsm8k_all_ucb_20.out

# CUDA_VISIBLE_DEVICES=3 nohup python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=3 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=200 --ucb_beta=25 --limit=200 --run_BO_on=all >> printout_BO/BO_time_200_gsm8k_all_ucb_25.out

# CUDA_VISIBLE_DEVICES=3 nohup python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=3 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=200 --ucb_beta=5 --limit=200 --run_BO_on=all >> printout_BO/BO_time_200_gsm8k_all_ucb_5.out

# CUDA_VISIBLE_DEVICES=3 nohup python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=3 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=200 --ucb_beta=0.1 --limit=200 --run_BO_on=all >> printout_BO/BO_time_200_gsm8k_all_ucb_.1.out