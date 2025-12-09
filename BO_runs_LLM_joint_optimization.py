import random
import string
from BO import joint_opt_BO_LLM, joint_opt_BO_LLM_only_model, joint_opt_random, joint_opt_BO_LLM_only_data,  joint_opt_BO_LLM_with_dkl, joint_opt_BO_LLM_generalized

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)

from argparse import ArgumentParser
from transformers import TrainerCallback
import time

run_name = "BO_runs_LLM_joint_optimization"
parser = ArgumentParser()

parser.add_argument("--iterations", help="iterations BO?", type=int, default=10)
parser.add_argument("--num_data", help="total_data?", type=int, default=10000)
parser.add_argument("--epochs", help="epochs", default=1)
parser.add_argument("--trials", help="trials", default=1)
parser.add_argument("--evaluation_cuda", help="evaluation_cuda", default=0)
parser.add_argument("--eval_tasks", help="eval_tasks") # see task_metrics for different tasks
parser.add_argument("--eval_method", help="eval_method")
parser.add_argument("--experiments_setting", help="either ood or in_dist")
parser.add_argument("--time_limit", help="time_limit")
parser.add_argument("--lora_rank", help="max lora_rank")
parser.add_argument("--model", help="model name or path")

# BO stuff, 
parser.add_argument("--acq_function", help="acquisition function")
parser.add_argument("--ucb_beta", help="lora_rank", default=10.0)
parser.add_argument("--optimize_method", help="optimize_method")
parser.add_argument("--save_name", help="save_name")

parser.add_argument("--seed", help="seed value for single eval", type=int)
parser.add_argument("--limit", help="no. of samples for performance evaluation. Default is 100", default=100)
parser.add_argument("--run_BO_on", help="all or model or data or single_eval", default="all")
parser.add_argument("--training_batch", help="training_batch", type=int)
parser.add_argument("--evaluation_batch", help="evaluation_batch", type=int)
                    
parser.add_argument("--dkl_feature_dim", help="dkl feature dim", type=int, default=32)
parser.add_argument("--dkl_hidden", help="dkl hidden layers", type=int, default=64)
parser.add_argument("--dkl_freeze_nn", help="dkl freeze nn", type=bool, default=False)

# deepspeed
parser.add_argument("--local_rank", type=int, default=0)
class TimerCallback(TrainerCallback):
    def __init__(self, max_duration_seconds):
        self.max_duration = int(max_duration_seconds)
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_duration:
            print(f"⏰ Max training time of {self.max_duration} seconds reached. Stopping.")
            control.should_training_stop = True
        return control
    
args = vars(parser.parse_args())
print("command-line args: ", args)

eval_method = str(args["eval_method"])
model=str(args["model"])
setting=(args["experiments_setting"])
time_limit = int(args["time_limit"])
epochs=int(args["epochs"])
trials=int(args["trials"])
cuda=int(args["evaluation_cuda"])
cuda="cuda:"+str(cuda)
BO_run = int(args["iterations"])
total_data = int(args["num_data"])
tasks = str(args["eval_tasks"]).split(",") # list of eval task
evaluation_weights = [1/len(tasks)] * len(tasks)
lora_rank =int(args["lora_rank"])
ucb_beta = float(args["ucb_beta"])
run_BO_on = str(args["run_BO_on"]) # either "model" or "data"
limit = int(args["limit"]) # either "model" or "data" or "all"
sample_methods = ["random"] # only random sampling for now
save_name = str(args["save_name"])

acq_function = str(args["acq_function"])
optimize_method = str(args["optimize_method"])

training_domain_metrics = {
  "commonsense_qa": "acc,none",
  "gsm8k": "exact_match,strict-match",
  "rowan_hellaswag": "acc,none",
  "sciq": "acc_norm,none",
  "triviaqa": "exact_match,remove_whitespace",
  "truthfulqa_gen": "bleu_acc,none",
  "wikitext": "word_perplexity,none",
  "mmlu": "acc,none",
  "arc_challenge": "acc,none"
}

eval_metrics = {
  "commonsense_qa": "acc,none",
  "gsm8k": "exact_match,strict-match",
  "rowan_hellaswag": "acc,none",
  "sciq": "acc_norm,none",
  "triviaqa": "exact_match,remove_whitespace",
  "truthfulqa_gen": "bleu_acc,none",
  "wikitext": "word_perplexity,none",
  "mmlu": "acc,none",
  "arc_challenge": "acc,none"
}

# set up training data (depending if we want ood)
data_domains_initial = list(training_domain_metrics.keys())
print("current eval task: ", tasks)
if setting == "ood":
    data_domains =  [x for x in data_domains_initial if x not in tasks] # remove training domain that is in task
else:
    data_domains = [x for x in data_domains_initial]

# set up evaluation tasks (and weights, if we have more than one evaluation task)
evaluation_task = {}
for task, weight in zip(tasks, evaluation_weights):
    evaluation_task[task] = (float(weight), eval_metrics[task])

print("evaluation tasks and weights: ", evaluation_task)

train_epochs = int(args["epochs"])
training_batch = int(args["training_batch"])
evaluation_batch = int(args["evaluation_batch"])
evaluation_steps = 25
final_info_stored = {"command line args": args,
                    "training domain": data_domains,
                    "evaluation domain": tasks,
                    "weight": evaluation_weights} # weight in str
            
BO_params = {
    "acq_function": acq_function, # either "ucb" or "EI"
    "ucb_beta": ucb_beta,
    "optimize_method": optimize_method, # either "mixed" or "standard"
}

for sample_method in sample_methods: # random sampling
    results = []
    for x in range(trials):
        #model_id="Qwen/Qwen2.5-7B-Instruct" # pass this into next function if necessary
        #model_id: str = "LLM/llama_8b_instruct"
        
        rng = random.Random()
        seed = rng.randint(0, 1000)
        default_lora_config = LoraConfig(
                r=128,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",)
        
        if run_BO_on == "general": # run BO on both data and model
            print("running BO on both data and model")
            
            if model == "llama-8b":
                model_id="meta-llama/Meta-Llama-3-8B-Instruct"
            elif model == "qwen-7b":
                model_id="Qwen/Qwen2.5-7B-Instruct"
            elif model == "qwen-14b":
                model_id="Qwen/Qwen3-14B"
            elif model == "qwen-32b":
                model_id="Qwen/Qwen3-32B"
            else:
                assert False, "model not recognized"
            GP_input, observed_output, gp = joint_opt_BO_LLM_generalized(default_lora_config=default_lora_config, 
                                                                         time_callback=TimerCallback(time_limit),
                                                                         lora_rank_max=lora_rank,
                                                                         data_domains = data_domains,
                                                                        BO_run = BO_run,
                                                                        total_data = total_data,
                                                                        evaluation_task = evaluation_task,
                                                                        eval_method=eval_method,
                                                                        BO_params = BO_params,
                                                                        sampling_method = sample_method, 
                                                                        train_epochs=train_epochs, 
                                                                        training_batch=training_batch, 
                                                                        evaluation_batch=evaluation_batch,
                                                                        eval_steps=evaluation_steps,
                                                                        limit=limit,
                                                                        seed=seed,
                                                                        model_id=model_id,
                                                                        what_to_optimize="both")
            
         

        # if run_BO_on == "all": # run BO on both data and model
        #     print("running BO on both data and model")
        #     GP_input, observed_output, gp = joint_opt_BO_LLM(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
        #                                                 random_dir = random_string, 
        #                                                     BO_run = BO_run,
        #                                                     total_data = total_data, 
        #                                                     evaluation_cuda = cuda, 
        #                                                     evaluation_task = evaluation_task,
        #                                                     trial_number=x,
        #                                                     sampling_method = sample_method, 
        #                                                     train_epochs=train_epochs, 
        #                                                     training_batch=training_batch, 
        #                                                     evaluation_batch=evaluation_batch,
        #                                                     eval_steps=evaluation_steps,
        #                                                     ucb_beta=ucb_beta,
        #                                                     limit=limit,
        #                                                     printout=True,
        #                                                     seed=seed,
        #                                                     output_dir=output_dir)
        # elif run_BO_on == "model": # run BO only on model
        #     print("running BO only on model")
        #     GP_input, observed_output, gp = joint_opt_BO_LLM_only_model(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
        #                                                 random_dir = random_string, 
        #                                                     BO_run = BO_run,
        #                                                     total_data = total_data, 
        #                                                     evaluation_cuda = cuda, 
        #                                                     evaluation_task = evaluation_task,
        #                                                     trial_number=x,
        #                                                     sampling_method = sample_method, 
        #                                                     train_epochs=train_epochs, 
        #                                                     training_batch=training_batch, 
        #                                                     evaluation_batch=evaluation_batch,
        #                                                     eval_steps=evaluation_steps,
        #                                                     ucb_beta=ucb_beta,
        #                                                     limit=limit,
        #                                                     printout=True,
        #                                                     seed=seed,
        #                                                     output_dir=output_dir)
        # elif run_BO_on == "data":
        #     print("running BO only on data")
        #     GP_input, observed_output, gp = joint_opt_BO_LLM_only_data(time_callback=TimerCallback(time_limit), default_alpha=16, default_dropout=0.05, default_layer=[1,1,1,1,1],
        #                                                                default_num_layers_to_apply=16, default_rank=72,
        #                                                                data_domains = data_domains,
        #                                                                 random_dir = random_string, 
        #                                                                 BO_run = BO_run,
        #                                                                 total_data = total_data, 
        #                                                                 evaluation_cuda = cuda, 
        #                                                                 evaluation_task = evaluation_task,
        #                                                                 trial_number=x,
        #                                                                 sampling_method = sample_method, 
        #                                                                 train_epochs=train_epochs, 
        #                                                                 training_batch=training_batch, 
        #                                                                 evaluation_batch=evaluation_batch,
        #                                                                 eval_steps=evaluation_steps,
        #                                                                 ucb_beta=ucb_beta,
        #                                                                 limit=limit,
        #                                                                 printout=True,
        #                                                                 seed=seed,
        #                                                                 output_dir=output_dir)
        # elif run_BO_on == "random":
        #     print("using random configurations")
        #     GP_input, observed_output, = joint_opt_random(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
        #                                                 random_dir = random_string, 
        #                                                     BO_run = BO_run,
        #                                                     total_data = total_data, 
        #                                                     evaluation_cuda = cuda, 
        #                                                     evaluation_task = evaluation_task,
        #                                                     sampling_method = sample_method, 
        #                                                     train_epochs=train_epochs, 
        #                                                     training_batch=training_batch, 
        #                                                     evaluation_batch=evaluation_batch,
        #                                                     eval_steps=evaluation_steps,
        #                                                     ucb_beta=ucb_beta,
        #                                                     limit=limit,
        #                                                     printout=True,
        #                                                     seed=seed)
        # elif run_BO_on == "dkl":
        #     print("running BO with DKL")
        #     GP_input, observed_output, gp = joint_opt_BO_LLM_with_dkl(time_callback=TimerCallback(time_limit),
        #                                                             lora_rank_max=lora_rank, data_domains=data_domains,
        #                                                             random_dir=random_string, BO_run=BO_run, total_data=total_data,
        #                                                             evaluation_cuda=cuda, evaluation_task=evaluation_task,
        #                                                             ucb_beta=ucb_beta, sampling_method=sample_method,
        #                                                             train_epochs=train_epochs, training_batch=training_batch,
        #                                                             evaluation_batch=evaluation_batch, printout=True,
        #                                                             max_steps=evaluation_steps, eval_steps=evaluation_steps,
        #                                                             limit=limit, seed=seed,
        #                                                             dkl_feature_dim=args["dkl_feature_dim"], dkl_hidden=args["dkl_hidden"],dkl_freeze_nn=args["dkl_freeze_nn"])
        # else:
        #     assert False

        current_max = float('-inf')  # Start with negative infinity
        max_until_now = []           # List to store max values at each step

        # Iterate through the list
        for num in observed_output:
            current_max = max(current_max, num)  # Update the current maximum
            max_until_now.append(current_max)    # Store the max up to this step

        # best performance seen by BO at every step
        print("Best at every step:", max_until_now)
        results.append(max_until_now)
    final_info_stored[sample_method] = results
    
    
import json
import os

print("final results: ", final_info_stored)
# Combine the info you want to save
output_data = {
    "final_info_stored": final_info_stored,
    "BO_params": BO_params
}

# Define a path to save the JSON file
save_path = "/home/chenzhil/results/" + "_".join(tasks) + "/" + save_name  # You can change this to any directory you like, e.g., "/home/user/bo_results.json"

# Optionally create the directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Write to JSON
with open(save_path, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Results saved to {save_path}")







