import json
from BO import run_BO_for_LLM, joint_opt_BO_LLM, joint_opt_BO_LLM_only_model, joint_opt_BO_LLM_only_data, joint_opt_BO_LLM_fixed_feature_list

from argparse import ArgumentParser
from transformers import TrainerCallback
import time

run_name = "BO_runs_LLM_joint_optimization"
parser = ArgumentParser()
parser.add_argument("--contaminate", help="to contaminate training data?", type=int, default=0) # WIP. not used.
parser.add_argument("--iterations", help="iterations BO?", type=int, default=10)
parser.add_argument("--num_data", help="total_data?", type=int, default=10000)
parser.add_argument("--epochs", help="epochs", default=1)
parser.add_argument("--trials", help="trials", default=1)
parser.add_argument("--evaluation_cuda", help="evaluation_cuda", default=0)
parser.add_argument("--sample_method", help="sample_method", default="random") # random, IF_random, IF_remove_harmful. But default to random
parser.add_argument("--eval_tasks", help="eval_tasks") # see task_metrics for different tasks
parser.add_argument("--experiments_setting", help="either ood or in_dist")
parser.add_argument("--output_dir", help="output_dir")
parser.add_argument("--time_limit", help="time_limit")
parser.add_argument("--lora_rank", help="lora_rank")
parser.add_argument("--ucb_beta", help="lora_rank")
parser.add_argument("--limit", help="no. of samples for performance evaluation. Default is 100", default=100)
parser.add_argument("--run_BO_on", help="all or model", default="all")
parser.add_argument("--model")

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

# WIP
to_contaminate= bool(args["contaminate"])
if not to_contaminate:
    influence_path="influence/"
    print("getting influence from: ", influence_path)
else:
    influence_path="influence/contaminated/"
    print("getting influence from: ", influence_path)

setting=(args["experiments_setting"])
output_dir=(args["output_dir"])
time_limit = int(args["time_limit"])
epochs=int(args["epochs"])
trials=int(args["trials"])
cuda=int(args["evaluation_cuda"])
cuda="cuda:"+str(cuda)
BO_run = int(args["iterations"])
total_data = int(args["num_data"])
sample_methods = str(args["sample_method"]).split(",")
tasks = str(args["eval_tasks"]).split(",") # list of eval task
evaluation_weights = [1/len(tasks)] * len(tasks)
lora_rank =int(args["lora_rank"])
ucb_beta = float(args["ucb_beta"])
run_BO_on = str(args["run_BO_on"]) # either "model" or "data"
limit = int(args["limit"]) # either "model" or "data" or "all"
model_id = str(args["model"])

if limit < 0:
    limit = None

import random
import string
seed = random.randint(0,1000)
# Generate a random string of size 5 using uppercase and lowercase letters
random_string = ''.join(random.choices(string.ascii_letters, k=5))
print("random sentence created:", random_string)

task_metrics = {
  "commonsense_qa": "acc,none",
  "gsm8k": "exact_match,strict-match",
  "headqa_en": "acc,none",
  "hellaswag": "acc,none",
  "pubmedqa": "acc,none",
  "sciq": "acc_norm,none",
  "triviaqa": "exact_match,remove_whitespace",
  "truthfulqa_gen": "bleu_acc,none",
  "wikitext": "word_perplexity,none",
}

# set up training data (depending if we want ood)
data_domains_initial = list(task_metrics.keys())
print("current eval task: ", tasks)
if setting == "ood":
    data_domains =  [x for x in data_domains_initial if x not in tasks] # remove training domain that is in task
else:
    data_domains = [x for x in data_domains_initial]

# set up evaluation tasks (and weights, if we have more than one evaluation task)
evaluation_task = {}
for task, weight in zip(tasks, evaluation_weights):
    evaluation_task[task] = (float(weight), task_metrics[task])

print("evaluation tasks and weights: ", evaluation_task)

train_epochs = 1
training_batch = 8
evaluation_batch = 4
evaluation_steps=50000
final_info_stored = {"command line args": args,
                    "hash string": random_string,
                    "training domain": data_domains,
                    "evaluation domain": tasks,
                    "weight": evaluation_weights} # weight in str

for sample_method in sample_methods: # random sampling
    results = []
    for x in range(trials):
        #model_id="Qwen/Qwen2.5-7B-Instruct" # pass this into next function if necessary
        #model_id: str = "LLM/llama_8b_instruct"
        if run_BO_on == "all": # run BO on both data and model
            print("running BO on both data and model")
            GP_input, observed_output, gp = joint_opt_BO_LLM(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
                                                        random_dir = random_string, 
                                                            BO_run = BO_run,
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            sampling_method = sample_method, 
                                                            train_epochs=train_epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            ucb_beta=ucb_beta,
                                                            limit=limit,
                                                            printout=True)
        elif run_BO_on == "all_fixed_features":
            print("running BO on both data and model with fixed feature list")
            GP_input, observed_output, gp = joint_opt_BO_LLM_fixed_feature_list(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
                                                        random_dir = random_string, 
                                                            BO_run = BO_run,
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            sampling_method = sample_method, 
                                                            train_epochs=train_epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            ucb_beta=ucb_beta,
                                                            limit=limit,
                                                            printout=True)
        elif run_BO_on == "model": # run BO only on model
            print("running BO only on model")
            GP_input, observed_output, gp = joint_opt_BO_LLM_only_model(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
                                                        random_dir = random_string, 
                                                            BO_run = BO_run,
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            sampling_method = sample_method, 
                                                            train_epochs=train_epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            ucb_beta=ucb_beta,
                                                            limit=limit,
                                                            printout=True)
        elif run_BO_on == "data":
            print("running BO only on data")
            GP_input, observed_output, gp = joint_opt_BO_LLM_only_data(time_callback=TimerCallback(time_limit), default_alpha=16, default_dropout=0.05, default_layer=[1,1,1,1,1],
                                                                       default_num_layers_to_apply=16, default_rank=16,
                                                                       data_domains = data_domains,
                                                                        random_dir = random_string, 
                                                                        BO_run = BO_run,
                                                                        total_data = total_data, 
                                                                        evaluation_cuda = cuda, 
                                                                        evaluation_task = evaluation_task,
                                                                        sampling_method = sample_method, 
                                                                        train_epochs=train_epochs, 
                                                                        training_batch=training_batch, 
                                                                        evaluation_batch=evaluation_batch,
                                                                        eval_steps=evaluation_steps,
                                                                        ucb_beta=ucb_beta,
                                                                        limit=limit,
                                                                        printout=True)
        else:
            assert False

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
print("final results: ", final_info_stored)
# store results
# try:
#     with open("LLM/BO/" + output_dir + "/" +run_name+ "_".join(tasks)+".json", 'w') as f:
#         json.dump(final_info_stored, f)
# except:
#     print("error with storing json, it's ok. moving to next...")





