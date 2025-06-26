
import torch
import numpy as np
torch.set_warn_always(False)


from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from BO import arrange_lora_config
from LLM.llm import *
import shutil

from transformers import TrainerCallback
import time
import random
import string

def simple_train_and_evaluation(train_time_limit,
                                setting,
                                evaluation_cuda,
                                task,
                                lora_r,
                                num_layers_to_apply,
                                five_dim_vector,
                                dropout,
                                evaluation_limit,
                                train_epochs,
                                total_data,
                                model_id="LLM/llama_8b_instruct"):
    
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

    sample_methods = "random"
    tasks = [task]
    evaluation_weights = [1/len(tasks)] * len(tasks)
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

    data_domains_initial = list(task_metrics.keys())
    print("current eval task: ", tasks)
    if setting == "ood":
        data_domains =  [x for x in data_domains_initial if x not in tasks] # remove training domain that is in task
    else:
        data_domains = [x for x in data_domains_initial]
    mixing_ratio = (len(data_domains))*[float(1/len(data_domains))] # mixing ratio balanced
    evaluation_task = {}
    for task, weight in zip(tasks, evaluation_weights):
        evaluation_task[task] = (float(weight), task_metrics[task])

    print("evaluation tasks and weights: ", evaluation_task)

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)

    # lora_r, lora_dropout, num_layers_to_apply, five_dim_vector
    lora_config = arrange_lora_config(lora_r, dropout, num_layers_to_apply, five_dim_vector)
    
    time_callback=TimerCallback(train_time_limit)
    all_influences = []
    for train_domain in data_domains:
        all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    # sample from each domain and train a model
    path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_string, tokenizer=tokenizer, 
                                                    train_datasets=train_datasets, 
                                                    val_datasets=val_datasets, 
                                                    data_domains=data_domains, 
                                                    mixing_ratio=mixing_ratio, 
                                                    additional_info=all_influences, # not needed
                                                    total_number_datapoints=total_data, 
                                                    run_name="sanity_check",
                                                    method=sample_methods,
                                                    train_epochs=train_epochs, 
                                                    batch_size=training_batch,
                                                    lora_config=lora_config,
                                                    eval_steps=evaluation_steps, callback=[time_callback])
    # free gpu memory
    with torch.no_grad():
        torch.cuda.empty_cache()
    print("evaluating...")
    lora_path = path_to_final_model
    config = PeftConfig.from_pretrained(lora_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
    lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)

    tasks = list(evaluation_task.keys())

    results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=evaluation_limit)
    observed_performance = 0.0
    for task in evaluation_task:
        task_weight, metric = evaluation_task[task]
        perf = results["results"][task][metric]
        if task == "wikitext":
            perf = - perf # we want to maximize the score, so for perplexity we maximize instead
        observed_performance += (perf * task_weight)
    print("current iteration weighted performance: ", observed_performance)
    print("deleting lora model after evaluation.")
    shutil.rmtree(lora_path, ignore_errors=True)
    return observed_performance

def read_from_list(input_X):
    # [16, 1, 0, 1, 0, 0, 16, 0.05] -> num_layers_to_apply, five_dim_vector, lora_r, dropout
    lora_r = input_X[-2]
    five_dim_vector = input_X[1:6]
    num_layers_to_apply = input_X[0]
    dropout = input_X[7]
    return lora_r, num_layers_to_apply, five_dim_vector, dropout
# 5 dim:
# ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
results = {}

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--train_time", help="set to very high if want to train for epoch", type=int, default=1000000)
parser.add_argument("--setting", help="ood or in_dist", type=str)
parser.add_argument("--task", help="task?", type=str)
parser.add_argument("--evaluation_limit", help="evaluation_limit?", type=int)
parser.add_argument("--epochs", help="epochs?", type=int)
parser.add_argument("--total_data", help="e.g. 5000", type=int)
parser.add_argument("--lora_r", help="e.g. 128", type=int)
parser.add_argument("--num_layers", help="e.g. 16", type=int)

args = parser.parse_args()
train_time_limit = args.train_time
setting = args.setting
task = args.task # "gsm8k", "headqa_en", "hellaswag", "pubmedqa", "sciq", "triviaqa", "truthfulqa_gen", "wikitext"
evaluation_limit = args.evaluation_limit
train_epochs = args.epochs
total_data = args.total_data
lora_r = args.lora_r
num_layers_to_apply = args.num_layers

cuda="cuda:0"
evaluation_cuda = "cuda:0"

training_batch = 8
evaluation_batch = 4
evaluation_steps=50000

configurations = [
    [num_layers_to_apply, 0, 0, 0, 0, 1, lora_r, 0.05],
    [num_layers_to_apply, 0, 0, 0, 1, 1, lora_r, 0.05],
    [num_layers_to_apply, 0, 0, 1, 1, 1, lora_r, 0.05],
    [num_layers_to_apply, 0, 1, 1, 1, 1, lora_r, 0.05],
    [num_layers_to_apply, 1, 1, 1, 1, 1, lora_r, 0.05],
    [num_layers_to_apply, 1, 0, 0, 0, 0, lora_r, 0.05],
    [num_layers_to_apply, 1, 1, 0, 0, 0, lora_r, 0.05],
    [num_layers_to_apply, 1, 1, 1, 0, 0, lora_r, 0.05],
    [num_layers_to_apply, 1, 1, 1, 1, 0, lora_r, 0.05],
    [num_layers_to_apply, 1, 1, 1, 1, 1, lora_r, 0.05]
]

print("configurations: ", configurations)
for input_X in configurations:
    trial_perf = []
    for trial in range(3):
        lora_r, num_layers_to_apply, five_dim_vector, dropout = read_from_list(input_X)
        trial_perf.append(simple_train_and_evaluation(train_time_limit,
                                        setting,
                                        evaluation_cuda,
                                        task,
                                        lora_r,
                                        num_layers_to_apply,
                                        five_dim_vector,
                                        dropout,
                                        evaluation_limit,
                                        train_epochs,
                                        total_data))
        print("that performance was from: ", input_X)
    results[tuple(input_X)] = trial_perf