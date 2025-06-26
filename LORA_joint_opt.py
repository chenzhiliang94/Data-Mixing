import os
from LLM.llm import extract_data_mixture_and_train, evaluate_tasks, load_data, get_tokenizer_and_model

from peft import PeftModel, PeftConfig
from copy import deepcopy

import torch
torch.set_warn_always(False)
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainerCallback
import time
    
import datasets

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
    
parser = ArgumentParser()
parser.add_argument("--domain", help="output_dir")
parser.add_argument("--time_limit", help="output_dir")
parser.add_argument("--few_shot", help="output_dir")
parser.add_argument("--trials", help="trials")
parser.add_argument("--batch", help="trials")

args = vars(parser.parse_args())
few_shot = int(args["few_shot"])
batch = int(args["batch"])
trials = int(args["trials"])
domain = args["domain"]
time_limit = int(args["time_limit"])
print("command-line args: ", args)

training_domain = ["gsm8k",
  "headqa_en",
  "hellaswag",
  "pubmedqa",
  "sciq",
  "triviaqa",
  "commonsense_qa", "truthfulqa_gen", "wikitext"]

data_domains = [domain]
training_batch = batch
evaluation_batch = 16
train_epochs = 1
cuda="cuda:0"
mixing_ratio = [1.0]
torch.manual_seed(42)
model_id = "LLM/llama_8b_instruct"

train_datasets = []
val_datasets = []
for data_domain in data_domains:
    train_dataset, val_dataset = load_data(data_domain=data_domain)
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)

resources = len(train_dataset)
results = {}
for time_limit in range(100, time_limit, 100):
    time_callback=TimerCallback(time_limit)
    results[time_limit] = {}
    for lora_r in [100,200,300,400,500,600,700,800,900,1000]:
        total_data = resources

        # get tokenizer and model

        lora_alpha = 16
        lora_dropout= 0.05
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ]
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        tokenizer, base_model = get_tokenizer_and_model(model_id = model_id)
        run_name = "test"
        post_training_performance_trials = []
        for _ in range(0,trials):
            path_to_final_model = extract_data_mixture_and_train(model=deepcopy(base_model), random_dir="joint_optimization_"+str(domain), tokenizer=tokenizer, 
                                                                train_datasets=train_datasets, 
                                                                val_datasets=val_datasets, 
                                                                data_domains=data_domains, 
                                                                mixing_ratio=mixing_ratio, 
                                                                additional_info=[None], 
                                                                total_number_datapoints=total_data, 
                                                                run_name=run_name, 
                                                                method="random",
                                                                train_epochs=train_epochs, 
                                                                batch_size=training_batch,
                                                                eval_steps=1000000, lora_config=config, callback=[time_callback])

            # evaluation
            lora_path = path_to_final_model #final_model_after_training
            config_eval = PeftConfig.from_pretrained(lora_path)
            model_eval = AutoModelForCausalLM.from_pretrained(config_eval.base_model_name_or_path, torch_dtype='auto')
            lora_model = PeftModel.from_pretrained(model_eval, lora_path).to(cuda)
            tokenizer = AutoTokenizer.from_pretrained(config_eval.base_model_name_or_path, trust_remote_code=True,)
            result = evaluate_tasks(data_domains, lora_model, tokenizer, evaluation_batch, few_shot)
            post_training_performance_trials.append(result['results'][domain])
            print("lora rank: ", lora_r)
            print("time limit: ", time_limit)
            print("results: ", result['results'][domain])

        results[time_limit][lora_r] = post_training_performance_trials
print("all results:", results)
import json

# Save it as a JSON file
with open("output_joint_optimization/" + domain + str(time_limit) + "_" + str(trials) + "_full_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
        
  