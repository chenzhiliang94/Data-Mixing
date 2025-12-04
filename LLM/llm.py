import json
import time
#import torch_influence
import torch
import numpy as np
import matplotlib.pyplot as plt
from influence import *
torch.set_warn_always(False)
from copy import deepcopy
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import lm_eval
from lm_eval.models.huggingface import HFLM

import datasets
import os

import transformers
from datasets import concatenate_datasets
import gc

from peft import (
    get_peft_model,
)

from LLM.tokenize_util import *
import torch
import torch.nn as nn
# Define a simple MLP
class Predictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
        
def get_model_and_predict(X):
    
    # --- Recreate the model with correct input size ---
    input_dim = len(X)   # or set explicitly if you know the feature dimension
    model = Predictor(input_dim=input_dim)

    # --- Load the saved weights ---
    model_path = "output_joint_optimization/scaling_law_predictor/commonsense_qa_predictor.pt"
    model.load_state_dict(torch.load(model_path))

    # --- Set to eval mode for inference ---
    model.eval()

    # Example: make a prediction
    with torch.no_grad():
        sample = torch.tensor(X).unsqueeze(0)  # shape (1, input_dim)
        pred = model(sample)
        return pred
    
def get_tokenizer_and_model(model_id = "LLM/llama_8b_instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # adjust tokenizer (from alpaca repo, MIGHT NOT BE NEEDED IDK)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    model  = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto'
    )

    return tokenizer, model

batch_size = 8
num_epochs = 100
learning_rate= 3e-4
cutoff_len = 256
val_set_size = 2000
# lora hyperparams
lora_r = 8
lora_alpha = 16
lora_dropout= 0.05
lora_target_modules = [
    "q_proj",
    "v_proj",
]
# llm hyperparams
train_on_inputs = True  # if False, masks out inputs in loss
add_eos_token = False
group_by_length = False  # faster, but produces an odd training loss curve
gradient_accumulation_steps = 1

import numpy as np

from sklearn.utils.extmath import fast_logdet
from sklearn.metrics.pairwise import rbf_kernel

def get_log_det(kernel_values):
    return fast_logdet(kernel_values)

def get_kernel(list_of_embeddings):
    return rbf_kernel(list_of_embeddings, list_of_embeddings)

def get_next_best_datapoint(current_list_datapoint, current_log_det, dataset):
    
    best_idx = -1
    best_gain = -10000000000
    for idx, embedded_datapoint in enumerate(dataset): #datapoint is already an embedding
        new_list = current_list_datapoint + [embedded_datapoint]
        new_kernel = rbf_kernel(new_list, new_list)
        new_log_det = get_log_det(new_kernel)
        change_in_log_det = new_log_det - current_log_det
        
        if change_in_log_det > best_gain:
            best_idx = idx
            best_gain = change_in_log_det
    return best_idx, best_gain

def get_best_N_set(initial_dataset, N):
    current_chosen_dataset = []
    current_chosen_dataset.append(initial_dataset[0]) # take 1 datapoint at the start
    all_selected_idx = []
    all_selected_idx.append(0)
    for i in range(N):
        current_log_det = fast_logdet(rbf_kernel(current_chosen_dataset, current_chosen_dataset))
        best_idx, best_gain = get_next_best_datapoint(current_chosen_dataset, current_log_det, initial_dataset)
        current_chosen_dataset.append(initial_dataset[best_idx])
        all_selected_idx.append(best_idx)
        
    assert len(current_chosen_dataset) == N + 1
    return current_chosen_dataset, all_selected_idx

def sample(dataset, num_datapoints, additional_info, method, data_domain, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if method == "remove_harmful" and additional_info == None:
        assert False, "bad combination!"
    # print("method to use: ", method)
    # if random sample, just set all weight to 1
    if method == "random":
        additional_info = [1] * len(dataset)
    elif method == "IF_random":
        if sum(additional_info.tolist()) == 0:
            print("IF values are 0, go back to normal random sampling")
            additional_info = [1] * len(dataset)
    elif method == "IF_remove_harmful":
        if sum(additional_info.tolist()) == 0:
            print("IF values are 0, go back to normal random sampling")
            additional_info = [1] * len(dataset)
            method = "random"
    elif method == "log_det":
        pass
    else:
        assert False, "unknown method of sampling"

    if method == "log_det" and data_domain == None:
        method = "random"
    
    if method == "IF_remove_harmful":
        print("method is remove harmful, we will remove bottom 10% IF value datapoints")
        normalized_influences = deepcopy(additional_info)
        # remove lowest 10% data
        normalized_influences += abs(min(normalized_influences)) # make everything more than 0
        percentile_value = torch.quantile(additional_info, 0.2).item()
        # Set values below the 10th percentile to zero
        normalized_influences = normalized_influences.numpy()
        normalized_influences[normalized_influences < percentile_value] = 0
        
        num_harmful_points = sum(i == 0 for i in normalized_influences)
        
        # random sample from the rest of useful data uniformly
        normalized_influences[normalized_influences!=0] = 1/(len(normalized_influences) - num_harmful_points)
        indices = np.random.choice(len(normalized_influences), size=num_datapoints, p=normalized_influences) # sample from new list
    elif method == "log_det":
        # read the embedding from 
        embeddings = np.load("LLM/domain_training_embeddings/"+data_domain+".npy") # size N x embed_dim
        embeddings = np.squeeze(embeddings)
        _, indices = get_best_N_set(embeddings, num_datapoints)
    else:
        # print("method is to randomly sample")
        # influence sample; maybe use other methods
        normalized_influences = additional_info # normalized
        normalized_influences = np.asarray(normalized_influences).astype('float64')
        min_inf = abs(min(normalized_influences))
        normalized_influences += min_inf # normalized
        sum_inf = sum(normalized_influences)
        normalized_influences /= sum_inf # normalized
        normalized_influences[0] += (1 - sum(normalized_influences)) # bypass any smallprecision errors
        indices = np.random.choice(len(normalized_influences), size=num_datapoints, p=normalized_influences)
        # Use the `select` method to extract those samples
    sampled_dataset = dataset.select(indices)
    return sampled_dataset

def extract_data_mixture_and_train(model, tokenizer, train_datasets, val_datasets, data_domains, evaluation_dataset, mixing_ratio, additional_info, total_number_datapoints, seed=42, method="random", train_epochs=1, batch_size=8, max_step=-1, eval_steps=100, lora_config=None, callback=[]):

#     '''
#     model: llama base model
#     tokenizer: llama tokenizer
#     train_datasets: List of datasets
#     data_domains: List of data domain names, should be same size as train_datasets
#     mixing_ratio: List of mixing ratio, should sum to 1, list size same as train_datasets
#     additional_information: List of List of IF values for each dataset, for us to do sampling 
    
    all_sampled_train_data = [] # training data
    all_sampled_val_data = [] # same distribution as training data, but validation
    
    for train_dataset, val_dataset, data_domain, ratio, IF_values in zip(train_datasets, val_datasets, data_domains, mixing_ratio, additional_info):
        
        total_datapt = int(total_number_datapoints * ratio)
        if total_datapt == 0:
            continue # skip if no data needed

        sampled_train_data = sample(train_dataset, total_datapt, additional_info=IF_values, method=method, data_domain=data_domain, seed=seed)

        sampled_val_data = sample(val_dataset, total_datapt, additional_info=None, method="random", data_domain=None, seed=seed)

        
        sampled_train_data = sampled_train_data.shuffle(seed=seed).map(tokenizing_method[data_domain], fn_kwargs={"tokenizer": tokenizer,
                                                                                   "add_eos_token": add_eos_token,
                                                                                   "train_on_inputs": train_on_inputs,
                                                                                   })
        sampled_val_data = sampled_val_data.shuffle(seed=seed).map(tokenizing_method[data_domain], fn_kwargs={"tokenizer": tokenizer,
                                                                                   "add_eos_token": add_eos_token,
                                                                                   "train_on_inputs": train_on_inputs,
                                                                                   })
        
        sampled_train_data = sampled_train_data.select_columns(['input_ids', 'attention_mask', 'labels'])
        sampled_val_data = sampled_val_data.select_columns(['input_ids', 'attention_mask', 'labels'])
        
        all_sampled_train_data.append(sampled_train_data)
        all_sampled_val_data.append(sampled_val_data)
    
    combined_train_dataset = concatenate_datasets(all_sampled_train_data)
    # combined_val_dataset = concatenate_datasets(all_sampled_val_data)
    # combined_val_dataset = combined_val_dataset.shuffle(seed=42).select(range(100)) # this is validation data for training data
    
    print("length of training data: ", len(combined_train_dataset))
    if evaluation_dataset is None:
        print("evaluation dataset not given. This means we are not using evaluation loss. Will just use training data and evaluation loss")
        evaluation_dataset = combined_train_dataset.shuffle(seed=42).select(range(int(len(combined_train_dataset)/2)))
    print("length of validation data: ", len(evaluation_dataset))
    train_results = train(model, tokenizer, combined_train_dataset, evaluation_dataset, train_epochs, batch_size, max_step, eval_steps, callback=callback)
    return train_results

def extract_data_mixture_and_train_and_evaluate(input_X, evaluation_task : List, model, random_dir, tokenizer, train_datasets, val_datasets, data_domains, mixing_ratio, additional_info, total_number_datapoints, run_name, seed=42, method="random", train_epochs=1, batch_size=8, max_step=-1, eval_steps=100, lora_config=None, fix_training_samples=False):
    
    output_dir = "LLM/BO/"+random_dir+"/"+run_name # store this model here every BO runs, to be evaluated
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    config = lora_config
    model = get_peft_model(model, config)

    # apply tokenization to all data
    # print("tokenizing all data into correct format...")
    
    # sample the correct amount of data from each domain
    # print("iterating through each data domain and sampling the sufficient datapoints")
    # print("mixing ratio: ", mixing_ratio)
    # print("ALL DATA DOMAINS: ", data_domains)
    all_sampled_train_data = []
    all_sampled_val_data = []
    for train_dataset, val_dataset, data_domain, ratio, IF_values in zip(train_datasets, val_datasets, data_domains, mixing_ratio, additional_info):
        
        # print("doing sampling for domain: ", data_domain)
        # print("ratio: ", ratio)
        total_datapt = int(total_number_datapoints * ratio)
        # print("number of datapoints needed (ratio * total): ", total_datapt)
        if total_datapt == 0:
            continue # skip if no data needed
        print("sampling...")
        sampled_train_data = sample(train_dataset, total_datapt, additional_info=IF_values, method=method, data_domain=data_domain, seed=seed)
        print("done sampling training")
        sampled_val_data = sample(val_dataset, total_datapt, additional_info=None, method="random", data_domain=None, seed=seed)
        print("done sampling validation")
        
        sampled_train_data = sampled_train_data.shuffle(seed=seed).map(tokenizing_method[data_domain], fn_kwargs={"tokenizer": tokenizer,
                                                                                   "add_eos_token": add_eos_token,
                                                                                   "train_on_inputs": train_on_inputs,
                                                                                   })
        sampled_val_data = sampled_val_data.shuffle(seed=seed).map(tokenizing_method[data_domain], fn_kwargs={"tokenizer": tokenizer,
                                                                                   "add_eos_token": add_eos_token,
                                                                                   "train_on_inputs": train_on_inputs,
                                                                                   })
        # print("done mapping!")
        
        # drop columns
        
        sampled_train_data = sampled_train_data.select_columns(['input_ids', 'attention_mask', 'labels'])
        sampled_val_data = sampled_val_data.select_columns(['input_ids', 'attention_mask', 'labels'])
        
        all_sampled_train_data.append(sampled_train_data)
        all_sampled_val_data.append(sampled_val_data)
    
    combined_train_dataset = concatenate_datasets(all_sampled_train_data)
    combined_val_dataset = concatenate_datasets(all_sampled_val_data)
    print("length of training data: ", len(combined_train_dataset))
        
    evaluation_time_step = [1,50,100,200,300,500,1000]
    class TimeBasedEvalCallback(transformers.TrainerCallback):
        task_metrics = {
            "commonsense_qa": "acc,none",
            "gsm8k": "exact_match,strict-match",
            "headqa_en": "acc,none",
            "rowan_hellaswag": "acc,none",
            "pubmedqa": "acc,none",
            "sciq": "acc_norm,none",
            "triviaqa": "exact_match,remove_whitespace",
            "truthfulqa_gen": "bleu_acc,none",
            "wikitext": "word_perplexity,none",
            "mmlu": "acc,none",
            "arc_challenge": "acc,none"
            }
        def __init__(self, evaluation_times, tokenizer, task, configuration):
            self.configuration = configuration
            self.evaluation_times = sorted(evaluation_times)  # [30, 100] seconds
            self.tokenizer = tokenizer
            self.task = task # list
            self.start_time = None
            self.evaluated_times = set()  # Track which time points we've already evaluated
            self.performance_history = []  # Store (time, performance) tuples
            self.output_file = "predictor_trials/"+task[0]+".json"
            self.last_check_time = 0  # Track when we last checked
            
            # Initialize the JSON file with empty data
            self.initialize_json_file()
        
        def initialize_json_file(self):
            """Initialize or load existing JSON file"""
            # Check if file exists
            if os.path.exists(self.output_file):
                try:
                    # Load existing data
                    with open(self.output_file, 'r') as f:
                        existing_data = json.load(f)
                    print(f"Loaded existing JSON file: {self.output_file}")
                    
                    # Convert string keys back to tuples if needed
                    converted_data = {}
                    for key, value in existing_data.items():
                        # Convert string representation of tuple back to actual tuple
                        # Example: "(0.5, 0.5, 16)" -> (0.5, 0.5, 16)
                        if key.startswith('(') and key.endswith(')'):
                            # Simple conversion for numeric tuples
                            tuple_key = tuple(float(x) if '.' in x else int(x) for x in key.strip('()').split(','))
                            converted_data[tuple_key] = value
                        else:
                            # Keep as is if it's not a tuple string
                            converted_data[key] = value
                            
                    self.existing_data = converted_data
                except (json.JSONDecodeError, KeyError, ValueError):
                    # If file is corrupted or empty, start fresh
                    print("Existing JSON file is corrupted, starting fresh")
                    self.existing_data = {}
            else:
                # Start with empty data
                self.existing_data = {}
                print(f"Created new JSON file: {self.output_file}")
            
            # Ensure our current configuration exists in the data
            config_key = tuple(self.configuration)
            if config_key not in self.existing_data:
                self.existing_data[config_key] = []
            
            # Write initial state
            self._write_json_file()

        def update_json_file(self):
            """Update the JSON file with current performance history for this configuration"""
            config_key = tuple(self.configuration)
            
            # Update the performance list for this configuration
            performance_list = [(target_time, performance) for target_time, performance, _ in self.performance_history]
            self.existing_data[config_key] = performance_list
            
            # Write updated data to file
            self._write_json_file()
            print(f"Updated JSON file with {len(performance_list)} performance records for config {config_key}")

        def _write_json_file(self):
            """Helper method to write data to JSON file"""
            # Convert tuples to strings for JSON serialization
            serializable_data = {}
            for key, value in self.existing_data.items():
                if isinstance(key, tuple):
                    # Convert tuple to string representation
                    str_key = str(key)
                else:
                    str_key = key
                serializable_data[str_key] = value
            
            with open(self.output_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)


        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()
            print(f"Training started at: {self.start_time}")
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            print("logging")
            """on_log is called more frequently during training"""
            current_time = time.time() - self.start_time
            
            # Only check every 1 second to avoid too frequent checks
            if current_time - self.last_check_time > 1:
                self.last_check_time = current_time
                print(f"Current training time: {current_time:.2f}s")
                
                # Check if we've reached any of our target evaluation times
                for target_time in self.evaluation_times:
                    # Use approximation: if we're within 3 seconds of the target time and haven't evaluated yet
                    if (target_time <= current_time <= target_time + 3) and target_time not in self.evaluated_times:
                        print(f"EVALUATION CALLBACK - Reached {current_time:.1f}s (target: {target_time}s)")
                        
                        time_before_eval = time.time()
                        model = kwargs['model']
                        model.eval()
                        results = evaluate_tasks(self.task, model, self.tokenizer, batch=8, few_shot=3, limit=100)
                        model.train()
                        time_taken_for_eval = time.time() - time_before_eval
                        self.start_time += time_taken_for_eval # Adjust start time to account for eval duration
                        
                        # Store the results
                        performance = self.extract_performance(results)
                        print(f"Evaluation results: {performance}")
                        self.performance_history.append((target_time, performance, state.global_step))
                        
                        # Mark this time as evaluated
                        self.evaluated_times.add(target_time)
                        
                        # Update the JSON file
                        self.update_json_file()
                    
        def extract_performance(self, results):
            # Extract the performance metric you care about from results
            # This depends on your evaluate_tasks function structure
            # Example: return results["results"][self.task]["accuracy"]
            return results["results"][self.task[0]][self.task_metrics[self.task[0]]]  # Modify this based on your actual results structure

                
    class TimerCallback(transformers.TrainerCallback):
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
    callback = []
    callback.append(TimeBasedEvalCallback(evaluation_time_step, tokenizer, evaluation_task, configuration=input_X))
    callback.append(TimerCallback(1000))
    
    train_epochs = 20
    output_model_dir = train(model, tokenizer, combined_train_dataset, combined_val_dataset, output_dir, run_name, train_epochs, batch_size, max_step, eval_steps, lora_config=config, callback=callback)
    
    print("finished training.")
    return output_model_dir
    
def train(model, tokenizer, train_dataset, val_dataset, train_epochs=1, batch_size=8, max_step=-1, eval_steps=1000, callback=[]):
    model.is_parallelizable = False
    model.model_parallel = False
    
    transformers.set_seed(42)
    model.train()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=50,
            num_train_epochs=train_epochs,
            learning_rate=learning_rate,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=25,
            gradient_checkpointing=False,
            optim="adamw_torch",
            save_strategy="no",
            eval_strategy="steps",
            eval_steps=eval_steps,
            max_steps=max_step,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callback
    )

    trainer.train()

    train_results = {"eval_loss": [
    log["eval_loss"]
    for log in trainer.state.log_history
    if "eval_loss" in log
    ]}
    
    return train_results
    
    
    
    breakpoint()
        
def train_deepspeed(model, tokenizer, train_dataset, val_dataset, output_dir, run_name, train_epochs=1, batch_size=8, max_step=-1, eval_steps=1000, lora_config=None, callback=[]):
    # if torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True
    
    transformers.set_seed(42)
    model.train()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=10,
            num_train_epochs=train_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_total_limit=1,
            save_steps=eval_steps,
            max_steps=max_step,
            output_dir=output_dir,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=True,
            group_by_length=group_by_length,
            report_to=None,
            run_name=run_name,
            deepspeed="deepspeed/ds_config.json"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callback
    )

    print("Training model with DeepSpeed...")
    model.print_trainable_parameters()
    trainer.train()

    final_model_path = output_dir + "/" + "final_model_after_training"
    print("Saving final model LoRA weights at:", final_model_path)
    model.save_pretrained(final_model_path)
    
    model.to("cpu")
    del trainer
    with torch.no_grad():
        torch.cuda.empty_cache()
    del model
    gc.collect()

    return final_model_path
    
def evaluate_tasks(tasks : List[str], model, tokenizer, batch=1, few_shot=1, limit=None):

    print("creating HFLM wrapper for model_path")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, dtype=torch.bfloat16, max_length=tokenizer.model_max_length,
                batch_size=batch, trust_remote_code=True)

    print("evaluating on tasks: ", tasks)
    if limit == None:
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            task_manager=lm_eval.tasks.TaskManager(),batch_size=batch,max_batch_size=batch, num_fewshot=few_shot)
    else:
        results = lm_eval.simple_evaluate(
            model=lm,
            limit=limit,
            tasks=tasks,
            task_manager=lm_eval.tasks.TaskManager(),batch_size=batch,max_batch_size=batch, num_fewshot=few_shot)
    return results

def load_data(data_domain):
        # Load the dataset
    print(data_domain)
    if data_domain == "headqa_en":
        data_domain = "headqa"
    if data_domain == "wikitext":
        dataset = datasets.load_dataset(data_domain, "wikitext-2-v1", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "triviaqa":
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "pubmedqa":
        dataset = datasets.load_dataset("bigbio/pubmed_qa", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "truthfulqa_gen":
        dataset = datasets.load_dataset("truthfulqa/truthful_qa", "generation", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["validation"]
        val_dataset = dataset["validation"]
    elif data_domain == "commonsense_qa":
        dataset = datasets.load_dataset("tau/commonsense_qa", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "rowan_hellaswag":
        dataset = datasets.load_dataset("Rowan/hellaswag", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "sciq":
        dataset = datasets.load_dataset("allenai/sciq", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "gsm8k":
        dataset = datasets.load_dataset("openai/gsm8k", "main", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    elif data_domain == "squadv2":
        dataset = datasets.load_dataset("rajpurkar/squad_v2" , cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "headqa":
        dataset = datasets.load_dataset("dvilares/head_qa", "en", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "datologyai_hellaswag":
        dataset = datasets.load_dataset("DatologyAI/hellaswag", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["eval"]
        val_dataset = dataset["eval"]
    elif data_domain == "mmlu":
        dataset = datasets.load_dataset("cais/mmlu", "all", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["test"]
        val_dataset = dataset["validation"]
    elif data_domain == "arc_challenge":
        dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir = "/home/chenzhil/maplecg_nfs/data-mixing/datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    else:
        assert False, "data_domain not valid, pls check"
    return train_dataset, val_dataset