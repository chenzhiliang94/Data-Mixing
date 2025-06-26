import os
from LLM.llm import extract_data_mixture_and_train, evaluate_tasks, load_data, get_tokenizer_and_model
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)
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

def arrange_lora_config(lora_r, num_layers_to_apply, lora_modules_to_tune, lora_dropout=0.01):
    '''
    lora_r: float
    lora_dropout = float 
    num_layers_to_apply = int
    five_dim_vector = List[float]. Five dimension
    '''

    #lora_modules_to_tune = ["q_proj", "v_proj", "k_proj", "up_proj", "down_proj", "gate_proj"]
    lora_specific_modules = []

    for i in range(num_layers_to_apply):
        for module in lora_modules_to_tune:
            if module == "q_proj" or module == "v_proj" or module == "k_proj":
                lora_specific_modules.append("model.layers."+str(i)+".self_attn."+module)
            else:
                lora_specific_modules.append("model.layers."+str(i)+".mlp."+module)
    print("layers to tune: ", lora_specific_modules)
    config = LoraConfig(
    r=lora_r,
    lora_alpha=16,
    target_modules=lora_specific_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",)
    
    '''
    model.layers.0.self_attn
    model.layers.0.self_attn.q_proj
    model.layers.0.self_attn.k_proj
    model.layers.0.self_attn.v_proj
    model.layers.0.self_attn.o_proj
    model.layers.0.self_attn.rotary_emb
    model.layers.0.mlp
    model.layers.0.mlp.gate_proj
    model.layers.0.mlp.up_proj
    model.layers.0.mlp.down_proj
    model.layers.0.mlp.act_fn
    model.layers.0.input_layernorm
    model.layers.0.post_attention_layernorm
    '''

    return config

parser = ArgumentParser()
parser.add_argument("--domain", help="output_dir")
parser.add_argument("--time_intervals", help="output_dir")
parser.add_argument("--start_time", help="output_dir")
parser.add_argument("--end_time", help="output_dir")
parser.add_argument("--few_shot", help="output_dir")
parser.add_argument("--trials", help="trials")
parser.add_argument("--batch", help="trials")
parser.add_argument("--adjustable_module", help="trials")

parser.add_argument("--lora_r_start", help="lora_r_start")
parser.add_argument("--lora_r_end", help="lora_r_end")
parser.add_argument("--lora_r_intervals", help="lora_r_intervals")

args = vars(parser.parse_args())
few_shot = int(args["few_shot"])
batch = int(args["batch"])
trials = int(args["trials"])
domain = args["domain"]
adjustable_module = args["adjustable_module"]
adjustable_module = adjustable_module.split(',')

end_time = int(args["end_time"])
start_time = int(args["start_time"])
time_intervals = int(args["time_intervals"])

lora_r_start, lora_r_intervals, lora_r_end = int(args["lora_r_start"]), int(args["lora_r_intervals"]), int(args["lora_r_end"])

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
for time_limit in range(start_time, end_time, time_intervals):
    time_callback=TimerCallback(time_limit)
    results[time_limit] = {}
    for lora_layers in range(3,4,1):
        results[time_limit][lora_layers] = {}
        for lora_r in range(lora_r_start, lora_r_end+1, 8):
            
            total_data = resources
            config=arrange_lora_config(lora_r=lora_r,  num_layers_to_apply=lora_layers, lora_modules_to_tune=adjustable_module, lora_dropout=0.01)
            # get tokenizer and model

            tokenizer, base_model = get_tokenizer_and_model(model_id = model_id)
            run_name = "test"
            post_training_performance_trials = []
            for _ in range(0,trials):
                path_to_final_model = extract_data_mixture_and_train(model=deepcopy(base_model), random_dir="joint_optimization_"+str(domain)+"".join(adjustable_module), tokenizer=tokenizer, 
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
                result = evaluate_tasks(data_domains, lora_model, tokenizer, evaluation_batch, few_shot, limit=100)
                post_training_performance_trials.append(result['results'][domain])
                print("lora layers: ", lora_layers)
                print("time limit: ", time_limit)
                print("results: ", result['results'][domain])

            results[time_limit][lora_layers][lora_r] = post_training_performance_trials
print("all results:", results)
import json
results["command_line_args"] = args
# Save it as a JSON file
output_name = "".join(adjustable_module)
with open("output_joint_optimization/results_full/" + domain + "_time_" + str(end_time) + "_lorarank_" + str(lora_r_end) + "_modules_" + str(output_name) +".json", "w") as json_file:
    json.dump(results, json_file, indent=4)
        
  