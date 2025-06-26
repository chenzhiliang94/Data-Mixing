from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound, LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.models.transforms.outcome import Standardize
import shutil
import torch
from itertools import product

from helper import get_data_from_mixing_ratio
from image_training import train
from typing import List
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)

lora_alpha = 16
lora_dropout= 0.05
lora_r=16
lora_target_modules = [
    "q_proj",
    "v_proj",
]
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
    
def iterative_loop(data_sources : List[DataLoader], validation_data : DataLoader, method : str, additional_info : List[List[float]], seed, layers_freeze : int, cuda : str, num_epochs=10, iterations=10, data="images", printout=True):
    
    input_X = torch.Tensor((len(data_sources))*[float(1/len(data_sources))]) # initial X
    GP_input = []
    observed_output = []
    for i in range(iterations):
        print("iteration: ", i)
        
        if printout:
            print("mixing data with method: ", method)

        mixed_data = get_data_from_mixing_ratio(data_sources, mixing_ratio=input_X,
                                                additional_info=additional_info,
                                                method=method,
                                                seed=seed,
                                                base_number_of_batches=20) # each agent do some influence function process to get data
        
        if data=="images":
            acc_all, observed_performance, _ = train(mixed_data, validation_data, seed=seed, lr=5e-5, cuda=cuda, num_epochs=num_epochs, num_layer_to_unfreeze=layers_freeze, printout=printout) # observe the performance of this dataset from finetuning
        if printout:
            print("performance after training: ", observed_performance)
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X)
        #current_gp_input.append(current_mixing_parameter)
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=1)
        bounds = torch.stack([torch.zeros(len(current_gp_input)), torch.ones(len(current_gp_input))]) # need to change the bounds for parameters
        A = [1.0] * len(data_sources)
        x = list(range(len(data_sources)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=50,
            #equality_constraints = [(torch.tensor(list(range(len(data_sources)))), torch.tensor([1.0] * len(data_sources)), 1)]
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        input_X = [x if x >= 0.05 else 0 for x in candidate[0]]
        if printout:
            print("proposed parameters for next round by BO:", input_X)
    return GP_input, observed_output, gp

def get_BO_plots(observations):
    BO_to_plot = []
    for x in range(0,len(observations)):
        BO_to_plot.append((max(observations[:x+1])))
    return BO_to_plot

def run_BO(all_loaders, validaton_dataloader, method, additional_info, seed, iterations, num_epochs, cuda, layers_freeze, printout=False):
    print("running BO...")
    X, observations, gp = iterative_loop(all_loaders, validaton_dataloader, cuda=cuda, method=method, additional_info=additional_info, layers_freeze=layers_freeze, seed=seed, num_epochs=num_epochs, iterations=iterations, printout=printout)
    BO_to_plot = get_BO_plots(observations) # BO results
    naive_combine = BO_to_plot[0] # naive mixing result is the first iteration result of BO

    def get_optimal_mixture_from_GP_posterior():
        UCB = UpperConfidenceBound(gp, beta=0.0)
        bounds = torch.stack([torch.zeros(len(all_loaders)), torch.ones(len(all_loaders))]) # need to change the bounds for parameters
        A = [1.0] * len(all_loaders)
        x = list(range(len(all_loaders)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=30,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        return candidate

    def get_best_observation_mixture():
        
        # Find the index in list B that has the highest value
        highest_index = observations.index(max(observations))
        
        # Return the corresponding item in list A
        return X[highest_index]

    
    print("best mixture found in BO iterations is: ", get_best_observation_mixture())
    return BO_to_plot

from LLM.llm import load_data, get_tokenizer_and_model, extract_data_mixture_and_train
from LLM.llm import extract_data_mixture_and_train, evaluate_tasks, load_data, get_tokenizer_and_model

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from tqdm import tqdm

def run_BO_for_LLM(data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, model_id = "LLM/llama_8b_instruct"):
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    #model = prepare_model_for_kbit_training(model)

    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        
        if printout:
            print("mixing data with method: ", sampling_method)

        # sample from each domain and train a model
        path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                        train_datasets=train_datasets, 
                                                        val_datasets=val_datasets, 
                                                        data_domains=data_domains, 
                                                        mixing_ratio=input_X, 
                                                        additional_info=all_influences, # add IF value
                                                        total_number_datapoints=total_data, 
                                                        run_name="BO_run_" +str(i),
                                                        method=sampling_method,
                                                        train_epochs=train_epochs, 
                                                        batch_size=training_batch,
                                                        max_step=max_steps,
                                                        lora_config=lora_config,
                                                        eval_steps=eval_steps)
        # free gpu memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        print("evaluating...")
        lora_path = path_to_final_model #final_model_after_training
        config = PeftConfig.from_pretrained(lora_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
        lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
        
        observed_performance = 0
        tasks = list(evaluation_task.keys())
        lora_model.eval()
        results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch)
        print("deleting lora model after evaluation.")
        shutil.rmtree(lora_path, ignore_errors=True)
        print("results: ", results["results"])
        for task in evaluation_task:
            task_weight, metric = evaluation_task[task]
            print(task_weight)
            print(metric)
            print(results["results"][task][metric])
            perf = results["results"][task][metric]
            if task == "wikitext":
                perf = - perf # we want to maximize the score, so for perplexity we maximize instead
            observed_performance += (perf * task_weight)
        print("current iteration performance: ", observed_performance)
        lora_model.to("cpu")
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X)
        
        #current_gp_input.append(current_mixing_parameter)
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=1)
        bounds = torch.stack([torch.zeros(len(current_gp_input)), torch.ones(len(current_gp_input))]) # need to change the bounds for parameters
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=50,
            #equality_constraints = [(torch.tensor(list(range(len(data_sources)))), torch.tensor([1.0] * len(data_sources)), 1)]
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        input_X = [x if x >= 0.05 else 0 for x in candidate[0]]
        if printout:
            print("proposed parameters for next round by BO:", input_X)
    return GP_input, observed_output, gp

def arrange_lora_config(lora_r, lora_dropout, num_layers_to_apply, five_dim_vector):
    '''
    lora_r: float
    lora_dropout = float 
    num_layers_to_apply = int
    five_dim_vector = List[float]. Five dimension
    '''
    print("arranging lora config with parameters: ", lora_r, lora_dropout, num_layers_to_apply, five_dim_vector)
    # only .mlp layers have up, down, gate proj
    # only .self_attn layers have q, v, k proj
    # ["model.layers.0.self_attn.k_proj"]
    lora_modules_all = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    lora_module_to_tune = [mod for mod, flag in zip(lora_modules_all, five_dim_vector) if flag == 1]
    lora_specific_modules = []
    print(lora_module_to_tune)
    for module in lora_module_to_tune:
        if module == "q_proj" or module == "v_proj" or module == "k_proj":
            for i in range(num_layers_to_apply):
                lora_specific_modules.append("model.layers."+str(i)+".self_attn."+module)
        else:
            for i in range(num_layers_to_apply):
                lora_specific_modules.append("model.layers."+str(i)+".mlp."+module)
    
    # if we choose all 0
    if len(lora_specific_modules) == 0:
        return None

    # lora r is chosen as 0
    if lora_r == 0:
        return None
    config = LoraConfig(
    r=lora_r,
    lora_alpha=16,
    target_modules=lora_specific_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",)

    return config
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

def joint_opt_BO_LLM_only_data(default_rank, default_layer, default_num_layers_to_apply, default_dropout, default_alpha, time_callback, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):
    
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    
    # mixing ratio
    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]

    # mixing ratio bounds
    lower_bound = [0] * (len(data_domains))
    upper_bound = [1] * (len(data_domains))
    
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        print("input_X: ", input_X) 
        
        lora_config = arrange_lora_config(default_rank, default_dropout, default_num_layers_to_apply, default_layer)
        
        # sample from each domain and train a model
        path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                        train_datasets=train_datasets, 
                                                        val_datasets=val_datasets, 
                                                        data_domains=data_domains, 
                                                        mixing_ratio=input_X[:len(data_domains)], 
                                                        additional_info=all_influences, # add IF value
                                                        total_number_datapoints=total_data, 
                                                        run_name="BO_run_" +str(i),
                                                        method=sampling_method,
                                                        train_epochs=train_epochs, 
                                                        batch_size=training_batch,
                                                        max_step=max_steps,
                                                        lora_config=lora_config,
                                                        eval_steps=eval_steps, callback=[time_callback])
        # free gpu memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        print("evaluating...")
        lora_path = path_to_final_model
        config = PeftConfig.from_pretrained(lora_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
        lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
        
        observed_performance = 0
        tasks = list(evaluation_task.keys())
        results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
        print("deleting lora model after evaluation.")
        shutil.rmtree(lora_path, ignore_errors=True)
        for task in evaluation_task:
            task_weight, metric = evaluation_task[task]
            perf = results["results"][task][metric]
            if task == "wikitext":
                perf = - perf # we want to maximize the score, so for perplexity we maximize instead
            observed_performance += (perf * task_weight)
        print("current iteration weighted performance: ", observed_performance)
        lora_model.to("cpu")
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X_between_0_1)
        
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        print("GP past observed values (should be between [0,1]): ", GP_input)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output))
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)] # edit this TODO.
        )
        
        print("proposed candidate before processing:", candidate[0])
        input_X_between_0_1 = list(candidate[0])
        input_X = list(candidate[0])
        
    return GP_input, observed_output, gp

def joint_opt_BO_LLM(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)
    # input_X is the input to our GP:
    # first len(data_domains) are the mixing ratio (0 to 1, constrained to sum to 1)
    # next 1 dimension is the number of layers to apply to (integer)
    # next 5 dimension vector to indicate which layer to apply to (0 or 1)
    # then lora rank (integer)
    # then lora dropout (float)
    
    # for discrete BO
    fixed_features_list =[{len(data_domains)+2:0},{len(data_domains)+2:1},
                          {len(data_domains)+3:0},{len(data_domains)+3:1},
                          {len(data_domains)+4:0},{len(data_domains)+4:1},
                          {len(data_domains)+5:0},{len(data_domains)+5:1},
                          {len(data_domains)+6:0},{len(data_domains)+6:1}]
    
    # mixing ratio
    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]
    # lora number of layers
    input_X.append(int(lora_max_num_layers*0.5))
    input_X_between_0_1.append(0.5)
    # lora which layer to apply to
    input_X = input_X + [1, 1, 1, 1, 1] # 5 dimension vector to indicate apply to all layers as initial input
    input_X_between_0_1 = input_X_between_0_1 + [1, 1, 1, 1, 1]
    # lora rank
    input_X.append(72) # initial rank = 16
    input_X_between_0_1.append(72.0/lora_rank_max)
    # lora dropout
    input_X.append(0.05) # initial dropout=0.05
    input_X_between_0_1.append(0.05)
    # mixing ratio bounds
    lower_bound = [0] * (len(data_domains))
    upper_bound = [1] * (len(data_domains))
    # lora number of layers bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # which layer to apply to bounds
    lower_bound+=[0, 0, 0, 0, 0]
    upper_bound+=[1, 1, 1, 1, 1]
    # lora rank bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # lora dropout bounds
    lower_bound.append(0.0)
    upper_bound.append(0.1)
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        
        lora_config = arrange_lora_config(input_X[-2], input_X[-1], input_X[len(data_domains)], input_X[len(data_domains)+1:len(data_domains)+6])
        
        # sample from each domain and train a model
        path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                        train_datasets=train_datasets, 
                                                        val_datasets=val_datasets, 
                                                        data_domains=data_domains, 
                                                        mixing_ratio=input_X[:len(data_domains)], 
                                                        additional_info=all_influences, # add IF value
                                                        total_number_datapoints=total_data, 
                                                        run_name="BO_run_" +str(i),
                                                        method=sampling_method,
                                                        train_epochs=train_epochs, 
                                                        batch_size=training_batch,
                                                        max_step=max_steps,
                                                        lora_config=lora_config,
                                                        eval_steps=eval_steps, callback=[time_callback])
        # free gpu memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        print("evaluating...")
        lora_path = path_to_final_model
        config = PeftConfig.from_pretrained(lora_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
        lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
        
        observed_performance = 0
        tasks = list(evaluation_task.keys())
        results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
        print("deleting lora model after evaluation.")
        shutil.rmtree(lora_path, ignore_errors=True)
        for task in evaluation_task:
            task_weight, metric = evaluation_task[task]
            perf = results["results"][task][metric]
            if task == "wikitext":
                perf = - perf # we want to maximize the score, so for perplexity we maximize instead
            observed_performance += (perf * task_weight)
        print("current iteration weighted performance: ", observed_performance)
        lora_model.to("cpu")
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X_between_0_1)
        
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        print("GP past observed values (should be between [0,1]): ", GP_input)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output))
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)] # edit this TODO.
        )
        
        def process_values(values, data_domains_len):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            for v in values[:data_domains_len]:
                result.append(0 if v.item() < 0.05 else v)
            
            # Step 2: lora layers
            if len(values) > data_domains_len:
                result.append(round(lora_max_num_layers*values[data_domains_len].item()))
            
            # Step 3: Round the next 5 elements: integer options
            start = data_domains_len + 1
            for v in values[start:start+5]:
                result.append(round(v.item()))
            
            # Step 4: lora rank
            if len(values) > start + 5:
                result.append(round(lora_rank_max * values[start + 5].item()))
            
            # Step 5: drop out; unchanged
            if len(values) > start + 6:
                result.append(values[start + 6].item())
            print("proposed candidate after normalizing:", result)
            return result
        print("proposed candidate before processing:", candidate[0])
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0], len(data_domains))
        
    return GP_input, observed_output, gp

def joint_opt_BO_LLM_fixed_feature_list(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):
    
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)
    # input_X is the input to our GP:
    # first len(data_domains) are the mixing ratio (0 to 1, constrained to sum to 1)
    # next 1 dimension is the number of layers to apply to (integer)
    # next 5 dimension vector to indicate which layer to apply to (0 or 1)
    # then lora rank (integer)
    # then lora dropout (float)
    
    # for discrete BO
    '''
    fixed_features_list (list[dict[int, float]] | None) 
    A list of maps {feature_index: value}.
    The i-th item represents the fixed_feature for the i-th optimization.
    If fixed_features_list is provided, optimize_acqf_mixed is invoked.
    All indices (feature_index) should be non-negative.
    '''

    # All possible combinations of 0 or 1 for 5 dimensions
    combinations = list(product([0, 1], repeat=5))
    d = len(data_domains)
    # Convert each combination into a dictionary
    dict_list = [{i + d + 1: val for i, val in enumerate(combo)} for combo in combinations]
    print("fixed feature list generated:")
    for x in (dict_list):
        print(x)

    fixed_features_list = dict_list
    
    # mixing ratio
    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]
    # lora number of layers
    input_X.append(int(lora_max_num_layers*0.5))
    input_X_between_0_1.append(0.5)
    # lora which layer to apply to
    input_X = input_X + [1, 1, 1, 1, 1] # 5 dimension vector to indicate apply to all layers as initial input
    input_X_between_0_1 = input_X_between_0_1 + [1, 1, 1, 1, 1]
    # lora rank
    input_X.append(72) # initial rank = 16
    input_X_between_0_1.append(72.0/lora_rank_max)
    # lora dropout
    input_X.append(0.05) # initial dropout=0.05
    input_X_between_0_1.append(0.05)
    # mixing ratio bounds
    lower_bound = [0] * (len(data_domains))
    upper_bound = [1] * (len(data_domains))
    # lora number of layers bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # which layer to apply to bounds
    lower_bound+=[0, 0, 0, 0, 0]
    upper_bound+=[1, 1, 1, 1, 1]
    # lora rank bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # lora dropout bounds
    lower_bound.append(0.0)
    upper_bound.append(0.1)
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        
        lora_config = arrange_lora_config(input_X[-2], input_X[-1], input_X[len(data_domains)], input_X[len(data_domains)+1:len(data_domains)+6])
        if lora_config is None:
            observed_performance = 0.1 # very bad performance if we use this
        else: # sample from each domain and train a model
            path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                            train_datasets=train_datasets, 
                                                            val_datasets=val_datasets, 
                                                            data_domains=data_domains, 
                                                            mixing_ratio=input_X[:len(data_domains)], 
                                                            additional_info=all_influences, # add IF value
                                                            total_number_datapoints=total_data, 
                                                            run_name="BO_run_" +str(i),
                                                            method=sampling_method,
                                                            train_epochs=train_epochs, 
                                                            batch_size=training_batch,
                                                            max_step=max_steps,
                                                            lora_config=lora_config,
                                                            eval_steps=eval_steps, callback=[time_callback])
            # free gpu memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            print("evaluating...")
            lora_path = path_to_final_model
            config = PeftConfig.from_pretrained(lora_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
            lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
            
            observed_performance = 0
            tasks = list(evaluation_task.keys())
            results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
            print("deleting lora model after evaluation.")
            shutil.rmtree(lora_path, ignore_errors=True)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = - perf # we want to maximize the score, so for perplexity we maximize instead
                observed_performance += (perf * task_weight)
            print("current iteration weighted performance: ", observed_performance)
            lora_model.to("cpu")
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X_between_0_1)
        
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        print("GP past observed values (should be between [0,1]): ", GP_input)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output))
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf_mixed(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)],
            fixed_features_list = fixed_features_list# edit this TODO.
        )
        
        def process_values(values, data_domains_len):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            for v in values[:data_domains_len]:
                result.append(0 if v.item() < 0.05 else v)
            
            # Step 2: lora layers
            if len(values) > data_domains_len:
                result.append(round(lora_max_num_layers*values[data_domains_len].item()))
            
            # Step 3: Round the next 5 elements: integer options
            start = data_domains_len + 1
            for v in values[start:start+5]:
                result.append(round(v.item()))
            
            # Step 4: lora rank
            if len(values) > start + 5:
                result.append(round(lora_rank_max * values[start + 5].item()))
            
            # Step 5: drop out; unchanged
            if len(values) > start + 6:
                result.append(values[start + 6].item())
            print("proposed candidate after normalizing:", result)
            return result
        print("proposed candidate before processing:", candidate[0])
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0], len(data_domains))
        
    return GP_input, observed_output, gp

def joint_opt_BO_LLM_only_model(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):
    
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)
    # input_X is the input to our GP:
    # first len(data_domains) are the mixing ratio (0 to 1, constrained to sum to 1)
    # next 1 dimension is the number of layers to apply to (integer)
    # next 5 dimension vector to indicate which layer to apply to (0 or 1)
    # then lora rank (integer)
    # then lora dropout (float)
    
    # for discrete BO
    fixed_features_list =[{len(data_domains)+2:0},{len(data_domains)+2:1},
                          {len(data_domains)+3:0},{len(data_domains)+3:1},
                          {len(data_domains)+4:0},{len(data_domains)+4:1},
                          {len(data_domains)+5:0},{len(data_domains)+5:1},
                          {len(data_domains)+6:0},{len(data_domains)+6:1}]
    
    # mixing ratio
    input_X = []
    input_X_between_0_1 = []
    lower_bound = []
    upper_bound = []
    mixing_ratio = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all

    # lora number of layers
    input_X.append(int(lora_max_num_layers*0.5))
    input_X_between_0_1.append(0.5)
    # lora which layer to apply to
    input_X = input_X + [1, 0, 1, 0, 0] # 5 dimension vector to indicate apply to all layers as initial input
    input_X_between_0_1 = input_X_between_0_1 + [1, 1, 0, 0, 0]
    # lora rank
    input_X.append(16) # initial rank = 16
    input_X_between_0_1.append(16.0/lora_rank_max)
    # lora dropout
    input_X.append(0.05) # initial dropout=0.05
    input_X_between_0_1.append(0.05)

    # lora number of layers bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # which layer to apply to bounds
    lower_bound+=[0, 0, 0, 0, 0]
    upper_bound+=[1, 1, 1, 1, 1]
    # lora rank bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # lora dropout bounds
    lower_bound.append(0.0)
    upper_bound.append(0.1)
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        # lora_r, lora_dropout, num_layers_to_apply, five_dim_vector
        lora_config = arrange_lora_config(input_X[6], input_X[7], input_X[0], input_X[1:6])
            
        '''
        0: layers
        1:6: 5 dim
        6: lora rank
        7: lora dropout
        '''
        # lora number of layers 
        # input_X.append(int(lora_max_num_layers*0.5))
        # input_X_between_0_1.append(0.5)
        # # lora which layer to apply to
        # input_X = input_X + [1, 0, 1, 0, 0] # 5 dimension vector to indicate apply to all layers as initial input
        # input_X_between_0_1 = input_X_between_0_1 + [1, 1, 0, 0, 0]
        # # lora rank
        # input_X.append(16) # initial rank = 16
        # input_X_between_0_1.append(16.0/lora_rank_max)
        # # lora dropout
        # input_X.append(0.05) # initial dropout=0.05
        # input_X_between_0_1.append(0.05)
    
        # sample from each domain and train a model
        path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                        train_datasets=train_datasets, 
                                                        val_datasets=val_datasets, 
                                                        data_domains=data_domains, 
                                                        mixing_ratio=mixing_ratio, 
                                                        additional_info=all_influences, # add IF value
                                                        total_number_datapoints=total_data, 
                                                        run_name="BO_run_" +str(i),
                                                        method=sampling_method,
                                                        train_epochs=train_epochs, 
                                                        batch_size=training_batch,
                                                        max_step=max_steps,
                                                        lora_config=lora_config,
                                                        eval_steps=eval_steps, callback=[time_callback])
        # free gpu memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        print("evaluating...")
        lora_path = path_to_final_model
        config = PeftConfig.from_pretrained(lora_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
        lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
        
        observed_performance = 0
        tasks = list(evaluation_task.keys())

        results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)

        print("deleting lora model after evaluation.")
        shutil.rmtree(lora_path, ignore_errors=True)
        for task in evaluation_task:
            task_weight, metric = evaluation_task[task]
            perf = results["results"][task][metric]
            if task == "wikitext":
                perf = - perf # we want to maximize the score, so for perplexity we maximize instead
            observed_performance += (perf * task_weight)
        print("current iteration weighted performance: ", observed_performance)
        lora_model.to("cpu")
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X_between_0_1)
        
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        print("GP past observed values (should be between [0,1]): ", GP_input)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output))
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)] # edit this TODO.
        )
        
        def process_values(values):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            
            # Step 2: lora layers
            result.append(round(lora_max_num_layers*values[0].item()))
            
            # Step 3: Round the next 5 elements: integer options
            for v in values[1:6]:
                result.append(round(v.item()))
            
            # Step 4: lora rank
            result.append(round(lora_rank_max * values[6].item()))
            
            # Step 5: drop out; unchanged
            result.append(values[7].item())
            print("proposed candidate after normalizing:", result)
            return result
        print("proposed candidate before processing:", candidate[0])
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0])
        
    return GP_input, observed_output, gp
