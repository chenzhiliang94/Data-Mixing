from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
import torch

from helper import *

def iterative_loop():
    
    current_scores_to_find = [0.5,0.5,0.5] # assume 3 agents (size of list). Number indicates insruction to agent (e.g., influence score to search, or something else)
    current_mixing_parameter = 0.5 #  another parameter
    GP_input = []
    observed_output = []
    iterations = 10 # number of iterations to run algorithm
    for i in range(iterations):
        
        data_from_agents = get_data_from_agent(current_scores_to_find) # each agent do some influence function process to get data
        mixed_data = mixup(data_from_agents, current_mixing_parameter) # globally mix data into a dataset
        observed_performance = get_performance(mixed_data) # observe the performance of this dataset from finetuning
        
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(current_scores_to_find)
        current_gp_input.append(current_mixing_parameter)
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=0.1)
        bounds = torch.stack([torch.zeros(len(current_gp_input)), torch.ones(len(current_gp_input))]) # need to change the bounds for parameters
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        print("proposed parameters for next round by BO:", candidate)
        
        # next round candidates and repeat the loop
        current_scores_to_find = candidate[0][:3]
        current_mixing_parameter = candidate[0][3:]
        
iterative_loop()
        
        