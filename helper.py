from typing import List
import random
from torch.utils.data import DataLoader
from data_loader.loader import sample_from


# give agent some instructions/score req/something else and ask them to give dataset
def get_data_from_mixing_ratio(data_sources : List[DataLoader], scores : List[float], additional_info, base_number_of_batches=10) -> DataLoader:
    resulting_dataloader = sample_from(data_sources, scores, base_number_of_batches=base_number_of_batches)
    return resulting_dataloader

# def get_data_from_influence_function(data_sources : List[DataLoader], scores : List[float], base_number_of_batches=10) -> DataLoader:
#     resulting_dataloader = None
    
#     return resulting_dataloader

def mixup(agent_data : List[float], mixing_parameter : float): # change data format of agent data (should be a list of datasets, not list of float)
    
    # DATA MIXING CODE GOES HERE
    data = 0.0 # data set format
    
    return data

def get_performance(data):
    # FINETUNE AND FIND PERFORMANCE; CODE GOES HERE
    performance = random.random()
    
    return performance