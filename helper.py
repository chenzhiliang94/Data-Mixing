from typing import List
import random

# use some influence score to retrieve data
def get_data(score : float):
    
    # DATA INFLUENCE SCORE RETRIEVAL GOES HERE
    data = 0.0 # change this, definitely not float.
    
    return data

# give agent some instructions/score req/something else and ask them to give dataset
def get_data_from_agent(scores : List[float]):

    returned_data = []
    for score in scores:
        returned_data.append(get_data(score))
    
    return returned_data

def mixup(agent_data : List[float], mixing_parameter : float): # change data format of agent data (should be a list of datasets, not list of float)
    
    # DATA MIXING CODE GOES HERE
    data = 0.0 # data set format
    
    return data

def get_performance(data):
    # FINETUNE AND FIND PERFORMANCE; CODE GOES HERE
    performance = random.random()
    
    return performance