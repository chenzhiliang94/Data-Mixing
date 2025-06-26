
from data_loader.loader import get_cat_dog, get_fashion_mnist, sample_from
from influence import load_training_influence, load_val_influence
from image_training import train
from BO import iterative_loop, get_BO_plots
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
from tabulate import tabulate
from copy import deepcopy
import csv


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser()
parser.add_argument("--contaminate", help="to contaminate training data?", type=int, default=0)
parser.add_argument("--epochs", help="epochs", default=50)
parser.add_argument("--trials", help="trials", default=5)
parser.add_argument("--cuda", help="trials", default=0)

args = vars(parser.parse_args())
print("command-line args: ", args)
to_contaminate= bool(args["contaminate"])
if not to_contaminate:
    influence_path="influence/"
    print("getting influence from: ", influence_path)
else:
    influence_path="influence/contaminated/"
    print("getting influence from: ", influence_path)

epochs=int(args["epochs"])
trials=int(args["trials"])
cuda=int(args["cuda"])
cuda="cuda:"+str(cuda)

seed = 0

# load relevant data
print("getting fashion mnist data")
# training data sources
mnist_train_loader_1, sandals_val_loader = get_fashion_mnist([5,0], batch_size=16, seed = 0, portion_training=0.95) # sandals
mnist_train_loader_2, sneakers_val_loader = get_fashion_mnist([7,1], batch_size=16, seed = 0, portion_training=0.95) # sneakers
mnist_train_loader_4, bag_val_loader = get_fashion_mnist([8,4], batch_size=16, seed = 0, portion_training=0.95) # bag

# evaluation
mnist_train_loader_5, shirt_val_loader = get_fashion_mnist([6,4], batch_size=16, portion_training=0.95, seed = 0) # shirt
mnist_train_loader_3, boots_val_loader = get_fashion_mnist([9,2], batch_size=16, seed = 0, portion_training=0.95) # boots

print("getting cat dog data")
cat_dog_train_loader, cat_dog_val_loader = get_cat_dog(batch_size=16, seed = 0, portion_training=0.95) # not really useful cat and dog data (for classifying fashion mnist)

sandals_loader = mnist_train_loader_1
sneakers_loader = mnist_train_loader_2
bag_loader = mnist_train_loader_4

boots_loader=mnist_train_loader_3
shirt_loader=mnist_train_loader_5

to_contaminate=True
influence_path="influence/contaminated/"
smaller_shirt_loader = sample_from([shirt_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_cd_loader = sample_from([cat_dog_train_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_sandal_loader = sample_from([sandals_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_sneaker_loader = sample_from([sneakers_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_bag_loader = sample_from([bag_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_boot_loader = sample_from([boots_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)

# def run_trials(train_loader, validation_loader, cuda):
#     acc_obs = []
#     for x in range(5):
#         all_acc,final_acc,_ = train(train_loader, validation_loader, seed=x, num_epochs=50, printout=False, lr=5e-5, cuda=cuda, num_layer_to_unfreeze=1)
#         acc_obs.append(max(all_acc))
#     print("avg acc across trials: ", np.mean(acc_obs))
#     print("std dev: ", np.std(acc_obs))
#     return np.mean(acc_obs)

# domains = {"model_bag":(smaller_bag_loader, bag_val_loader),
#            "model_boots":(smaller_boot_loader, boots_val_loader),
#            "model_cd":(smaller_cd_loader, cat_dog_val_loader),
#            "model_sandal":(smaller_sandal_loader, sandals_val_loader),
#            "model_shirt":(smaller_shirt_loader, shirt_val_loader),
#            "model_sneaker":(smaller_sneaker_loader, sneakers_val_loader)}

# pairwise_domains = list(itertools.combinations(domains.keys(), 2))
# print("all pairwise domains: ", pairwise_domains)
# cuda=("cuda:5")
# all_results = []
# for pair in pairwise_domains:
#     result = []
#     print("domains: ", pair)
#     result.append(pair)
#     domain_A = pair[0]
#     domain_B = pair[1]
    
#     domain_A_influence = load_training_influence(influence_path, domain_A, "cg")
#     domain_B_influence = load_training_influence(influence_path, domain_B, "cg")
    
#     domain_A_train_loader=domains[domain_A][0]
#     domain_B_train_loader=domains[domain_B][0]
#     # hard code mixing ratio
#     validation_loader = sample_from([domains[domain_A][1], domains[domain_B][1]], seed=seed, mixing_ratio=[0.7, 0.3], method="random_sample", base_number_of_batches=20, batch_size=16, shuffle=True, contaminate=False)

#     mixing_ratio = [0.5, 0.5]
#     sampling_methods = [
#        "random_sample", # randomly sample
#         "highest_influence", # take top K influence
#         "lowest_influence", # take bottom K influence 
#         "influence_sample", # sample based on influence distribution
#         "reverse_influence_sample", # sample based on reverse influence distribution 
#         "remove_harmful_then_uniform", # remove bottom 10% influence, then sample from the rest uniformly
#         "remove_harmful_then_follow_IF", # remove bottom 10% influence, then sample from the rest based on influence distribution
#         "remove_tail_ends_then_uniform",
#     ]
    
#     for sample_method in sampling_methods:
        
#         # random sampling 7:3, todo: replace with non training data
#         additional_info = deepcopy([domain_A_influence, domain_B_influence])
#         random_sampling_training_loader = sample_from([domains[domain_A][0], domains[domain_B][0]], seed=None, mixing_ratio=mixing_ratio, method=sample_method, additional_info=additional_info, base_number_of_batches=30, batch_size=16, shuffle=True, contaminate=False)

#         # random
#         print(sample_method)
#         result.append(run_trials(random_sampling_training_loader, validation_loader, cuda=cuda))
#     all_results.append(result)
    
# all_results.insert(0, ["data"]+sampling_methods)
# print(tabulate(all_results))



### Non contaminated

sampling_methods = [
    "random_sample", # randomly sample
    "highest_influence", # take top K influence
    "lowest_influence", # take bottom K influence 
    "influence_sample", # sample based on influence distribution
    "reverse_influence_sample", # sample based on reverse influence distribution 
    "remove_harmful_then_uniform", # remove bottom 10% influence, then sample from the rest uniformly
    "remove_harmful_then_follow_IF", # remove bottom 10% influence, then sample from the rest based on influence distribution
    "remove_tail_ends_then_uniform",
]
    

smaller_shirt_loader = sample_from([shirt_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_cd_loader = sample_from([cat_dog_train_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_sandal_loader = sample_from([sandals_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_sneaker_loader = sample_from([sneakers_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_bag_loader = sample_from([bag_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_boot_loader = sample_from([boots_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)


def run_trials(train_loader, validation_loader, cuda):
    acc_obs = []
    for x in range(trials):
        all_acc,final_acc,_ = train(train_loader, validation_loader, seed=x, num_epochs=epochs, printout=False, lr=5e-5, cuda=cuda, num_layer_to_unfreeze=1)
        acc_obs.append(max(all_acc))
    print("avg acc across trials: ", np.mean(acc_obs))
    print("std dev: ", np.std(acc_obs))
    return np.mean(acc_obs)

domains = {"model_bag":(smaller_bag_loader, bag_val_loader),
           "model_boots":(smaller_boot_loader, boots_val_loader),
           "model_cd":(smaller_cd_loader, cat_dog_val_loader),
           "model_sandal":(smaller_sandal_loader, sandals_val_loader),
           "model_shirt":(smaller_shirt_loader, shirt_val_loader),
           "model_sneaker":(smaller_sneaker_loader, sneakers_val_loader)}

pairwise_domains = list(itertools.combinations(domains.keys(), 2))
print("all pairwise domains: ", pairwise_domains)

all_results = []

# hard code mixing ratio
validation_loader = sample_from([cat_dog_val_loader, boots_val_loader, smaller_sneaker_loader], seed=seed, mixing_ratio=[0.2, 0.4, 0.4], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=False)

for pair in pairwise_domains:
    result = []
    print("domains: ", pair)
    result.append(pair)
    domain_A = pair[0]
    domain_B = pair[1]
    
    domain_A_influence = load_training_influence(influence_path, domain_A, "cg") # val influence is always on clean data, but training data could be contaminated (depending on command arg)
    domain_B_influence = load_training_influence(influence_path, domain_B, "cg")
    
    domain_A_train_loader=domains[domain_A][0]
    domain_B_train_loader=domains[domain_B][0]
    
    # # hard code mixing ratio
    # validation_loader = sample_from([domains[domain_A][1], domains[domain_B][1]], seed=seed, mixing_ratio=[0.7, 0.3], method="random_sample", base_number_of_batches=20, batch_size=16, shuffle=True, contaminate=False)

    mixing_ratio = [0.5, 0.5]
    
    for sample_method in sampling_methods:
        
        # random sampling 7:3, todo: replace with non training data
        additional_info = deepcopy([domain_A_influence, domain_B_influence])
        random_sampling_training_loader = sample_from([domains[domain_A][0], domains[domain_B][0]], seed=None, mixing_ratio=mixing_ratio, method=sample_method, additional_info=additional_info, base_number_of_batches=30, batch_size=16, shuffle=True, contaminate=False)

        # random
        print(sample_method)
        result.append(run_trials(random_sampling_training_loader, validation_loader, cuda=cuda))
    all_results.append(result)
    
all_results.insert(0, ["data"]+sampling_methods)
print(tabulate(all_results))


if to_contaminate:
    result = "contaminated.csv"
else:
    result = "clean.csv"
with open("result/"+result, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(all_results)