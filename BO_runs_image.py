
from data_loader.loader import get_cat_dog, get_fashion_mnist, sample_from
from influence import load_training_influence, load_val_influence
from image_training import train
from BO import iterative_loop, get_BO_plots, run_BO
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
from tabulate import tabulate
from copy import deepcopy
import csv
from itertools import combinations
import json
from collections import defaultdict


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser()
parser.add_argument("--contaminate", help="to contaminate training data?", type=int, default=0)
parser.add_argument("--epochs", help="epochs", default=50)
parser.add_argument("--trials", help="trials", default=5)
parser.add_argument("--cuda", help="trials", default=0)
parser.add_argument("--in_dist", help="in_dist", type=int)
parser.add_argument("--num_eval", help="num_eval")
parser.add_argument("--out_dir", help="out_dir")

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
in_dist=bool(args["in_dist"])
N=int(args["num_eval"])
out_dir = args["out_dir"]

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

smaller_shirt_loader = sample_from([shirt_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_cd_loader = sample_from([cat_dog_train_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_sandal_loader = sample_from([sandals_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_sneaker_loader = sample_from([sneakers_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_bag_loader = sample_from([bag_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)
smaller_boot_loader = sample_from([boots_loader], seed=seed, mixing_ratio=[1.0], method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate=to_contaminate)

inf_cd=load_val_influence(influence_path, "model_cd", "cg")
inf_boot = load_val_influence(influence_path, "model_boots", "cg")
inf_sneakers=load_val_influence(influence_path, "model_sneaker", "cg")
inf_shirt=load_val_influence(influence_path, "model_shirt", "cg")
inf_sandal=load_val_influence(influence_path, "model_sandal", "cg")
inf_bag=load_val_influence(influence_path, "model_bag", "cg")

domains = {"model_bag":(smaller_bag_loader, bag_val_loader, inf_bag),
           "model_boots":(smaller_boot_loader, boots_val_loader, inf_boot),
           "model_cd":(smaller_cd_loader, cat_dog_val_loader, inf_cd),
           "model_sandal":(smaller_sandal_loader, sandals_val_loader, inf_sandal),
           "model_shirt":(smaller_shirt_loader, shirt_val_loader, inf_shirt),
           "model_sneaker":(smaller_sneaker_loader, sneakers_val_loader, inf_sneakers)}

sampling_methods = [
    "random_sample", # randomly sample
    #"highest_influence", # take top K influence
    #"lowest_influence", # take bottom K influence 
    "influence_sample", # sample based on influence distribution
    #"reverse_influence_sample", # sample based on reverse influence distribution 
    "remove_harmful_then_uniform", # remove bottom 10% influence, then sample from the rest uniformly
    #"remove_harmful_then_follow_IF", # remove bottom 10% influence, then sample from the rest based on influence distribution
    #"remove_tail_ends_then_uniform",
]

# test_domain = ["model_boots", "model_shirt", "model_sneaker"]

# Generate all combinations of size N
test_domains = list(combinations(domains.keys(), N))

result_dir = out_dir


for idx, test_domain in enumerate(test_domains):
    print("test domain: ", str(test_domain))
    test_ratio = [1/(len(test_domain))] * len(test_domain)
    validation_loader = sample_from([domains[loader][1] for loader in test_domain], seed=idx, mixing_ratio=test_ratio, method="random_sample", base_number_of_batches=50, batch_size=16, shuffle=True, contaminate= True)

    result = defaultdict(list)
    iterations = 30
    BO_seed = seed
    if in_dist: # in dist
        train_domain = list(domains.keys())
    else: # ood
        train_domain = list(domains.keys())
        train_domain = [x for x in train_domain if x not in test_domain]
    print("training domain: ", train_domain)

    for trial in range(trials):
        for method in sampling_methods:
            print("sampling method: ", method)
            BO_to_plot = run_BO([domains[loader][0] for loader in train_domain], 
                validation_loader, additional_info=[domains[loader][2] for loader in train_domain],
                layers_freeze=2,
                cuda=cuda,
                method=method, 
                seed=seed, 
                iterations=iterations, 
                num_epochs=epochs, 
                printout=False)
            plt.plot(range(len(BO_to_plot)), BO_to_plot, alpha=0.6, label=method)
            result[method].append(BO_to_plot)
            if method == "random_sample":
                naive_combine = BO_to_plot[0]
                plt.axhline(naive_combine, linestyle="--", c="red", label="sample from each data source equally")
            plt.xlabel("BO iterations")
            plt.ylabel("accuracy on evaluation task")
            plt.title("test data: "+str(test_domain))
        plt.legend()
        plt.savefig(result_dir+str(idx)+".png")
        plt.clf()
        with open(result_dir + str(idx) + ".json", 'w') as f:
            json.dump(result, f)