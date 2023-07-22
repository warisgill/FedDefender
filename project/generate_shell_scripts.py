import os
import shlex
from diskcache import Index

cache_name = "cache_backdoor2"
storage_dir = './.backdoor_storage/'

def getCompletedTrainingKeys():
    keys = []
    cache = Index(storage_dir + cache_name)
    for k in cache.keys():
        if "-round:" not in k:
            d = cache[k]            
            if 'complete' in d.keys():
                keys.append(k)
    return keys

def trainingScripts():
   strategies = ["FedAvg+FedFuzz_Defense_Strategy", "FedAvg+Gradient_Clipping_Strategy"] # "FedAvg"
   batch_size = 32 
   num_rounds = 15
   client_epochs = 15
   malacious_clients_lists  = [[-1], [0]] # -1 means no malacious clients, 0 means client 0 is malacious
   m = 1
   groups = -1
   with open('_run_training.sh', 'w') as f:
    # for groups in [2,5]:
    for dname in ["mnist"]: # , "fashionmnist" 
        for strategy in strategies:
            for malacious_clients in malacious_clients_lists:
                for num_clients in [20]: 
                    cmd =  f"python main_training.py --storage_dir {storage_dir} --cache_name {cache_name}  --num_clients {num_clients} --num_rounds {num_rounds} --groups {groups} --batch_size {batch_size} --dname {dname} --percentage_of_randomly_selected_clients {m} --client_epochs {client_epochs} --strategy {strategy} --malacious_clients {malacious_clients}"
                    f.write(cmd)
                    f.write("\n")

def evaluateScript():
    keys = []
    keys = getCompletedTrainingKeys()    
    with open('_run_evaluate.sh', 'w') as f:
        for exp_key in keys:
            s = f"python evaluate.py --fl_config_key {shlex.quote(exp_key)} --cache_name {cache_name} --storage_dir {storage_dir}" 
            f.write(s)
            f.write("\n")



evaluateScript()
trainingScripts()