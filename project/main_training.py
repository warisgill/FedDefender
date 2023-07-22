#!/usr/bin/env python

import fire
import os
from diskcache import Index
from utils.fl_simulation import Simulation


def main(storage_dir, cache_name,  num_clients, num_rounds, groups, batch_size, dname, percentage_of_randomly_selected_clients, client_epochs, strategy, malacious_clients=[-1], ray_storage="/home/gulzar/ray_storage/ray", client_lr=0.001, architecture="simplecnn",  distribution_type="nonIID"):
    # fixed variables
    print(f"stratgey = {strategy}")
    GRAY_SCALE_DATASETS = ["mnist", "fashionmnist"]
    os.system(f"rm -r {ray_storage}")

    cache = Index(storage_dir + cache_name)
    # cache.clear()
    # exit()

    exp_key = f"[Pattern 20x20 FL_Configuration = arch-{architecture}, dataset-{dname}, totalclients-{num_clients}, groups-{groups}, total_rounds-{num_rounds}, percentage_of_randomly_selected_clients-{percentage_of_randomly_selected_clients}, client_epochs-{client_epochs}, client_lr-{client_lr}, batch_size-{batch_size}, data_distribution-{distribution_type}, strategy-{strategy}, malacious_clients-{malacious_clients}]"
    channels = 1 if dname in GRAY_SCALE_DATASETS else 3

    # if exp_key in cache:
    #     temp_dict =  cache[exp_key]
    #     if 'complete' in temp_dict:
    #         if temp_dict['complete']:
    #             print(f"Experiment already completed: {exp_key}")
    #             return 


    # neural network config
    nn_config = {}
    nn_config["architecture"] = architecture
    nn_config["channels"] = channels

    # data distribution config
    data_distribution_config = {}
    data_distribution_config["DATASET_NAME"] = dname
    data_distribution_config["NUM_CLIENTS"] = num_clients
    data_distribution_config["NUM_GROUPS"] = groups
    data_distribution_config["BATCH_SIZE"] = batch_size
    data_distribution_config["DATA_DISTRIBUTION"] = distribution_type
    data_distribution_config["STORAGE_DIR"] = storage_dir

    # complete simulation config
    sim_config = {}
    sim_config["MALICIOUS_CLIENTS_IDS"] = malacious_clients
    sim_config["RAY_STORAGE"] = ray_storage
    sim_config["STORAGE_DIR"] = storage_dir
    sim_config["CACHE_NAME"] = cache_name
    sim_config["exp_main_key"] = exp_key
    sim_config["NUM_CLIENTS"] = num_clients
    sim_config["NUM_ROUNDS"] = num_rounds
    sim_config["percentage_of_randomly_selected_clients"] = percentage_of_randomly_selected_clients
    sim_config["stratgey"] = strategy
    sim_config["client_epochs"] = client_epochs
    sim_config["data_distribution_config"] = data_distribution_config
    sim_config["nn_config"] = nn_config

    sim = Simulation(sim_config, cache)
    sim.run()

    temp_dict =  cache[exp_key]
    temp_dict["complete"] =True
    cache[exp_key] = temp_dict
    print(f"Simulation Complete for: {exp_key}")



if __name__ == "__main__":
    fire.Fire(main)
