#!/usr/bin/env python
# coding: utf-8
import fire
import time
import logging
from torch.nn.init import uniform_, normal_, xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_, orthogonal_


import torch
from diskcache import Index


from utils.models import initializeModel


import logging
import time

from pytorch_lightning import seed_everything
from torch.nn.init import kaiming_uniform_ 
from faulty_client_localization.FaultyClientLocalization import FaultyClientLocalization
from faulty_client_localization.InferenceGuidedInputs import InferenceGuidedInputs


logging.basicConfig(filename='example.log', level=logging.ERROR)
logger = logging.getLogger("pytorch_lightning")
seed_everything(786)







def getRoundKeys(fl_key, training_cache):
    r_keys = []
    for k in training_cache.keys():
        if fl_key == k:
            continue
        elif fl_key in k and len(k) > len(fl_key):
            r_keys.append(k)
    return r_keys


def evalauteFedFuzz(participating_clients_ids, malicious_ids, fedfuzz_clients_combs):
    # print(f"Participating Clients: {participating_clients_ids}")
    # print(f"Malicious Clients: {malicious_ids}")
    benign_clients = participating_clients_ids - malicious_ids
    # print(f"benign_clients {benign_clients}")
    # exit()

    true_seq = 0
    for comb in fedfuzz_clients_combs:
        print("comb", comb)
        if comb == benign_clients:
            true_seq += 1
        print(f"Predited client {participating_clients_ids-comb}")

    detection_accuracy = (true_seq/len(fedfuzz_clients_combs)) 
    return detection_accuracy


def getRoundParticipantModels(round_key, config_key, training_cache):

    nn_config = training_cache[config_key]["sim_config"]["nn_config"]

    round2ws = training_cache[round_key]
    gws, c2ws = round2ws["gm_ws"], round2ws["client2ws"]

    global_model = initializeModel(nn_config)
    client2model = {k: initializeModel(nn_config) for k in c2ws.keys()}
    global_model.load_state_dict(gws)

    _ = {k: client2model[k].load_state_dict(c2ws[k]) for k in c2ws.keys()}

    global_model = global_model.cpu().eval()
    client2model = {k: v.cpu().eval()
                    for k, v in client2model.items()}

    return global_model, client2model


# def runFedfuzz(client2models, all_combinations, dname, input_shape, n_fuzz_inputs=10, random_generator=kaiming_normal_, apply_transform=True, nc_thresholds=[0.0], num_bugs=1, use_gpu=True):
#     min_t = -1
#     max_t = 1

#     fuzz_gen = FuzzGeneration(client2models, input_shape,
#                               use_gpu, randomGenerator=random_generator, apply_transform=apply_transform, dname=dname, majority_threshold=5, n_fuzz_inputs=n_fuzz_inputs, min_t=min_t, max_t=max_t)
#     fuzz_inputs, input_gen_time = fuzz_gen.getFuzzInputs()

#     total_time = 0
#     start = time.time()
#     fedfuzz = FedFuzz(client2models, fuzz_inputs,
#                       all_combinations, use_gpu=use_gpu)
#     total_time += (time.time() - start)
#     results = None
#     start = time.time()
#     # if num_bugs ==1:
#     fedfuzz_results = fedfuzz.runFedFuzz(0.0)
#     # else:
#     # fedfuzz_results = [fedfuzz.runMultiFedFuzz(t, num_bugs=num_bugs) for t in nc_thresholds]
#     fedfuzz_time = total_time + ((time.time()-start)/len(nc_thresholds))

#     return fedfuzz_results,  input_gen_time, fedfuzz_time


def evaluateFaultLocalization(predicted_faulty_clients_on_each_input, true_faulty_clients):
    true_faulty_clients = set(true_faulty_clients)
    detection_acc = 0
    for pred_faulty_clients in predicted_faulty_clients_on_each_input:
        print(f"+++ Faulty Clients {pred_faulty_clients}")
        correct_localize_faults = len(
            true_faulty_clients.intersection(pred_faulty_clients))
        acc = (correct_localize_faults/len(true_faulty_clients))*100
        detection_acc += acc
    fault_localization_acc = detection_acc / \
        len(predicted_faulty_clients_on_each_input)
    return fault_localization_acc


def runFaultyClientLocalization(client2models, num_bugs, random_generator=kaiming_uniform_, apply_transform=True, k_gen_inputs=10, na_threshold=0.003, use_gpu=True):
    print(">  Running FaultyClientLocalization ..")
    input_shape = [1,3,32,32]
    generate_inputs = InferenceGuidedInputs(client2models, input_shape, randomGenerator=random_generator, apply_transform=apply_transform,
                                            dname="cifar10", min_nclients_same_pred=5, k_gen_inputs=k_gen_inputs)
    selected_inputs, input_gen_time = generate_inputs.getInputs()

    start = time.time()
    faultyclientlocalization = FaultyClientLocalization(
        client2models, selected_inputs, use_gpu=use_gpu)

    potential_faulty_clients_for_each_input = faultyclientlocalization.runFaultLocalization(
        na_threshold, num_bugs=num_bugs)
    fault_localization_time = time.time()-start
    return potential_faulty_clients_for_each_input, input_gen_time, fault_localization_time




def main():

    fl_config_key = '[Pattern 20x20 FL_Configuration = arch-densenet121, dataset-cifar10, totalclients-30, groups--1, total_rounds-1, percentage_of_randomly_selected_clients-1, client_epochs-5, client_lr-0.001, batch_size-512, data_distribution-iid, strategy-FedAvg, malacious_clients-[0, 1, 3, 5, 7]]'

    cache_name: str =  'c_waris'
    storage_dir: str = 'storage_fed_debug'

    training_cache = Index(storage_dir + cache_name)
    results_cache = Index(storage_dir + "cache_defense_results")

    sim_config = training_cache[fl_config_key]["sim_config"]
    

    rounds_keys = getRoundKeys(fl_config_key, training_cache)
    round2def_storage = {}    

    for round_key in rounds_keys:
        global_model, client2model = getRoundParticipantModels(
            round_key=round_key, config_key=fl_config_key, training_cache=training_cache)
        
        fautly_clients_ids = sim_config['MALICIOUS_CLIENTS_IDS']
        
        potential_faulty_clients, _, _ =  runFaultyClientLocalization(client2models=client2model, num_bugs=len(fautly_clients_ids))
        acc = evaluateFaultLocalization(potential_faulty_clients, fautly_clients_ids)
        print(acc)

        temp_dict = {}
        temp_dict["clients"] = list(client2model.keys())
        temp_dict["Defense Acc"] = -1
        print(f"info {temp_dict}")
        round2def_storage[round_key.replace(
            fl_config_key, "").replace("round", "")] = temp_dict

    results_cache[fl_config_key] = {
        "round2def_storage": round2def_storage,  "training_cache_config": training_cache[fl_config_key]}

    return fl_config_key


if __name__ == "__main__":
    experiment_key = fire.Fire(main)
    logging.info(
        f" Done  {experiment_key}")
