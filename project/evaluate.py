#!/usr/bin/env python
# coding: utf-8
import fire
import time
import argparse
import copy
import logging
import math
import random
import sys
from torch.nn.init import uniform_, normal_, xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_, orthogonal_
from datetime import datetime

# from utilss import ImageClassifer
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from captum.attr import (InternalInfluence, LayerActivation, LayerConductance,
                         LayerDeepLift, LayerDeepLiftShap,
                         LayerFeatureAblation, LayerGradCam, LayerGradientShap,
                         LayerGradientXActivation, LayerIntegratedGradients,
                         LayerLRP)
from diskcache import Index
# from dotmap import DotMap
from scipy.stats import entropy
from sklearn.metrics import f1_score
from torch.nn import functional as F

# from temp_defense.FL_Provenance import FL_Provenance, getAllLayers
from utils.data_distribution import getTrainTestDatasets
from utils.models import initializeModel, test

from defenses.FedFuzz import FedFuzz, makeAllSubsetsofSizeN
from defenses.FuzzGeneration import FuzzGeneration
from utils.fl_simulation import getBackDoorPatterGrey
from utils.data_distribution import AttackBackdoor


pl.utilities.seed.seed_everything(786)

# logging setup
logging.basicConfig(filename=f'defense.log', filemode='a',
                    level=logging.DEBUG, format='%(levelname)s: %(message)s')  # only to file
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter('%(levelname)s: %(message)s'))
logging.getLogger().addHandler(stream_handler)  # to terminal as well


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


def runFedfuzz(client2models, all_combinations, dname, input_shape, n_fuzz_inputs=10, random_generator=kaiming_normal_, apply_transform=True, nc_thresholds=[0.0], num_bugs=1, use_gpu=True):
    min_t = -1
    max_t = 1

    fuzz_gen = FuzzGeneration(client2models, input_shape,
                              use_gpu, randomGenerator=random_generator, apply_transform=apply_transform, dname=dname, majority_threshold=5, n_fuzz_inputs=n_fuzz_inputs, min_t=min_t, max_t=max_t)
    fuzz_inputs, input_gen_time = fuzz_gen.getFuzzInputs()

    total_time = 0
    start = time.time()
    fedfuzz = FedFuzz(client2models, fuzz_inputs,
                      all_combinations, use_gpu=use_gpu)
    total_time += (time.time() - start)
    results = None
    start = time.time()
    # if num_bugs ==1:
    fedfuzz_results = fedfuzz.runFedFuzz(0.0)
    # else:
    # fedfuzz_results = [fedfuzz.runMultiFedFuzz(t, num_bugs=num_bugs) for t in nc_thresholds]
    fedfuzz_time = total_time + ((time.time()-start)/len(nc_thresholds))

    return fedfuzz_results,  input_gen_time, fedfuzz_time


def main(fl_config_key: str, cache_name: str, storage_dir: str, attr_tech=LayerGradientXActivation):

    training_cache = Index(storage_dir + cache_name)
    results_cache = Index(storage_dir + "cache_defense_results")

    # results_cache.clear()
    # exit()

    # if fl_config_key in results_cache.keys():
    #     logging.info(f"Already evaluated {fl_config_key}")
    #     return

    sim_config = training_cache[fl_config_key]["sim_config"]
    _, test_data = getTrainTestDatasets(
        sim_config["data_distribution_config"]["DATASET_NAME"], storage_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    rounds_keys = getRoundKeys(fl_config_key, training_cache)
    # logging.debug(f"rounds_keys {rounds_keys}")

    round2def_storage = {}    
    backdor_dataset = AttackBackdoor(dataset=copy.deepcopy(test_data), class_ids_to_poison=[
                                     0, 1, 2, 3, 4, 5, 6, 7, 8], attack_pattern=getBackDoorPatterGrey(test_data[0][0].squeeze().shape), backdoor_target_class_id=9)

    for round_key in rounds_keys:
        global_model, client2model = getRoundParticipantModels(
            round_key=round_key, config_key=fl_config_key, training_cache=training_cache)

        global_model.to(device)
        _ = [m.to(device) for m in client2model.values()]

        loss, acc = test(global_model, torch.utils.data.DataLoader(
            test_data, batch_size=2048, shuffle=True, num_workers=8), DEVICE=device)

        _, attack_acc = test(global_model, torch.utils.data.DataLoader(
            backdor_dataset, batch_size=2048, shuffle=True, num_workers=8), DEVICE=device)

        temp_dict = {}
        temp_dict["clients"] = list(client2model.keys())
        temp_dict["Attack Success Rate"] = attack_acc
        temp_dict["Defense Acc"] = -1
        temp_dict["test_acc"] = acc
        temp_dict["test_loss"] = loss
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
