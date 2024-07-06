import argparse
import copy
import logging
import math
import random
import sys
import time
from datetime import datetime

# from utilss import ImageClassifer
import pytorch_lightning as pl
from .FedFuzz import FedFuzz, makeAllSubsetsofSizeN
from .FuzzGeneration import FuzzGeneration
from torch.nn.init import (kaiming_normal_, kaiming_uniform_, normal_,
                           orthogonal_, trunc_normal_, uniform_,
                           xavier_normal_, xavier_uniform_)

from pytorch_lightning import seed_everything

seed_everything(786)

def evalauteFedFuzz(participating_clients_ids, malicious_ids, fedfuzz_clients_combs):
    benign_clients = participating_clients_ids - malicious_ids
    true_seq = 0
    for comb in fedfuzz_clients_combs:
        print("comb", comb)
        if comb == benign_clients:
            true_seq += 1
        print(f"Predited client {participating_clients_ids-comb}")
    detection_accuracy = (true_seq/len(fedfuzz_clients_combs)) 
    return detection_accuracy

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
    start = time.time()
    # if num_bugs ==1:
    fedfuzz_results = fedfuzz.runFedFuzz(0.0)
    # else:
    # fedfuzz_results = [fedfuzz.runMultiFedFuzz(t, num_bugs=num_bugs) for t in nc_thresholds]
    fedfuzz_time = total_time + ((time.time()-start)/len(nc_thresholds))
    return fedfuzz_results,  input_gen_time, fedfuzz_time


def getPotentialMaliciousClients(fedfuzz_clients_combs, participating_clients_ids):
    # malicious_ids = set()
    client2count = {cid: 0 for cid in participating_clients_ids}
    for comb in fedfuzz_clients_combs:
        # malicious_ids = malicious_ids.union(participating_clients_ids-comb)
        for cid in participating_clients_ids-comb:
            client2count[cid] += 1

    # get the client id with max count
    # get key with max value
    max_count = max(client2count.values())
    total = sum(client2count.values()) 

    malicious2freq = {cid:count/total for cid, count in client2count.items() if count>= max_count} # now it will work multiple malicious clients but can be improved
    
    return malicious2freq


def fedFuzzDefense(client2model, input_shape, dname:str, courrupt_clients_ids:set[str]):
    fedfuzz_acc = -1
    all_combinations = makeAllSubsetsofSizeN(
            set(list(client2model.keys())), len(client2model) - 1)
    participating_clients_ids = set(list(client2model.keys()))

    fedfuzz_combs, _, _ = runFedfuzz(client2model, all_combinations=all_combinations,
                                         dname=dname, input_shape=input_shape)
    
    first_mal_client:str = ""
    for cid in courrupt_clients_ids:
        first_mal_client = cid
        break

    if first_mal_client != "-1":
        fedfuzz_acc = evalauteFedFuzz(
                participating_clients_ids, courrupt_clients_ids, fedfuzz_clients_combs=fedfuzz_combs) 
    
    malacious2confidence = getPotentialMaliciousClients(fedfuzz_combs, participating_clients_ids)   
    return malacious2confidence, fedfuzz_acc
