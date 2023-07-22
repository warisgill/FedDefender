import torch
import gc
import sys
import itertools
from .main_neuron_coverage import getAllLayers, getNeuronCoverage

def makeAllSubsetsofSizeN(s, n):
    assert n < len(s)
    l_of_subsets = list(itertools.combinations(s, n))
    l_of_lists = [set(sub) for sub in l_of_subsets]
    return l_of_lists

class FedFuzz:
    def __init__(self, client2model, fuzz_inputs, combinations, use_gpu) -> None:
        self.fuzz_inputs = fuzz_inputs
        # self.clients_models = models
        self.all_combinations = combinations
        self.use_gpu = use_gpu
        self.clients2fuzzinputs_neurons_activations = {}
        self.client2layeracts = {}
        # self.num_layers = len(getAllLayers(client2model[0]))
        # print(f"Num of layers {self.num_layers}")
        self._updateNeuronCoverage(client2model)
        self.clientids = set([c for c in client2model.keys()])

    def _updateNeuronCoverage(self, client2model):
        device = torch.device("cpu")
        if self.use_gpu:
            device = torch.device("cuda")

        for client_id, model in client2model.items():
            model = model.to(device)
            outs = [getNeuronCoverage(model, img.to(device), device)
                    for img in self.fuzz_inputs]
            self.clients2fuzzinputs_neurons_activations[client_id] = [
                all_acts for all_acts, _ in outs]
            self.client2layeracts[client_id] = [
                layer_acts for _, layer_acts in outs]

            model = model.to(torch.device("cpu"))
            gc.collect()
            torch.cuda.empty_cache()

    def runFedFuzz(self, nc_t):
        all_fedfuzz_seqs = []
        for i in range(len(self.fuzz_inputs)):
            # seq = self._findNormalClientsSeqV2LayerBased(i, nc_t)
            seq = self._findNormalClientsSeqV1(i, nc_t)
            all_fedfuzz_seqs.append(seq)
        return all_fedfuzz_seqs

    def runMultiFedFuzz(self, nc_t, num_bugs):
        all_fedfuzz_seqs = []
        self.all_combinations =  makeAllSubsetsofSizeN(self.clientids, len(self.clientids)-1) # resetting for next fuzz iteration
        for i in range(len(self.fuzz_inputs)):
            courrpt_clients = None
            for bug_1 in range(num_bugs):
                vlaid_clients_ids = self._findNormalClientsSeqV1(i, nc_t)
                # all_fedfuzz_seqs.append(seq)
                courrpt_clients = self.clientids - vlaid_clients_ids
                # print(f"{bug_1} corrupt client ids {courrpt_clients}")
                self._updateSettings(courrpt_clients)
            
            
            all_fedfuzz_seqs.append(self.clientids - courrpt_clients)
            self.all_combinations =  makeAllSubsetsofSizeN(self.clientids, len(self.clientids)-1) # resetting for next fuzz iteration
            
        return all_fedfuzz_seqs


    def _updateSettings(self, courrpt_clients):
        after_cleints = self.clientids  - courrpt_clients
        self.all_combinations = makeAllSubsetsofSizeN(after_cleints, len(after_cleints)-1)
    
    
    
    
    def _findNormalClientsSeqV1(self, input_id, nc_t):

        client2NC = {cid: self.clients2fuzzinputs_neurons_activations[cid][input_id]
                     > nc_t for cid in self.clients2fuzzinputs_neurons_activations.keys()}

        # client2NC = [self.clients2fuzzinputs_nc[cid][input_id]
        #              for cid in self.clients2fuzzinputs_nc.keys()]
        clients_ids = self.getClientsIDsWithHighestCommonNeurons(client2NC)
        # clients_ids_seq = ",".join(str(c) for c in clients_ids)
        # print(f"clients ids seq {clients_ids_seq}")
        return clients_ids

    def getClientsIDsWithHighestCommonNeurons(self, clients2neurons2boolact):

        select_neurons = self.torchIntersetion(
            clients2neurons2boolact) == False

        clients_neurons2boolact = {cid: t[select_neurons]
                                   for cid, t in clients2neurons2boolact.items()}

        count_of_common_neurons = [(self.torchIntersetion({cid: clients_neurons2boolact[cid]
                                    for cid in comb}) == True).sum().item() for comb in self.all_combinations]
        highest_number_of_common_neurons = max(count_of_common_neurons)
        val_index = count_of_common_neurons.index(
            highest_number_of_common_neurons)
        val_parties_ids = self.all_combinations[val_index]


        return val_parties_ids

    def torchIntersetion(self, client2tensors):
        intersct = True
        for k, v in client2tensors.items():
            intersct = intersct*v
            torch.cuda.synchronize()
        return intersct


