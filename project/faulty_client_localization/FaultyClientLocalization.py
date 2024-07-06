import gc
import torch
import itertools
from .neuron_activation import getNeuronsActivations

def makeAllSubsetsofSizeN(s, n):
    assert n < len(s)
    l_of_subsets = list(itertools.combinations(s, n))
    l_of_lists = [set(sub) for sub in l_of_subsets]
    return l_of_lists

class FaultyClientLocalization:
    def __init__(self, client2model, generated_inputs, use_gpu) -> None:
        self.generated_inputs = generated_inputs
        self.use_gpu = use_gpu
        self.clients2randominputs_neurons_activations = {}
        self.client2layeracts = {}
        self._updateNeuronCoverage(client2model)
        self.clientids = set(list(client2model.keys()))

        self.all_clients_combinations = makeAllSubsetsofSizeN(self.clientids, len(
                self.clientids)-1)  # resetting for next random iteration

    def _updateNeuronCoverage(self, client2model):
        device = torch.device("cpu")
        if self.use_gpu:
            device = torch.device("cuda")

        for client_id, model in client2model.items():
            model = model.to(device)
            outs = [getNeuronsActivations(model, img.to(device))
                    for img in self.generated_inputs]
            self.clients2randominputs_neurons_activations[client_id] = [
                all_acts for all_acts, _ in outs]
            self.client2layeracts[client_id] = [
                layer_acts for _, layer_acts in outs]

            model = model.to(torch.device("cpu"))
            gc.collect()
            torch.cuda.empty_cache()

    def runFaultLocalization(self, na_t, num_bugs):
        faulty_clients_on_gen_inputs = []
        for i in range(len(self.generated_inputs)):
            potential_faulty_clients = None
            # for the given input i find "num_bugs" number of faulty clients
            for _ in range(num_bugs): 
                benign_clients_ids = self._findNormalClientsSeqV1(i, na_t)
                potential_faulty_clients = self.clientids - benign_clients_ids
                self._updateClientsCombinations(potential_faulty_clients)

            faulty_clients_on_gen_inputs.append(potential_faulty_clients)

            self.all_clients_combinations = makeAllSubsetsofSizeN(self.clientids, len(
                self.clientids)-1)  # resetting for next generated input

        assert len(faulty_clients_on_gen_inputs) == len(self.generated_inputs)
        return faulty_clients_on_gen_inputs

    def _updateClientsCombinations(self, potential_faulty_clients):
        remaining_clients = self.clientids - potential_faulty_clients
        self.all_clients_combinations = makeAllSubsetsofSizeN(
            remaining_clients, len(remaining_clients)-1)

    def _findNormalClientsSeqV1(self, input_id, na_t):

        client2NA = {cid: self.clients2randominputs_neurons_activations[cid][input_id]
                     > na_t for cid in self.clients2randominputs_neurons_activations.keys()}

        # client2NA = [self.clients2randominputs_nc[cid][input_id]
        #              for cid in self.clients2randominputs_nc.keys()]
        clients_ids = self.getClientsIDsWithHighestCommonNeurons(client2NA)
        return clients_ids

    def getClientsIDsWithHighestCommonNeurons(self, clients2neurons2boolact):

        select_neurons = self.torchIntersetion(
            clients2neurons2boolact) == False

        clients_neurons2boolact = {cid: t[select_neurons]
                                   for cid, t in clients2neurons2boolact.items()}

        count_of_common_neurons = [(self.torchIntersetion({cid: clients_neurons2boolact[cid]
                                    for cid in comb}) == True).sum().item() for comb in self.all_clients_combinations]
        highest_number_of_common_neurons = max(count_of_common_neurons)
        val_index = count_of_common_neurons.index(
            highest_number_of_common_neurons)
        val_clients_ids = self.all_clients_combinations[val_index]        
        return val_clients_ids

    def torchIntersetion(self, client2tensors):
        intersct = True
        for k, v in client2tensors.items():
            intersct = intersct*v
            torch.cuda.synchronize()
        return intersct

   