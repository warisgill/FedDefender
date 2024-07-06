from collections import OrderedDict
import flwr as fl
import numpy as np
import torch
import os

from flwr.common import (EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
                         ndarrays_to_parameters, parameters_to_ndarrays)

from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

# from flwr.common.logger import log

from defenses.fedfuzz_defense import fedFuzzDefense

from .models import  initializeModel


#   ndarrays_to_parameters,
#     parameters_to_ndarrays,

def getStateDictFromParameters(nn_config, parameters):
    ndarr = fl.common.parameters_to_ndarrays(parameters)
    temp_net = initializeModel(nn_config)
    params_dict = zip(temp_net.state_dict().keys(), ndarr)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    temp_net.load_state_dict(state_dict, strict=True)
    return temp_net.state_dict()




class FedAvg_Custom_Strategy(fl.server.strategy.FedAvg):
    """SaveModelStrategy implements a custom strategy for Flower."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        provdict = {}
        provdict["client2ws"] = {client.cid: getStateDictFromParameters(self.nn_config,
            fit_res.parameters) for client, fit_res in results}
        print(
            f"\n\n **** Custom Metrics from clients: {[f'client : {c.cid}, metrics: {resp.metrics}||' for c, resp in results] }")
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)
        round_key = f"{self.exp_key}-round:{server_round}"
        print(round_key)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            gm_net = initializeModel(self.nn_config)
            params_dict = zip(gm_net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v)
                                     for k, v in params_dict})
            gm_net.load_state_dict(state_dict, strict=True)
            provdict["gm_ws"] = gm_net.state_dict()
            self.cache[round_key] = provdict
        
        return aggregated_parameters, aggregated_metrics
    
    def setCacheAndExpKey(self, cache, exp_key, sim_config, input_shape, malicious_clients):
        self.cache = cache
        self.exp_key = exp_key
        self.data_config = sim_config["data_distribution_config"]
        self.nn_config = sim_config["nn_config"] 
        self.input_shape = input_shape
        self.malicious_clients = set([f"{cid}" for cid in malicious_clients])


    

class Gradient_Clipping_Strategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:


        #clipping

        clipUpdates(results=results)


        provdict = {}
        provdict["client2ws"] = {client.cid: getStateDictFromParameters(self.nn_config,
            fit_res.parameters) for client, fit_res in results}
        print(
            f"\n\n **** Custom Metrics from clients: {[f'client : {c.cid}, metrics: {resp.metrics}||' for c, resp in results] }")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        round_key = f"{self.exp_key}-round:{server_round}"
        print(round_key)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            gm_net = initializeModel(self.nn_config)
            params_dict = zip(gm_net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v)
                                     for k, v in params_dict})
            gm_net.load_state_dict(state_dict, strict=True)
            provdict["gm_ws"] = gm_net.state_dict()
            self.cache[round_key] = provdict
        
        return aggregated_parameters, aggregated_metrics
    
    def setCacheAndExpKey(self, cache, exp_key, sim_config, input_shape, malicious_clients):
        self.cache = cache
        self.exp_key = exp_key
        self.data_config = sim_config["data_distribution_config"]
        self.nn_config = sim_config["nn_config"] 
        self.input_shape = input_shape
        self.malicious_clients = set([f"{cid}" for cid in malicious_clients])



def _lambda_clipUpdate(ndarrays, clip : float):
    l2_update = np.linalg.norm(np.concatenate([t.flatten() for t in ndarrays]).flatten(), ord=2)
    print(f"l2_update before: {l2_update}")
    ndarrays = [t / max(1, l2_update/clip) for t in ndarrays]  
    l2_update = np.linalg.norm(np.concatenate([t.flatten() for t in ndarrays]).flatten(), ord=2)
    print(f"l2_update after: {l2_update}")
    return ndarrays

def clipUpdates(results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]):    
   for i in range(len(results)):
        ndarrays =  _lambda_clipUpdate(parameters_to_ndarrays(results[i][1].parameters), 10.0) # 3, 5, 10  
        results[i][1].parameters = ndarrays_to_parameters(ndarrays) 


class FedFuzz_Defense_Strategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        provdict = {}
        provdict["client2ws"] = {client.cid:getStateDictFromParameters(self.nn_config,
            fit_res.parameters) for client, fit_res in results}
        
        malacious_clients2conf, acc =  self._fedFuzzDefense(provdict["client2ws"])
        self._updateResults(malacious_clients2conf, results)
        
        
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)
        round_key = f"{self.exp_key}-round:{server_round}"
        # print(round_key)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            gm_net = initializeModel(self.nn_config)
            params_dict = zip(gm_net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v)
                                     for k, v in params_dict})
            gm_net.load_state_dict(state_dict, strict=True)
            provdict["gm_ws"] = gm_net.state_dict()
            self.cache[round_key] = provdict
        
        return aggregated_parameters, aggregated_metrics
    
    def _fedFuzzDefense(self, client2model_ws):
        client2model = {cid: initializeModel(self.nn_config) for cid in client2model_ws.keys()}
        for cid, model in client2model.items():
            model.load_state_dict(client2model_ws[cid], strict=True)
        
        # FedFuzz Defense
        potential_malclient2_confidence, feddfuzz_acc = fedFuzzDefense(client2model=client2model, input_shape=self.input_shape, dname=self.data_config["DATASET_NAME"], courrupt_clients_ids=self.malicious_clients)
        return potential_malclient2_confidence, feddfuzz_acc
    
    def _updateResults(self, malacious_clients2confidence:dict[str:int],results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]):
        # fuzz_inputs = 10
        min_nk = min([r[1].num_examples for r in results])
        for i in range(len(results)):
            cid =  results[i][0].cid 
            if cid in malacious_clients2confidence:
                before = results[i][1].num_examples
                if malacious_clients2confidence[cid] > 0.4:
                    results[i][1].num_examples = 0
                else:
                    results[i][1].num_examples =  int(min_nk * malacious_clients2confidence[cid])
                print(f">> Server Defense Result: Client {cid} is malicious, confidence {malacious_clients2confidence[cid]}, num_examples before: {before}, after: {results[i][1].num_examples}")
    
    def setCacheAndExpKey(self, cache, exp_key, sim_config, input_shape, malicious_clients):
        self.cache = cache
        self.exp_key = exp_key
        self.data_config = sim_config["data_distribution_config"]
        self.nn_config = sim_config["nn_config"] 
        self.input_shape = input_shape
        self.malicious_clients = set([f"{cid}" for cid in malicious_clients])





# def clip_updates(self, agent_updates_dict):
#     for update in agent_updates_dict.values():
#         l2_update = torch.norm(update, p=2) 
#         update.div_(max(1, l2_update/self.args.clip))
#     return







