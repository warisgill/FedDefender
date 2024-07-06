from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl
import torch
from torch.utils.data import DataLoader

from .data_distribution import (AttackBackdoor, NoisyDataset, iid_split)
from .flwr_client import FlowerClient, get_parameters, set_parameters
from .custom_agg_stratgies import FedAvg_Custom_Strategy, Gradient_Clipping_Strategy, FedFuzz_Defense_Strategy 
from .models import initializeModel, test

 
class Simulation:
    def __init__(self, config, cache):        
        self.malicious_clients_ids = config["MALICIOUS_CLIENTS_IDS"]
        self.ray_storage = config["RAY_STORAGE"]
        self.CACHE = cache
        self.strategy_name = config["stratgey"]
        self.strategy = None

        self.sim_config = config
        self.DEVICE = torch.device("cuda")
        self.nn_config = config["nn_config"]
        self.percentage_of_randomly_selected_clients = config[
            "percentage_of_randomly_selected_clients"]
        self.NUM_CLIENTS = self.sim_config["NUM_CLIENTS"]
        self.NUM_ROUNDS = self.sim_config["NUM_ROUNDS"]
        self.exp_main_key = self.sim_config["exp_main_key"]
        train_dsets, val_dset, server_test_dset, self.client2class = loadIID_Distribution(
            self.sim_config["data_distribution_config"])

        for cid in range(len(train_dsets)):
            if cid in self.malicious_clients_ids:
                print(f"Injecting backdoor to client {cid}")
                s =  train_dsets[cid][0][0].squeeze().shape
                backdor_dataset = AttackBackdoor(dataset=train_dsets[cid], class_ids_to_poison=[0,1,2,3,4,5,6,7,8], attack_pattern=getBackDoorPatterGrey(s), backdoor_target_class_id=9)
                # backdor_dataset = NoisyDataset(dataset=train_dsets[cid], num_classes=10, noise_rate=1)
                train_dsets[cid] = backdor_dataset
                # for i in range(len(train_dsets[cid])):
                #     print(f"cid {cid}  class {train_dsets[cid][i][1]}")

        self.trainloaders, self.valloaders, self.server_testloader = [], [], DataLoader(server_test_dset, batch_size=self.sim_config["data_distribution_config"]["BATCH_SIZE"])
        
        for cid in range(len(train_dsets)):
            self.trainloaders.append(DataLoader(train_dsets[cid], batch_size=self.sim_config["data_distribution_config"]["BATCH_SIZE"], shuffle=True))
            self.valloaders.append(DataLoader(val_dset[cid], batch_size=self.sim_config["data_distribution_config"]["BATCH_SIZE"]))
        
        
        self.client_epochs = self.sim_config["client_epochs"]
        print(f"Data per cleitns {[len(dl) for dl in self.trainloaders]}")
        self.CACHE[self.exp_main_key] = {
            "client2class": self.client2class, "sim_config": self.sim_config}
        self.input_shape = server_test_dset[0][0].shape
        
        if self.strategy_name == "FedAvg":
            self._setStrategy(Agg_Strategy=FedAvg_Custom_Strategy) 
        elif self.strategy_name == "FedAvg+Gradient_Clipping_Strategy":
            self._setStrategy(Agg_Strategy=Gradient_Clipping_Strategy)
        elif self.strategy_name == "FedAvg+FedFuzz_Defense_Strategy":
            self._setStrategy(Agg_Strategy=FedFuzz_Defense_Strategy)
        else:
            raise ValueError("Invalid strategy name")
        

    def _setStrategy(self, Agg_Strategy):
        initial_net = initializeModel(self.nn_config)
        self.strategy = Agg_Strategy(
            fraction_fit=self.percentage_of_randomly_selected_clients,
            fraction_evaluate=0.0,
            accept_failures=False,
            min_fit_clients=self.NUM_CLIENTS,  # Minimum number of clients to fit <--------------------------------------
            min_evaluate_clients=0,
            min_available_clients=self.NUM_CLIENTS,
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_parameters(initial_net)),
            evaluate_fn=self._evaluateGlobalModel,
            on_fit_config_fn=self._getFit_Config,  # Pass the fit_config function
        )
        self.strategy.setCacheAndExpKey(
            self.CACHE, self.exp_main_key, self.sim_config, self.input_shape, self.malicious_clients_ids)

    def _getFit_Config(self, server_round: int):
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": self.client_epochs,  #
        }
        return config

    def _evaluateGlobalModel(self,
                             server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
                             ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # The `evaluate` function will be by Flower called after every round
        net = initializeModel(self.nn_config)
        valloader = self.server_testloader
        # Update model with the latest parameters
        set_parameters(net, parameters)
        loss, accuracy = test(net, valloader, DEVICE=self.DEVICE)
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        return loss, {"accuracy": accuracy}

    def _getClient(self, cid) -> FlowerClient:
        # if int(cid) == 0:
        #     print("\n\n\n ========****  Client 0 is training  ***********  \n\n\n")
        net = initializeModel(self.nn_config).to(self.DEVICE)
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader, self.DEVICE, malicious_clients_ids=self.malicious_clients_ids).to_client()

    def run(self):
        ray_init_args = {"num_cpus": 8, "num_gpus":1}
        
        fl.simulation.start_simulation(
            ray_init_args= ray_init_args,
            client_fn=self._getClient,
            num_clients=self.NUM_CLIENTS,
            config=fl.server.ServerConfig(
                num_rounds=self.NUM_ROUNDS),  # Just three rounds
            strategy=self.strategy,
             client_resources={"num_gpus": 0.33, "num_cpus": 1}
        )



def getBackDoorPatterGrey(shape):
    print(f"shape is {shape}")
    pattern = torch.zeros(shape)
    # pattern[22:,22:] = 255
    pattern[20:,20:] = 255
    return pattern
    

def loadIID_Distribution(config):
    num_clients = config["NUM_CLIENTS"]
    storage_dir = config["STORAGE_DIR"]
    batch_size = config["BATCH_SIZE"]
    dname = config["DATASET_NAME"]
    return iid_split(dname=dname, num_clients=num_clients, storage_dir=storage_dir, batch_size=batch_size)