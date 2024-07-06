from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch
from utils.models import train, test

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().detach().clone().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    net = net.cpu()
    params_dict = zip(net.state_dict().keys(), parameters)
    new_state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    # net.load_state_dict(state_dict, strict=True)
    # Assuming 'model' is your PyTorch model with the same architecture
    
    # for name, param in net.named_parameters():
    #     for p in parameters:
    #         param.data = torch.from_numpy(p)

    # new_state_dict = {name: torch.from_numpy(array) for name, array in state_dict.items()}
    net.load_state_dict(new_state_dict, strict=True)



# def set_parameters(net, parameters: List[np.ndarray]):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)


# def set_parameters2(model, parameters):
#     """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if
#     available.
#     """
#     # model.train()
#     model.zero_grad()
#     model = model.cpu()
#     params_dict = zip(model.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     model.load_state_dict(state_dict, strict=True)




class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, device, malicious_clients_ids):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.DEVICE = device
        self.malicious_clients_ids = malicious_clients_ids
        # print(f"malacious clients {malicious_clients_ids}")

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(
            f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        nk_client_data_points = train(self.net, self.trainloader,
                                      epochs=local_epochs, DEVICE=self.DEVICE)  # this is only for elaborating the nk data points in paper, but can be done with valloader as well (orignal flwr tutorial)
        print(f"\n --->client_id {self.cid}, malicious_clients_ids {self.malicious_clients_ids}, nk_client_data_points {nk_client_data_points}")

        # print(f"client type {type(self.cid)}, malacious client type {type(self.malicious_clients_ids[0])}"  )

        if int(self.cid) in self.malicious_clients_ids:
            print(f"----------------------------------> [Client {self.cid}] is malicious")
            nk_client_data_points = 20 * nk_client_data_points
        # else:
        #     print(f"----------------------------------> [Client {self.cid}] is benign")
        #     nk_client_data_points = 1 * nk_client_data_points


        return get_parameters(self.net), nk_client_data_points, {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, DEVICE=self.DEVICE)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}