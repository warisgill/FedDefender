import sys
# sys.path.append("../")

from torch.nn.init import uniform_, normal_, xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_, orthogonal_

from .main_neuron_coverage import getNeuronCoverage, getAllLayers, Scale
import os
# from utils.dl_models import ImageClassifer, initialize_model
from joblib import Parallel, delayed
import torchvision
from diskcache import Index
import gc
import itertools
import time
import torch
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
import random

from pytorch_lightning import seed_everything


seed_everything(786)

global Hooks_Storage
Hooks_Storage = []


def getInputAndOutputofLayer(self, input_t, output_t):
    global Hooks_Storage
    try:
        # if the input is a tuple, we have to get the first element
        input_t = input_t[0]
        input_t = input_t.detach()
    except:
        pass

    # assert len(
    #     input_t) == 1, f"Hook, {self.__class__.__name__} Expected 1 input, got {len(input_t)}"
    
    input_t = input_t.clone()
    output_t = output_t.detach().clone()
    # print(f"input_hape {input_t.shape}")

    Hooks_Storage.append((input_t, output_t))


def insertHooks(layers):
    all_hooks = []
    for layer in layers:
        h = layer.register_forward_hook(getInputAndOutputofLayer)
        all_hooks.append(h)
    return all_hooks




class FuzzGeneration:
    def __init__(self, clients2models, shape, use_gpu, randomGenerator, apply_transform, dname=None, n_fuzz_inputs=10, majority_threshold=5, time_delta=60, min_t=-1, max_t=1):
        self.clients2models = clients2models
        self.majority_threshold = majority_threshold
        print(f"Majority Threshold {self.majority_threshold}")
        self.same_seqs_set = set()
        self.n_fuzz_inputs = n_fuzz_inputs
        self.size = 1024
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
        self.use_gpu = use_gpu
        self.device = torch.device("cpu")
        self.fuzz_inputs = []
        self.input_shape = shape
        self.time_delta = time_delta
        self.min_t = min_t
        self.max_t = max_t
        self.apply_transform = apply_transform
        self.randomGenerator = None
        func_names = [f.__name__ for f in [uniform_, normal_, xavier_uniform_,
                                           xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_, orthogonal_]]
        if randomGenerator.__name__ in func_names:
            self.randomGenerator = randomGenerator
        else:
            raise Exception(f"Error: {type(randomGenerator)} not supported")

        # print(
        #     f" \n\n ({self.randomGenerator.__name__}) func, transform = {self.apply_transform}. Majority threshold: {self.majority_threshold}")

        if dname is not None:
            self.transform = self._getDataSetTransformation(dname)
            # print("Orignal data transform.")

        if use_gpu:
            self.device = torch.device("cuda")

    def _getRandomInput(self):
        img = torch.empty(self.input_shape)
        self.randomGenerator(img)
        if self.apply_transform:
            return self.transform(img)
        return img

    def _simpleFuzzInputs(self):
        print("Sime Fuzz inputs")
        start = time.time()
        fuzz_inputs = [self._getRandomInput()
                       for _ in range(self.n_fuzz_inputs)]
        return fuzz_inputs, time.time()-start

    def getFuzzInputs(self):
        return self._simpleFuzzInputs()
        # return self._generateFeedBackFuzzInputs1()

    def _predictFun(self, model, input_tensor):
        # model = model.to(self.device)
        # input_tensor  = input_tensor.to(self.device)
        logits = model(input_tensor)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        pred = preds.item()

        # device = torch.device("cpu")
        # model = model.to(device)
        # input_tensor = input_tensor.to(device)
        return pred
        
    # # feedback loop to create diverse set of inputs
    def _generateFeedBackFuzzInputs1(self):
        print("Feedback Fuzz inputs")
        def appendOrNot(input_tensor):
            preds = [self._predictFun(m, input_tensor) for m in self.clients2models.values()]
            for ci1, pred1 in enumerate(preds):
                seq = set()
                seq.add(ci1)
                for ci2, pred2 in enumerate(preds):
                    if ci1 != ci2 and pred1 == pred2:
                        seq.add(ci2)

                s = ",".join(str(p) for p in seq)
                if s not in same_prediciton and len(seq) >= self.majority_threshold:
                    # print(s)
                    same_prediciton.add(s)
                    fuzz_inputs.append(input_tensor)
                    return

        timeout = 60
        fuzz_inputs = [] 
        same_prediciton = set()
        start = time.time()
        while len(fuzz_inputs) < self.n_fuzz_inputs:
            img = self._getRandomInput()
            if self.majority_threshold > 1:
                appendOrNot(img)
            else:
                fuzz_inputs.append(img)
                
            if time.time() - start > timeout:
                timeout += 60
                self.majority_threshold -= 1
                print(
                    f">> Timeout: Number of distinct inputs: {len(fuzz_inputs)}, so decreasing the majority_threshold to {self.majority_threshold} and trying again with timiout of {timeout} seconds")
        return fuzz_inputs, time.time()-start



    def _getDataSetTransformation(self, dname):
        if dname in ["CIFAR10", "cifar10"]:
            return torchvision.transforms.Compose([
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dname in ["femnist", "mnist"]:
            return torchvision.transforms.Compose([
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif dname  == "fashionmnist":
            return torchvision.transforms.Compose([
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            raise Exception(f"Dataset {dname} not supported")
