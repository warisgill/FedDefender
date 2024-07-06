import time
import torch
import torch.nn.functional as F
import torchvision
from torch.nn.init import (kaiming_normal_, kaiming_uniform_, normal_,
                           orthogonal_, trunc_normal_, uniform_,
                           xavier_normal_, xavier_uniform_)


class InferenceGuidedInputs:
    def __init__(self, clients2models, shape, randomGenerator, apply_transform, dname=None, k_gen_inputs=10, min_nclients_same_pred=5, time_delta=60):
        self.clients2models = clients2models
        self.min_nclients_same_pred = 3 #min_nclients_same_pred
        # print(f"Same prediction threshold {self.min_nclients_same_pred}")
        self.same_seqs_set = set()
        self.k_gen_inputs = k_gen_inputs
        self.size = 1024
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
        # self.use_gpu = use_gpu
        # self.device = torch.device("cpu")
        self.random_inputs = []
        self.input_shape = shape
        self.time_delta = time_delta    
        self.apply_transform = apply_transform
        self.randomGenerator = None
        func_names = [f.__name__ for f in [uniform_, normal_, xavier_uniform_,
                                           xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_, orthogonal_]]
        if randomGenerator.__name__ in func_names:
            self.randomGenerator = randomGenerator
        else:
            raise Exception(f"Error: {type(randomGenerator)} not supported")

        if dname is not None:
            self.transform = self._getDataSetTransformation(dname)
        
        # if use_gpu:
        #     self.device = torch.device("cuda")

    def _getRandomInput(self):
        img = torch.empty(self.input_shape)
        self.randomGenerator(img)
        if self.apply_transform:
            return self.transform(img)
        return img

    def _simpleRandomInputs(self):
        start = time.time()
        random_inputs = [self._getRandomInput()
                       for _ in range(self.k_gen_inputs)]
        return random_inputs, time.time()-start

    def getInputs(self):
        if len(self.clients2models) <= 10:
            return self._simpleRandomInputs()
        else:
            return self._generateFeedBackRandomInputs1()

    def _predictFun(self, model, input_tensor):
        model.eval()
        logits = model(input_tensor)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        pred = preds.item()
        return pred

    # # feedback loop to create diverse set of inputs
    def _generateFeedBackRandomInputs1(self):
        print("Feedback Random inputs")

        def appendOrNot(input_tensor):
            preds = [self._predictFun(m, input_tensor)
                     for m in self.clients2models.values()]
            for ci1, pred1 in enumerate(preds):
                seq = set()
                seq.add(ci1)
                for ci2, pred2 in enumerate(preds):
                    if ci1 != ci2 and pred1 == pred2:
                        seq.add(ci2)

                s = ",".join(str(p) for p in seq)
                if s not in same_prediciton and len(seq) >= self.min_nclients_same_pred:
                    # print(s)
                    same_prediciton.add(s)
                    random_inputs.append(input_tensor)
                    return

        timeout = 60
        random_inputs = []
        same_prediciton = set()
        start = time.time()
        while len(random_inputs) < self.k_gen_inputs:
            img = self._getRandomInput()
            if self.min_nclients_same_pred > 1:
                appendOrNot(img)
            else:
                random_inputs.append(img)

            if time.time() - start > timeout:
                timeout += 60
                self.min_nclients_same_pred -= 1
                print(
                    f">> Timeout: Number of distinct inputs: {len(random_inputs)}, so decreasing the min_nclients_same_pred to {self.min_nclients_same_pred} and trying again with timiout of {timeout} seconds")
        return random_inputs, time.time()-start

    def _getDataSetTransformation(self, dname):
        if dname in ["CIFAR10", "cifar10"]:
            return torchvision.transforms.Compose([
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dname == "femnist":
            return torchvision.transforms.Compose([
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            raise Exception(f"Dataset {dname} not supported")
