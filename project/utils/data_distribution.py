# from dotmap import DotMap
import torch
import torchvision
from torch.utils.data import DataLoader

import random

from torch.utils.data import DataLoader, random_split


# import pytorch_lightning as pl
import logging



def getClient2Class(clients_trains_datasets):
    cli2class = []  # {0:[0,1,2,3,4], 1:[5,6,7,8,9]}
    for cli in range(len(clients_trains_datasets)):
        unique_labels = list(set(t[1] for t in clients_trains_datasets[cli]))
        cli2class.append(unique_labels)
    return cli2class

def getTrainTestDatasets(dname, storage_dir):
    dname = dname.lower()
    resize_trasnform = torchvision.transforms.Resize((32, 32))

    if dname == "mnist":
        traindata = torchvision.datasets.MNIST(storage_dir, train=True, download=True,
                                               transform=torchvision.transforms.Compose([resize_trasnform,
                                                                                         torchvision.transforms.ToTensor(),
                                                                                         torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

        testdata = torchvision.datasets.MNIST(storage_dir, train=False,
                                              transform=torchvision.transforms.Compose([resize_trasnform,
                                                                                        torchvision.transforms.ToTensor(),
                                                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        return traindata, testdata

    elif dname == "fashionmnist":

        traindata = torchvision.datasets.FashionMNIST(storage_dir, train=True, download=True,
                                                      transform=torchvision.transforms.Compose([resize_trasnform,
                                                                                                torchvision.transforms.ToTensor(),
                                                                                                torchvision.transforms.Normalize((0.5,), (0.5,))]))

        testdata = torchvision.datasets.FashionMNIST(storage_dir, train=False,
                                                     transform=torchvision.transforms.Compose([resize_trasnform,
                                                                                               torchvision.transforms.ToTensor(),
                                                                                               torchvision.transforms.Normalize((0.5,), (0.5,))]))
        return traindata, testdata

    elif dname == "cifar10":
        train_transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomCrop(32, padding=4),
            resize_trasnform,
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transforms = torchvision.transforms.Compose([
            resize_trasnform,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=storage_dir, train=True, download=True, transform=train_transforms)

        val_dataset = torchvision.datasets.CIFAR10(
            root=storage_dir, train=False, download=True, transform=test_transforms)

        return train_dataset, val_dataset

    else:
        raise ValueError("Dataset name not recognized")






def getLabelToDataset(data_set):
    unique_labels = set([d[1] for d in data_set])
    labels2indexes = {label: [] for label in unique_labels}
    _ = [labels2indexes[data_set[i][1]].append(
        i) for i in range(len(data_set))]
    label2data = {l: torch.utils.data.Subset(
        data_set, labels2indexes[l]) for l in unique_labels}
    return label2data


class FLDataDistribution:
    def __init__(self, data_set, dist_type, num_clients):
        self.data_set = data_set
        self.dist_type = dist_type
        self.num_clients = num_clients



    def _nonIIDSplittingClassBased(self):
        label2data = getLabelToDataset(self.data_set)
        logging.info("label2data ",[(l,len(d)) for l, d in label2data.items()] )
        

        assert self.num_clients <= len(label2data), f"clients{self.num_clients} are more than {len(label2data)} classes"
        num_classes_per_party = len(label2data)//self.num_clients
        remainder = len(label2data) % self.num_clients

        client2labels = {}
        l = 0
        for p in range(self.num_clients):
            client2labels[p] = []
            for _ in range(num_classes_per_party):
                client2labels[p].append(label2data[l])
                l += 1

        if remainder > 0:
            for _ in range(remainder):
                client2labels[self.num_clients-1].append(label2data[l])
                l += 1

        for p in range(self.num_clients):
            client2labels[p] = torch.utils.data.ConcatDataset(client2labels[p])
            

        clients_datasets = [client2labels[i] for i in range(self.num_clients)]
        return clients_datasets

    def _splitN(self, dataset, n):
        parts = [len(dataset)//n for _ in range(n)]
        parts[0] += len(dataset) % n
        subsets = torch.utils.data.random_split(dataset, parts)
        return subsets  # [SubsetToDataset(subset) for subset in subsets]

    def _iidSplittingBalancedClassesAndQuantity(self):
        label2data = getLabelToDataset(self.data_set)
        splitted_labels_datasets = [self._splitN(
            ldataset, self.num_clients) for ldataset in label2data.values()]
        clients_datasets = [torch.utils.data.ConcatDataset(
            [label_data_splits[cli] for label_data_splits in splitted_labels_datasets])
            for cli in range(self.num_clients)]
        return clients_datasets

    def getDataSetPerClient(self):
        clients_datasets = None
        # if self.dist_type == "nonIID":
        #     clients_datasets = self._nonIIDSplittingClassBased()
        if self.dist_type == "IID":
            clients_datasets = self._iidSplittingBalancedClassesAndQuantity()
        else:
            raise ValueError("Unknown distribution type")

        clients_classes = getClient2Class(clients_datasets)
        return clients_datasets, clients_classes


def nonIID(dname, num_groups, num_clients, batch_size, storage_dir):
    def lambdaFun(dset, clients_ids):
        fl_dist = FLDataDistribution(
        data_set=dset, dist_type="IID", num_clients=len(clients_ids))
        dsplits, client_classes  =   fl_dist.getDataSetPerClient()

        client2trainloader = {}
        client2valloader = {}
        for i in range(len(clients_ids)):
            cid = clients_ids[i]
            ds = dsplits[i] 
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]

            ds_train, ds_val = torch.utils.data.random_split(ds, lengths, torch.Generator().manual_seed(42))
            client2trainloader[cid]  = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            client2valloader[cid]  = DataLoader(ds_val, batch_size=batch_size)

        client2class = {clients_ids[i] : client_classes[i] for i in range(len(clients_ids))}
        return client2trainloader, client2valloader, client2class

    traindata, testdata = getTrainTestDatasets(dname, storage_dir)
    l2dataset = getLabelToDataset(traindata)

    logging.info(f"--Label2Count {[(l, len(l2dataset[l])) for l in l2dataset]}")

    group2labels = {g: [] for g in range(num_groups)}
    group2client_members ={g:[] for g in range(num_groups)}    
    
    g = 0
    i = 0
    for l in range(len(l2dataset)):
        group2labels[g].append(l)
        i += 1
        if i == len(l2dataset) // num_groups: 
            g += 1
            i = 0

    g = 0
    i = 0
    for cid in range(num_clients):
        group2client_members[g].append(cid)
        i += 1
        if i == num_clients//num_groups:
            g += 1
            i = 0
    

    logging.info(f"group2labels {group2labels}")
    logging.info(f"group2client_members {group2client_members}")
    
    

    client2trainDL = {}
    client2valDL = {}
    client2class = {}

    for g in range(num_groups):
        g_dataset =  torch.utils.data.ConcatDataset([l2dataset[l] for l in  group2labels[g]])
        temp_client2trainDL, temp_client2valDL, client2class_others = lambdaFun(g_dataset, group2client_members[g])


        client2trainDL = {**client2trainDL, **temp_client2trainDL}
        client2valDL = {**client2valDL, **temp_client2valDL}
        client2class = {**client2class, **client2class_others}

      
    tloader = DataLoader(testdata, batch_size=batch_size)

    

    return list(client2trainDL.values()), list(client2valDL.values()), tloader, client2class



def initiizeClientsData(dist_type, dname, num_groups, num_clients, batch_size, storage_dir):
    if dist_type == "nonIID":
        # assert 1 == 0 ## fix nonIId properly not implemented
        return nonIID(dname, num_groups, num_clients, batch_size, storage_dir)
    else:
        raise ValueError("Unknown distribution type")
    
 

class SubsetToDataset(torch.utils.data.Dataset):
    def __init__(self, subset, greyscale=False):
        self.subset = subset
        # self.X, self.Y = self.subset, self.subset.target
        self.greyscale = greyscale

    def __getitem__(self, index):
        x, y = self.subset[index]
        # print(">> XShape", x.shape, y)
        # trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        # x = trans(x)
        # import torchvision.transforms as T
        # x =  T.Grayscale(num_output_channels=3)(x)
        return x, y

    def __len__(self):
        return len(self.subset)




def iid_split2(dname:str, num_clients: int, storage_dir:str, batch_size=32):
  
    trainset, testset = getTrainTestDatasets(dname, storage_dir)

    
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    train_datasets = []
    val_datasets = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        train_datasets.append(ds_train)
        val_datasets.append(ds_val)
        
        # trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        # valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    # tloader = DataLoader(testset, batch_size=batch_size)
    return train_datasets, val_datasets, testset, {}


def splitDataSetIntoNClientsIID(dataset, clients):

    parts = [len(dataset)//clients for _ in range(clients)]

    parts[0] += len(dataset) % clients

    print(f"Spliting Datasets {len(dataset)} into parts:{parts}")


    subsets =  torch.utils.data.random_split(dataset, parts)

    return [SubsetToDataset(subset) for subset in subsets]


def iid_split(dname:str, num_clients: int, storage_dir:str, batch_size=-1):
  
    trainset, testset = getTrainTestDatasets(dname, storage_dir)


    train_datasets = splitDataSetIntoNClientsIID(trainset, num_clients)
    val_datasets = train_datasets
    


    
    
    return train_datasets, val_datasets, testset, {}

class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes, noise_rate):
        assert noise_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], f"Noise rate must be in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] but got {noise_rate}"
        self.dataset = dataset
        self.num_classes = num_classes
        self.class_ids = random.sample(range(num_classes), int(noise_rate*num_classes))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if y in self.class_ids:
            y_hat = random.randint(0, self.num_classes-1)
            if y_hat != y:
                y = y_hat
            else:
                y = (y+1)%self.num_classes
        return x, y


class AttackBackdoor(torch.utils.data.Dataset):
    def __init__(self, dataset, class_ids_to_poison, attack_pattern, backdoor_target_class_id):
        self.dataset = dataset
        self.class_ids = class_ids_to_poison
        self.attack_pattern = attack_pattern
        self.target_class_id = backdoor_target_class_id
        # self.backdoor_mod = backdoor_mod

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # if y in self.class_ids and idx%self.backdoor_mod == 0:
        if y in self.class_ids:
           y = self.target_class_id
        #    print("->>backdoored 1")
           x += self.attack_pattern 
        #    print("->>backdoored 2")
        return x,y