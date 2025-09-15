


import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


def get_mnist(data_path: str = "./data"): 

    tr= Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(root=data_path, train=True, download=True,transform=tr)
    testset = MNIST(root=data_path, train=False, download=True,transform=tr)


    return trainset,testset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float=0.1):

    trainset,testset=get_mnist()

    #iid, non-iid ,ldi partition scema
    #split the dataset into num_partitions parts
    num_image=len(trainset)//num_partitions 

    partition_len=[num_image]*num_partitions
    #num_image = 6000,num_partitions = 10,partition_len = [6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]


    trainsets= random_split(trainset,partition_len,torch.Generator().manual_seed(2023))

    #create dataloaders with train + val split

    trainloaders=[]
    validateloaders=[]
    for trainset_ in trainsets:
        num_total=len(trainset_)
        num_val=int(num_total*val_ratio)
        num_train=num_total-num_val

        for_train, for_val = random_split(trainset_,[num_train,num_val],torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train,batch_size=batch_size,shuffle=True,num_workers=2))
        validateloaders.append(DataLoader(for_val,batch_size=batch_size,shuffle=False,num_workers=2))


    testloader= DataLoader(testset,batch_size=128)
    return trainloaders, validateloaders, testloader
