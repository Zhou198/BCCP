import torch
import numpy as np
import pandas as pd

import torchvision
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset
import cifarDataset


def npTorchCGPU_manula_seed(device, myseed=1234):
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if device != "cpu":
        torch.cuda.manual_seed(myseed)

class targetOverride:
    def __init__(self, normalCate):
        self.normalCate = normalCate

    def __call__(self, y):
        return self.normalCate[y] if y in self.normalCate else 0


def dataGen(dataName, normalCate, device, seed, contamRatio=None):
    npTorchCGPU_manula_seed(device, seed)

    if dataName == "CIFAR10":
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              # transforms.RandomRotation(degrees=20),
                                              transforms.ColorJitter(0.5, 0.5, 0.5),
                                              transforms.RandomCrop(size=32, padding=2),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                              ])

        puObj = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train, target_transform=targetOverride(normalCate))
        testObj = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_train, target_transform=targetOverride(normalCate))
    elif dataName == "CIFAR100":
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              #   transforms.RandomRotation(degrees=20),
                                              transforms.ColorJitter(0.5, 0.5, 0.5),
                                              transforms.RandomCrop(size=32, padding=2),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                              ])

        puObj = cifarDataset.CIFAR100(root="data", train=True, download=True, transform=transform_train, coarse=True, target_transform=targetOverride(normalCate))
        testObj = cifarDataset.CIFAR100(root="data", train=False, download=True, transform=transform_train, coarse=True, target_transform=targetOverride(normalCate))
    elif dataName == "SVHN":
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.ColorJitter(0.2, 0.2, 0.2),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                              ])

        puObj = torchvision.datasets.SVHN(root="data", split="train", download=True, transform=transform_train, target_transform=targetOverride(normalCate))
        testObj = torchvision.datasets.SVHN(root="data", split="test", download=True, transform=transform_train, target_transform=targetOverride(normalCate))
    elif dataName == "MNIST":
        puObj = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor(),
                                           target_transform=targetOverride(normalCate))
        testObj = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor(),
                                             target_transform=targetOverride(normalCate))
    elif dataName == "fashionMNIST":
        puObj = torchvision.datasets.FashionMNIST(root="data", train=True, download=True,
                                                  transform=transforms.ToTensor(),
                                                  target_transform=targetOverride(normalCate))
        testObj = torchvision.datasets.FashionMNIST(root="data", train=False, download=True,
                                                    transform=transforms.ToTensor(),
                                                    target_transform=targetOverride(normalCate))
    else:
        raise ValueError("dataset name is wrong")

    IDidx, OODidx = [], []
    for i, (_, y) in enumerate(puObj):
        if y == 0:
            OODidx.append(i)
        else:
            IDidx.append(i)

    currRatio = len(OODidx) / len(puObj)
    if contamRatio is not None:
        if contamRatio > currRatio:
            IDidx = IDidx[:int(len(IDidx) * currRatio / (1 - currRatio))]
        else:
            OODidx = OODidx[:int(len(OODidx) * (1 - currRatio) / currRatio)]

        puObj = Subset(puObj, IDidx + OODidx)
        currRatio = len(OODidx) / (len(IDidx) + len(OODidx))

    print(f"OOD ratio in {dataName} data is: {currRatio * 100}%\npuObj has length {len(puObj)}")
    return puObj, testObj

def write_perfSummary(resultDict, batch_size, epoch, T, K, basedir, lengthsubt=2000, mostRecent=2000):
    xt = np.arange(1 + T)
    xs = np.arange(1, 1 + epoch)

    subxtnp = np.linspace(1, T, num=lengthsubt, dtype=int)

    accum_card = np.zeros([1 + T, 2])
    accum_card[1:, 0] = resultDict["accum_card"][1:, 0] / (xt[1:] * batch_size)
    accum_card[1:, 1] = resultDict["accum_card"][1:, 1] / resultDict["accum_card"][1:, 2]

    if "regret" in resultDict:
        regret = resultDict["regret"]
        regret[1:] = regret[1:] / xt[1:].reshape(-1, 1)
        for i in range(1 + K):
            pd.DataFrame([regret[subxtnp, i].tolist()]).to_csv(f"{basedir}regret_{['cardLoss', 'accLoss' + str(i)][i > 0]}.csv", mode="a", header=False, index=False)

        for i in range(K):
            pd.DataFrame([resultDict["lamMat"][:, i].tolist()]).to_csv(f"{basedir}lambda_class{i+1}.csv", mode="a", header=False, index=False)

    elif "regret_card" in resultDict:
        regret_card = resultDict["regret_card"]
        regret_card[1:] = regret_card[1:] / xt[1:]
        pd.DataFrame([regret_card[subxtnp].tolist()]).to_csv(f"{basedir}regret_cardLoss.csv", mode="a", header=False, index=False)

        regret_recall = resultDict["regret_recall"]
        regret_recall[1:] = regret_recall[1:] / xt[1:].reshape(-1, 1)
        for i in range(1 + K):
            pd.DataFrame([regret_recall[subxtnp, i].tolist()]).to_csv(f"{basedir}regret_accLoss{i}.csv", mode="a", header=False, index=False)
            if len(resultDict["lamMat"][0]) == K:
                if i > 0:
                    pd.DataFrame([resultDict["lamMat"][:, i - 1].tolist()]).to_csv(f"{basedir}lambda_class{i}.csv", mode="a", header=False, index=False)
            else:
                pd.DataFrame([resultDict["lamMat"][:, i].tolist()]).to_csv(f"{basedir}lambda_class{i}.csv", mode="a", header=False, index=False)


    accum_recall = np.zeros_like(resultDict["accum_recall"])
    accum_recall[1:] = resultDict["accum_recall"][1:] / resultDict["accum_count"][1:]

    interval_recall = np.zeros([1 + T, 1 + K])
    for i in range(1, 1 + T):
        numer = resultDict["accum_recall"][i] - resultDict["accum_recall"][max(i - mostRecent, 0)]
        denom = resultDict["accum_count"][i] - resultDict["accum_count"][max(i - mostRecent, 0)]
        interval_recall[i] = numer / denom

    pd.DataFrame([xt[subxtnp].tolist()]).to_csv(f"{basedir}iterationGrid.csv", mode="a", header=False, index=False)
    pd.DataFrame([xs.tolist()]).to_csv(f"{basedir}epochGrid.csv", mode="a", header=False, index=False)

    for i, val in enumerate(["overall_card", "normal_card"]):
        pd.DataFrame([accum_card[subxtnp, i].tolist()]).to_csv(f"{basedir}accum_{val}.csv", mode="a", header=False, index=False)

        pd.DataFrame([resultDict["holdout_card"][:, i].tolist()]).to_csv(f"{basedir}holdout_{val}.csv", mode="a", header=False, index=False)
        if "fullFB_train_card" in resultDict:
            pd.DataFrame([resultDict["fullFB_train_card"][:, i].tolist()]).to_csv(f"{basedir}fullFB_train_{val}.csv", mode="a", header=False, index=False)

    for i in range(1 + K):
        pd.DataFrame([accum_recall[subxtnp, i].tolist()]).to_csv(f"{basedir}accumRecall_class{i}.csv", mode="a", header=False, index=False)
        pd.DataFrame([interval_recall[subxtnp, i].tolist()]).to_csv(f"{basedir}recentRecall_class{i}.csv", mode="a", header=False, index=False)

        pd.DataFrame([resultDict["holdout_recall"][:, i].tolist()]).to_csv(f"{basedir}holdoutRecall_class{i}.csv", mode="a", header=False, index=False)
        if "fullFB_train_recall" in resultDict:
            pd.DataFrame([resultDict["fullFB_train_recall"][:, i].tolist()]).to_csv(f"{basedir}fullFB_trainRecall_class{i}.csv", mode="a", header=False, index=False)









