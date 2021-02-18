print("importing torch")

import copy
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import os
import wandb
torch.manual_seed(42)
random.seed(42)

from dataloaders import dataLoader, Train, Test, NewTrain
from model import Net
wandb.login()
net = Net()
net.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# LR Scedule I wanded momentum so I used SGD
optims = {
    5: optim.SGD(net.parameters(), 1,  momentum=0.875, weight_decay=0.00125),
    5: optim.SGD(net.parameters(), 0.1,  momentum=0.875, weight_decay=0.00125),
    5: optim.SGD(net.parameters(), 0.01,  momentum=0.875, weight_decay=0.000125), 
    5: optim.SGD(net.parameters(), 0.001,  momentum=0.875, weight_decay=0.000125),
    5: optim.SGD(net.parameters(), 0.00001,  momentum=0.875)
    }
crit = nn.MSELoss()
wandb.init(project="crispr-pots", config={
    "learning_rate": [0.1, 0.01, 0.001, 0.00001],
    "optim": "ADAM",
    "loss": "MSE",
    "architecture": "RES-CNN-4",
    "dataset": "CRSIPRSQL-VALSMALL",
    "epochs": 250,
    "batch": 5
})
wandb.watch(net)
print('loading data')
# dataLoader
data = dataLoader(batch=5)
# look at selected predictions before training
with torch.no_grad():
    for i, d in enumerate(dataLoader[1][6:7], 0):        
        inputs, labels = d[0], d[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        outputs = net(inputs)
        print(f"label: {labels}. output: {outputs}. L1loss: {F.l1_loss(outputs, labels)}. Loss: {crit(outputs, labels)}")
# train
NewTrain(250, optim1, crit, 1000, data[1][5:7], data[1][5:7], net, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), optims)
PATH = f'.net.pth'
torch.save(net.state_dict(), PATH)
print(f"Net saved to {PATH}")
Test(net,data[1][5:7], torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), crit)
# look at outputs after training
with torch.no_grad():
    for i, d in enumerate(data[1][33:35], 0):        
        inputs, labels = d[0], d[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        outputs = net(inputs)
        print(f"label: {labels}. output: {outputs}. L1loss: {F.l1_loss(outputs, labels)}. Loss: {crit(outputs, labels)}")




