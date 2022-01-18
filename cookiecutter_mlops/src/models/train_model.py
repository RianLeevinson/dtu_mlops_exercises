#!/usr/bin/env python3

import argparse
import sys
import os

import numpy as np
import torch
from torch import nn
from torch import optim
from data import mnist
from src.models.model import MyAwesomeModel
from data import CorruptedMNIST
import helper
import wandb
from omegaconf import OmegaConf
   
def train(self):

    np.random.seed(42)
    torch.manual_seed(42)
    workingdir = os.getcwd() + "/"

    # Load config file
    config = OmegaConf.load(workingdir + "config/config.yaml")

    # Initialize logging with wandb and track conf settings
    wandb.init(project="MLOps-Project", config=dict(config))
    print("Training day and night")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument("--e", type=int, default=10, help="epoch")
    parser.add_argument("--b", type=int, default=64, help="batch size")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    #hyperparameters
    epochs = args.e
    lr = args.lr
    batchsize = args.b
    
    #training_data

    train_data= mnist()[0]
    trainset=CorruptedMNIST(train_data)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)

    # TODO: Implement training loop here
    model = MyAwesomeModel()

    wandb.watch(model, log_freq=100)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    counter = 0
    #training loop
    for i in range(epochs):
        total_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0],-1)
            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
            total_loss += loss.item()
            if counter == 0:
                prev_loss = loss.item()
                torch.save(model.state_dict(), 'checkpoint.pth')
                print('saving first model')
            if counter > 0:
                if loss.item() < prev_loss:
                    torch.save(model.state_dict(), 'checkpoint.pth')
                    #print('saving subsequent model(s)')
            prev_loss = loss.item()
            counter += 1
        else:
            print(total_loss)
            print(f"Training loss: {total_loss/len(trainloader)}")