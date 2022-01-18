#!/usr/bin/env python3

import argparse
import sys

import helper
import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn, optim

from data import CorruptedMNIST, mnist
from src.models.model import MyAwesomeModel


def evaluate(self) -> None:
    '''evaluation function'''

    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('load_model_from', default="checkpoint.pth")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)
    test_data= mnist()[1]
    testset=CorruptedMNIST(test_data)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    state_dict = torch.load(args.load_model_from)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    with torch.no_grad():
    # set model to evaluation mode
        model.eval()
    for images, labels in testloader:
        images, labels = next(iter(testloader))
        # Get the class probabilities
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')
