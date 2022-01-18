
import argparse
import sys

import torch
from torch import nn
from torch import optim
from data import mnist
from src.models.model import MyAwesomeModel
from data import CorruptedMNIST
import helper
import wandb



import matplotlib.pyplot as plt

def evaluate(self):
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('load_model_from', default="checkpoint.pth")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)
    test_data= mnist()[1]
    testset=CorruptedMNIST(test_data)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    # TODO: Implement evaluation logic here
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
