

import argparse
import sys

import torch
from torch import nn
from torch import optim
from data import mnist
from model import MyAwesomeModel
from data import CorruptedMNIST
import helper
import wandb



import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        wandb.init(config=args)
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
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


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    