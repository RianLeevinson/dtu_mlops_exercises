import torch
import numpy as np
from torchvision import datasets, transforms
import glob
from os import walk
BASE_PATH = "/home//rianleevinson/dtu_mlops/"
DATA_PATH = "/home//rianleevinson/dtu_mlops/data/corruptmnist/"


class CorruptedMNIST():
    def __init__(self,data):
        self.data=data
        self.images=torch.from_numpy(self.data[0].copy()).float()
        self.labels=torch.from_numpy(np.array(self.data[1]).copy())

    def __getitem__(self, idx):

        return self.images[idx],self.labels[idx]

    def __len__(self):
        return (len(self.images))


def mnist():
    # DATA_PATH = "/home//rianleevinson/dtu_mlops/data/corruptmnist/"
    # # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)

    # train_file_list = glob.glob(DATA_PATH+"train*")
    # test_file_list = glob.glob(DATA_PATH+"test*")
    # train_image_list = []
    # train_labels_list = []
    # for paths in train_file_list:
    #     train_image_list.append(np.load(paths)['images'])
    #     train_labels_list.append(np.load(paths)['labels'])
    # train_images = np.vstack(train_image_list)
    # train_labels = np.vstack(train_labels_list)
    # trainset = [train_images, train_labels]

    # test_image_list = []
    # test_labels_list = []
    # for paths in test_file_list:
    #     test_image_list.append(np.load(paths)['images'])
    #     test_labels_list.append(np.load(paths)['labels'])
    # test_images = np.vstack(test_image_list)
    # test_labels = np.vstack(test_labels_list)
    # testset = [test_images, test_labels]


    # return trainset, testset
        # exchange with the corrupted mnist dataset
    filenames = next(walk(DATA_PATH), (None, None, []))[2]  # [] if no file
    train_paths=[DATA_PATH+i for i in filenames if 'train' in i]
    test_paths=[DATA_PATH+i for i in filenames if 'test' in i]
    train_images=np.concatenate([np.load(i, allow_pickle=True)['images'] for i in train_paths],axis=0)
    train_labels=np.concatenate([np.load(i,  allow_pickle=True)['labels'] for i in train_paths],axis=0)
    train = [train_images,train_labels]

    test_images=np.concatenate([np.load(i,  allow_pickle=True)['images'] for i in test_paths],axis=0)
    test_labels=np.concatenate([np.load(i,  allow_pickle=True)['labels'] for i in test_paths],axis=0)
    test = [test_images,test_labels]



    return [test, train]



print(len(mnist()[1][0]))
