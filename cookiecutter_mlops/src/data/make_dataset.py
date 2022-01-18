# -*- coding: utf-8 -*-
import glob
import logging
from os import walk
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
from omegaconf import OmegaConf

config = OmegaConf.load('config/data_config.yaml')

BASE_PATH = config.BASEPATH
DATA_PATH = config.DATA_PATH
TRAIN_PROCESSED_PATH = config.TRAIN_PROCESSED_PATH
TEST_PROCESSED_PATH = config.TEST_PROCESSED_PATH
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

class CorruptedMNIST():
    '''dataset class for processing the MNIST data'''

    def __init__(self,data):
        self.data=data
        self.images=torch.from_numpy(self.data[0].copy()).float()
        self.labels=torch.from_numpy(np.array(self.data[1]).copy())

    def __getitem__(self, idx):
        
        return self.images[idx],self.labels[idx]

    def __len__(self):
        return (len(self.images))

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    filenames = next(walk(DATA_PATH), (None, None, []))[2]  # [] if no file
    train_paths=[DATA_PATH+i for i in filenames if 'train' in i]
    test_paths=[DATA_PATH+i for i in filenames if 'test' in i]
    train_images=np.concatenate([np.load(i, allow_pickle=True)['images'] for i in train_paths],axis=0)
    train_labels=np.concatenate([np.load(i,  allow_pickle=True)['labels'] for i in train_paths],axis=0)
    train = [train_images,train_labels]

    test_images=np.concatenate([np.load(i,  allow_pickle=True)['images'] for i in test_paths],axis=0)
    test_labels=np.concatenate([np.load(i,  allow_pickle=True)['labels'] for i in test_paths],axis=0)
    test = [test_images,test_labels]

    torch.save(train, TRAIN_PROCESSED_PATH)
    torch.save(test, TEST_PROCESSED_PATH)
    (print("success"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
