# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
here's the file to make dataloader.
"""

from torch.utils import data

from .datasets.mnist import MNIST
from .transforms import build_transforms


def build_dataset(dataset_name=''):
    dataset = globals()[dataset_name]
    return dataset


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
    else:
        batch_size = cfg.test.batch_size
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg.dataset)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
