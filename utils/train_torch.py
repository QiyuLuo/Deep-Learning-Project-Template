from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.utils.data
from utils.opts import opts
from utils.logger import Logger
from modeling.build import create_model, save_model, load_model
import numpy as np
from engine.seti_trainer import setiTrainer
from utils.utils import get_transforms, seed_torch
import pandas as pd
import os.path as osp
from data.datasets.base_dataset import TrainDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from utils.utils import get_score
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data.dataset import random_split
import pdb

def get_scheduler(optimizer):
    if opt.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opt.factor, patience=opt.patience, verbose=True,
                                      eps=opt.eps)
    elif opt.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=opt.min_lr, last_epoch=-1)
    elif opt.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=opt.T_0, T_mult=1, eta_min=opt.min_lr, last_epoch=-1)
    return scheduler

def train_loop(train_loader, val_loader):
    # pdb.set_trace()

    print('Creating model...')
    model = create_model(opt, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    scheduler = get_scheduler(optimizer)
    trainer = setiTrainer(opt, model, optimizer, scheduler=scheduler)
    trainer.set_device(opt.gpus, opt.device)

    start_epoch = 0
    best_loss = np.inf
    model_dir = os.path.join(opt.save_dir, 'weights')

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        # train
        log_dict_train, outputs, labels = trainer.train(epoch, train_loader)
        avg_loss = log_dict_train['loss']
        score = get_score(labels, outputs)
        # print loss info and visualization loss
        epoch_info = 'Epoch: {} |'.format(epoch)
        for k, v in log_dict_train.items():
            logger.writer.add_scalar('train_{}'.format(k), v, epoch)
            epoch_info += '{} {:8f} | '.format(k, v)
        epoch_info += 'score {:8f} '.format(score)
        logger.log.info(epoch_info)
        logger.writer.add_scalar('lr', trainer.scheduler.get_last_lr()[0], epoch)
        # save model
        save_model(os.path.join(model_dir, f'{opt.arch}_model_last.pth'),
                   epoch, model)

        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.log.info(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
            save_model(os.path.join(model_dir, f'{opt.arch}_best_loss.pth'),
                       epoch, model)


def main(opt):
    print('Setting up data...')
    dataset = MNIST('')

    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size)
    split = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)
    if opt.debug:
        opt.epochs = 1
        train_sample = RandomSampler(dataset, replacement=True, num_samples=100)
        val_sample = RandomSampler(dataset, replacement=True, num_samples=100)
    train_dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sample, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=val_sample, num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    # train
    logger.log.info(f"========== start train ==========")
    train_loop(train_dataloader, val_dataloader)

if __name__ == '__main__':
    opt = opts().init()
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    logger = Logger(opt)
    seed_torch(opt.seed)
    data_root = os.getenv('data_root')
    main(opt)
