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
from utils.utils import get_score, get_transforms, seed_torch
import pandas as pd
import os.path as osp
from data.datasets.base_dataset import TrainDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import pdb

def train_loop(folds, fold):
    # pdb.set_trace()
    logger.log.info(f"========== fold: {fold} training ==========")
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[opt.target_col].values

    train_dataset = TrainDataset(train_folds,
                                 transform=get_transforms(opt, data='train'), opt=opt)
    valid_dataset = TrainDataset(valid_folds,
                                 transform=get_transforms(opt, data='valid'), opt=opt)

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valid_dataset,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    print('Creating model...')
    model = create_model(opt, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    trainer = setiTrainer(opt, model, optimizer, scheduler=None)
    trainer.set_device(opt.gpus, opt.device)

    start_epoch = 0
    best_score = 0.
    best_loss = np.inf
    model_dir = os.path.join(opt.save_dir, 'weights')

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        # train
        log_dict_train, outputs, labels = trainer.train(epoch, train_loader)
        train_score = get_score(labels, outputs)

        # print loss info and visualization loss
        epoch_info = 'epoch: {} |'.format(epoch)
        for k, v in log_dict_train.items():
            logger.writer.add_scalar('train_fold{}_{}'.format(fold, k), v, epoch)
            epoch_info += '{} {:8f} | '.format(k, v)
        epoch_info += 'score {:8f} '.format(train_score)
        logger.log.info(epoch_info)

        # eval
        ret, preds, _ = trainer.val(epoch, val_loader)

        # get val scoring
        score = get_score(valid_labels, preds)

        avg_val_loss = ret['loss']
        logger.log.info(f'Epoch {epoch} - val Score: {score:.4f}')

        # save model
        save_model(os.path.join(model_dir, f'{opt.arch}_fold{fold}_model_last.pth'),
                   epoch, model, optimizer, preds)

        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if score > best_score:
            best_score = score
            logger.log.info(f'Epoch {epoch} - Save Best Score: {best_score:.4f} Model')
            save_model(os.path.join(model_dir, f'{opt.arch}_fold{fold}_best_score.pth'),
                       epoch, model, optimizer, preds)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            logger.log.info(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
            save_model(os.path.join(model_dir, f'{opt.arch}_fold{fold}_best_loss.pth'),
                       epoch, model, optimizer, preds)
    valid_folds['preds'] = torch.load(os.path.join(model_dir, f'{opt.arch}_fold{fold}_best_loss.pth'),
                                      map_location=torch.device('cpu'))['preds']
    return valid_folds

def get_train_file_path(image_id):
    return data_root + "seti/train/{}/{}.npy".format(image_id[0], image_id)

def get_test_file_path(image_id):
    return data_root + "seti/test/{}/{}.npy".format(image_id[0], image_id)

def get_result(result_df):
    preds = result_df['preds'].values
    labels = result_df[opt.target_col].values
    score = get_score(labels, preds)
    logger.log.info(f'Score: {score:<.4f}')

def main(opt):
    torch.manual_seed(opt.seed)
    # torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')

    train = pd.read_csv(osp.join(data_root, 'seti/train_labels.csv'))
    test = pd.read_csv(osp.join(data_root, 'seti/sample_submission.csv'))
    train['file_path'] = train['id'].apply(get_train_file_path)
    test['file_path'] = test['id'].apply(get_test_file_path)

    if opt.debug:
        opt.epochs = 1
        train = train.sample(n=1000, random_state=opt.seed).reset_index(drop=True)

    Fold = StratifiedKFold(n_splits=opt.n_fold, shuffle=True, random_state=opt.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train[opt.target_col])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    if opt.state == 'train':
        # train
        oof_df = pd.DataFrame()
        for fold in range(opt.n_fold):
            if fold in opt.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.log.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # all folds CV result
        logger.log.info(f"========== CV ==========")
        get_result(oof_df)
        # Save the predicted results for all training sets
        oof_df.to_csv(opt.save_dir +'/oof_df.csv', index=False)

if __name__ == '__main__':
    opt = opts().init()
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    logger = Logger(opt)
    seed_torch(opt.seed)
    data_root = os.getenv('data_root')
    main(opt)
