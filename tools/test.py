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
from modeling.build import create_model, load_model
import numpy as np
from utils.utils import get_transforms, seed_torch
import pandas as pd
import os.path as osp
from data.datasets.base_dataset import TrainDataset
from torch.utils.data import DataLoader
from utils.utils import set_device
from tqdm import tqdm
import pdb

def test_loop(test_data):
    # pdb.set_trace()

    test_dataset = TrainDataset(test_data,
                                 transform=get_transforms(opt, data='valid'), opt=opt, is_train=False)

    test_loader = DataLoader(test_dataset,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    print('Creating model...')
    model = create_model(opt, pretrained=False)
    set_device(model, opt.gpus, opt.device)
    model.eval()
    if opt.load_model != '':
        model = load_model(model, opt.load_model)

    result = []
    for batch in tqdm(test_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        with torch.no_grad():
            output = model(batch['input']).cpu().sigmoid().numpy()
        result.append(output)
        del batch, output

    results = np.concatenate(result)
    res_df = test_data.copy()
    res_df['target'] = results
    res_df.to_csv()
    res_df.to_csv(osp.join(osp.dirname(opt.load_model), 'submit.csv'), index=False, columns=['id', 'target'])


def get_test_file_path(image_id):
    return data_root + "seti/test/{}/{}.npy".format(image_id[0], image_id)

def main(opt):
    torch.manual_seed(opt.seed)
    print('Setting up data...')

    test = pd.read_csv(osp.join(data_root, 'seti/sample_submission.csv'))
    test['file_path'] = test['id'].apply(get_test_file_path)

    if opt.debug:
        opt.epochs = 1
        test = test.sample(n=1000, random_state=opt.seed).reset_index(drop=True)
    test_loop(test)


if __name__ == '__main__':
    opt = opts().init()
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    seed_torch(opt.seed)
    data_root = os.getenv('data_root')
    main(opt)
