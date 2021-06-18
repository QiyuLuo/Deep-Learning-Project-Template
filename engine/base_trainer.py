from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import time
import torch
from progress.bar import Bar
from utils.utils import AverageMeter, mixup_data
import torch.nn as nn
import numpy as np

class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch, t_b=None, lam=None):
        outputs = self.model(batch['input'])
        mix_label = {}
        mix_label['labels'] = t_b
        loss, loss_stats = self.loss(outputs, batch, mix_label, lam)
        return outputs, loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer, scheduler):
        self.opt = opt
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_stats, self.loss = self._get_losses(opt)
        # self.model = model
        self.model_with_loss = ModleWithLoss(model, self.loss)
        # if self.loss.parameters():
        #     self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, device):

        if len(gpus) > 1:
            self.model_with_loss = nn.DataParallel(self.model_with_loss, gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.cuda()

        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        result = []
        labels = []
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar('{}'.format(opt.task), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            data_time.update(time.time() - end)
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            if phase == 'train':
                x, t = batch["input"], batch["labels"]
                # for mixup
                mixed_x, t_a, t_b, lam = mixup_data(opt.use_mixup, x, t, opt.mixup_alpha)
                batch['input'] = mixed_x
                batch['labels'] = t_a
                output, loss, loss_stats = model_with_loss(batch, t_b, lam)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output, loss, loss_stats = model_with_loss(batch)

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: epoch: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id + 1, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time) # Data显示当前批次的前向计算时间(平均时间)，Net显示
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}| {}'.format(opt.task, Bar.suffix))
            else:
                bar.next()
            labels.append(batch['labels'].cpu().numpy())
            result.append(output.clone().sigmoid().detach().cpu().numpy())
            del output, loss, loss_stats, batch
        if phase == 'train':
            self.scheduler.step()
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.

        return ret, np.concatenate(result), np.concatenate(labels)

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
