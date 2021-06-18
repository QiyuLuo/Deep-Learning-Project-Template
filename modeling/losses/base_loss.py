import torch
import torch.nn as nn

class baseLoss(torch.nn.Module):
    def __init__(self, opt):
        super(baseLoss, self).__init__()
        self.cls = nn.BCEWithLogitsLoss()
        self.opt = opt

    def forward(self, outputs, batch):
        loss = self.cls(outputs.view(-1), batch['labels'])
        loss_stats = {'loss': loss}
        return loss, loss_stats