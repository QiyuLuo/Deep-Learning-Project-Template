from engine.base_trainer import BaseTrainer
from modeling.losses.base_loss import baseLoss
from utils.utils import get_criterion
class setiTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None, scheduler=None):
        super(setiTrainer, self).__init__(opt, model, optimizer=optimizer, scheduler=scheduler)

    def _get_losses(self, opt):
        loss_states = ['loss']
        loss = baseLoss(opt)
        loss = get_criterion(use_mixup=opt.use_mixup, loss_func=loss)
        return loss_states, loss

    def save_result(self, output, batch, results):
        return output