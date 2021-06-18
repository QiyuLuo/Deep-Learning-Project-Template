import argparse
import os
import yaml
import sys


class opts(object):
    r'''usage:
        opt = opts().init()
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser("image classification")
        self.parser.add_argument('--task', help="task name",
                            type=str, default="seti")

        self.parser.add_argument('--config', help="configuration file",
                            type=str, default="../config/default.yml")
        self.parser.add_argument('--save_dir', type=str,
                            help="save exp floder name", default="")
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--num_stacks', type=int, default=1,
                                 help='The number of feature map layers of network output.')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--lr_step', type=str, default='4',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--seed', type=int, default=42,
                                 help='seed')

        self.parser.add_argument('--size', type=int, default=224,
                                 help='the size of input network')
    def parse(self, args=''):
        if args == '':
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(args)
        # 加载基础配置文件
        if os.path.exists(args.config):
            opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
            opt.update(vars(args))
            args = argparse.Namespace(**opt)
        else:
            print('config file does not exist, please check the path of config file {}'.format(args.config))
        opt = args
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.num_stacks = 1  # 网络输出的feature map 层数
        # 当训练被中断时，从最后一个保存的模型开始训练，将resume设置为True,load_model设置为空。
        if opt.resume and opt.load_model == '':
            opt.load_model = os.path.join(opt.save_dir, 'weight', 'model_last.pth')
        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt
