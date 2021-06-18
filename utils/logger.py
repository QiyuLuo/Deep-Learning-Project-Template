import logging
import yaml
import os
import shutil
import datetime
import random
import time
import glob
import sys
import os.path as osp
from utils.env import collect_env
USE_TENSORBOARD = True
try:
    from tensorboardX import SummaryWriter
    print('Using tensorboardX')
except:
    print('not find tensorboardX')
    USE_TENSORBOARD = False

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

class Logger:
    r'''
        usage:
        logger = Logger(args)
        logger.log.info('')
        logger.writer.add_scalar(tag, value, step)
        记录以下内容：
        日志文件：记录运行全过程的日志。
        权重文件：运行过程中保存的checkpoint。
        可视化文件：tensorboard中运行得到的文件。
        配置文件：详细记录当前运行的配置（调参必备）。
        文件备份：用于保存当前版本的代码，可以用于回滚。
    '''

    def __init__(self, args):
        # 设置实验名称
        args.exp_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + \
                        "{:04d}".format(random.randint(0, 1000))

        # 文件处理，默认在$(project)/exp
        exp_root = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'exp')
        self.cleanEmptyFile(exp_root) # 删除空的训练日志文件夹
        args.save_dir = os.path.join(exp_root, args.exp_name) # 实验保存目录
        print('The experiment log and models will be saved to ', args.save_dir)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # 配置文件
        with open(os.path.join(args.save_dir, "config.yml"), "w") as f:
            yaml.dump(args, f)

        # 日志文件
        log_format = "%(asctime)s - %(levelname)s: %(message)s"
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt="%Y-%m-%d %H:%M:%S ")

        fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        print(args)
        self.log = logging
        env_info = collect_env()
        self.log.info(env_info)
        # Tensorboard可视化文件
        if USE_TENSORBOARD:
            writer = SummaryWriter(os.path.join(args.save_dir, "runs/%s-%05d" %
                                   (time.strftime("%m-%d", time.localtime()), random.randint(0, 100))
                                                ))
            self.writer = writer

        # 文件备份，核心代码
        create_exp_dir(args.save_dir,
                       scripts_to_save=glob.glob('*.py'))

        # 模型保存目录
        if not os.path.exists(os.path.join(args.save_dir, 'weights')):
            os.makedirs(os.path.join(args.save_dir, 'weights'))
    def cleanEmptyFile(self, exp_root):
        seqs = os.listdir(exp_root)
        for seq in seqs:
            if len(os.listdir(osp.join(exp_root, seq, 'weights'))) == 0:
                shutil.rmtree(osp.join(exp_root, seq))
                print('delete empty train log directory {}'.format(osp.join(exp_root, seq)))
