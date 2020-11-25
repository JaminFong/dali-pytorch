import argparse
import ast
import logging
import os
import sys; sys.path.append(os.path.join(sys.path[0], '..'))
import time

import model_zoo
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from dataset import imagenet_data_dali
from mmcv import Config
from models import model_derived
from tensorboardX import SummaryWriter
from tools import env, utils
from tools.lr_scheduler import get_lr_scheduler
from tools.multadds_count import comp_multadds

from trainer import Trainer



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train_Params")
    parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
    parser.add_argument('--data_path', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--load_path', type=str, default='./model_path', help='model loading path')
    parser.add_argument('--save', type=str, default='../', help='experiment name')
    parser.add_argument('--tb_path', type=str, default='', help='tensorboard output path')
    parser.add_argument('--meas_lat', type=ast.literal_eval, default='False', help='whether to measure the latency of the model')
    parser.add_argument('--job_name', type=str, default='', help='job_name')
    parser.add_argument('--port', type=int, default=23333, help='dist port')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--evaluation', type=ast.literal_eval, default='False', help='first evaluation')
    parser.add_argument('--config', type=str, default='', help='the file of the config')
    args = parser.parse_args()

    config = Config.fromfile(os.path.join('configs/train_cfg', args.config))
    if config.net_config:
        net_config = config.pop('net_config')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        raise EnvironmentError
    else:
        distributed = True
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '%d' % args.port
        env.init_dist(args.launcher)
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()

    if args.job_name != '':
        args.job_name = time.strftime("%Y%m%d-%H%M%S-") + args.job_name
        args.save = os.path.join(args.save, args.job_name)
        if local_rank == 0:
            utils.create_exp_dir(args.save)
            os.system('cp -r ./* '+args.save)
    else:
        args.save = os.path.join(args.save, 'output')
        if local_rank == 0:
            utils.create_exp_dir(args.save)

    if args.tb_path == '':
        args.tb_path = args.save

    env.get_root_logger(log_dir=args.save)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    if config.train_params.use_seed:
        utils.set_seed(config.train_params.seed)

    logging.info("args = %s", args)
    logging.info('Training with config:')
    logging.info(config.pretty_text)
    writer = SummaryWriter(args.tb_path)

    if config.model_zoo.use_model_zoo:
        model = getattr(model_zoo, config.model_zoo.model_name)(
                        **config.model_zoo.cfg if config.model_zoo.cfg else {})
    else:
        if os.path.isfile(os.path.join(args.load_path, 'net_config')):
            net_config, config.net_type = utils.load_net_config(
                                    os.path.join(args.load_path, 'net_config'))
        derivedNetwork = getattr(model_derived, '%s_Net' % config.net_type.upper())
        model = derivedNetwork(net_config, config=config)

    model.eval()
    if hasattr(model, 'net_config'):
        logging.info("Network Structure: \n" + '|\n'.join(map(str, model.net_config)))
    if args.meas_lat:
        latency_cpu = utils.latency_measure(model, (3, 224, 224), 1, 2000, mode='cpu')
        logging.info('latency_cpu (batch 1): %.2fms' % latency_cpu)
        latency_gpu = utils.latency_measure(model, (3, 224, 224), 32, 1000, mode='gpu')
        logging.info('latency_gpu (batch 32): %.2fms' % latency_gpu)
        
    params = utils.count_parameters_in_MB(model)
    logging.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(model, input_size=config.data.input_size)
    logging.info("Mult-Adds = %.2fMB" % mult_adds)

    model.cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # whether to resume from a checkpoint
    if config.optim.if_resume:
        utils.load_model(model, config.optim.resume.load_path, distributed)
        start_epoch = config.optim.resume.load_epoch + 1
    else:
        start_epoch = 0

    if config.optim.label_smooth:
        criterion = utils.cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

    if config.optim.bn_wo_wd:
        parameters = utils.add_weight_decay(model, config.optim.weight_decay)
    else:
        parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        config.optim.init_lr,
        momentum=config.optim.momentum,
        weight_decay=config.optim.weight_decay
    )

    train_loader, val_loader = imagenet_data_dali.get_data_loader(
        config.data, world_size, local_rank, args.data_path
    )

    scheduler = get_lr_scheduler(config, optimizer, train_loader._size)
    scheduler.last_step = start_epoch * (train_loader._size // config.data.batch_size + 1)-1

    trainer = Trainer(train_loader, val_loader, optimizer, criterion, 
                    scheduler, config, args.report_freq, distributed)

    best_epoch = [0, 0, 0] # [epoch, acc_top1, acc_top5]
    if args.evaluation:
        with torch.no_grad():
            val_acc_top1, val_acc_top5, batch_time, data_time = trainer.infer(model, start_epoch-1)
        if val_acc_top1 > best_epoch[1]:
            best_epoch = [start_epoch-1, val_acc_top1, val_acc_top5]
        logging.info('BEST EPOCH %d  val_top1 %.2f val_top5 %.2f', best_epoch[0], best_epoch[1], best_epoch[2])

    for epoch in range(start_epoch, config.train_params.epochs):
        train_acc_top1, train_acc_top5, train_obj, batch_time, data_time = trainer.train(model, epoch)
        
        with torch.no_grad():
            val_acc_top1, val_acc_top5, batch_time, data_time = trainer.infer(model, epoch)
        if val_acc_top1 > best_epoch[1]:
            best_epoch = [epoch, val_acc_top1, val_acc_top5]
            if local_rank==0:
                utils.save(model, os.path.join(args.save, 'weights.pt'))
        logging.info('BEST EPOCH %d  val_top1 %.2f val_top5 %.2f', best_epoch[0], best_epoch[1], best_epoch[2])

        if local_rank == 0:
            writer.add_scalar('train_acc_top1', train_acc_top1, epoch)
            writer.add_scalar('train_loss', train_obj, epoch)
            writer.add_scalar('val_acc_top1', val_acc_top1, epoch)

    if hasattr(model.module, 'net_config'):
        logging.info("Network Structure: \n" + '|\n'.join(map(str, model.module.net_config)))
    logging.info("Params = %.2fMB" % params)
    logging.info("Mult-Adds = %.2fMB" % mult_adds)
