import logging
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(launcher, backend='nccl', **kwargs):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError


def _init_dist_slurm(backend, **kwargs):
    raise NotImplementedError


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_root_logger(log_dir=None, log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            level=log_level,
            datefmt='%m/%d %I:%M:%S %p')
    rank = dist.get_rank()
    if rank != 0:
        logger.setLevel('ERROR')

    if log_dir and rank == 0:
        filename = 'train_dist.log'
        log_file = osp.join(log_dir, filename)
        _add_file_handler(logger, log_file, level=log_level)
    return logger

def _add_file_handler(logger,
                    filename=None,
                    mode='w',
                    level=logging.INFO):
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s'))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt