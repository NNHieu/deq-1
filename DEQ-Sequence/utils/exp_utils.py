# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import functools
import os, shutil

import torch


# def logging(s, log_path, print_=True, log_=True):
#     if print_:
#         print(s)
#     if log_:
#         with open(log_path, 'a+') as f_log:
#             f_log.write(s + '\n')

# def get_logger(log_path, **kwargs):
#     return functools.partial(logging, log_path=log_path, **kwargs)

# def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
#     if debug:
#         print('Debug Mode : no experiment dir created')
#         return functools.partial(logging, log_path=None, log_=False)

#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)

#     print('Experiment dir : {}'.format(dir_path))
#     if scripts_to_save is not None:
#         script_path = os.path.join(dir_path, 'scripts')
#         if not os.path.exists(script_path):
#             os.makedirs(script_path)
#         for script in scripts_to_save:
#             dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
#             shutil.copyfile(script, dst_file)

#     if not os.path.exists(os.path.join(dir_path, 'log.txt')):
#         return get_logger(log_path=os.path.join(dir_path, 'log.txt'))
#     else:
#         for i in range(1, 20):
#             new_path = os.path.join(dir_path, f'log{i}.txt')
#             if not os.path.exists(new_path):
#                 return get_logger(log_path=new_path)

def create_logger(root_output_dir, log_dir, dataset, model_name, cfg_name, phase='train', scripts_to_save=None, mpi=False):
    root_output_dir = Path(root_output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    model = model_name
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    #-----
    # if mpi:
    #     from .mpilogger import MPILogHandler
    #     file_handler = MPILogHandler(str(final_log_file))
    #     # Construct an MPI formatter which prints out the rank and size
    #     mpifmt = logging.Formatter(fmt='[rank %(rank)s/%(size)s] %(asctime)s : %(message)s')
    #     file_handler.setFormatter(mpifmt)
    # else:
    file_handler = logging.FileHandler(str(final_log_file))
    fmt = logging.Formatter(fmt='%(asctime)-15s %(message)s')
    file_handler.setFormatter(fmt)
    #-----
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(log_dir) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    if scripts_to_save is not None:
        script_path = os.path.join(final_output_dir, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(final_output_dir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))
