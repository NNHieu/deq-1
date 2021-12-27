# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('../')

from data_utils import get_lm_corpus
from models.deq_transformer import DEQTransformerLM
from lib.solvers import anderson, broyden
from lib import radam
from utils.exp_utils import create_logger
from utils.data_parallel import BalancedDataParallel
from torch.utils.tensorboard import SummaryWriter

from lib.viztool.landscape import *
from lib.viztool.scheduler import *
from lib.viztool.utils import name_surface_file, create_surfile

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1,"

def add_parser():
    parser = argparse.ArgumentParser(description='PyTorch DEQ Sequence Model')
    parser.add_argument('--data', type=str, default='../data/wikitext-103',
                        help='location of the data corpus (default to the WT103 path)')
    parser.add_argument('--dataset', type=str, default='wt103',
                        choices=['wt103'],
                        help='dataset name')
    parser.add_argument('--n_layer', type=int, default=12,
                        help='number of total layers')
    parser.add_argument('--eval_n_layer', type=int, default=12,
                        help='number of total layers at evaluation')
    parser.add_argument('--n_head', type=int, default=10,
                        help='number of heads (default: 10)')
    parser.add_argument('--d_head', type=int, default=50,
                        help='head dimension (default: 50)')
    parser.add_argument('--d_embed', type=int, default=-1,
                        help='embedding dimension (default: match d_model)')
    parser.add_argument('--d_model', type=int, default=500,
                        help='model dimension (default: 500)')
    parser.add_argument('--d_inner', type=int, default=8000,
                        help='inner dimension in the position-wise feedforward block (default: 8000)')

    # Dropouts
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='global dropout rate (default: 0.05)')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention map dropout rate (default: 0.0)')

    # Sequence logistics
    parser.add_argument('--tgt_len', type=int, default=150,
                        help='number of tokens to predict')
    parser.add_argument('--eval_tgt_len', type=int, default=150,
                        help='number of tokens to predict for evaluation')
    parser.add_argument('--mem_len', type=int, default=150,
                        help='length of the retained previous heads')
    parser.add_argument('--local_size', type=int, default=0,
                        help='local horizon size')

    # DEQ related [Bai et al. 2019]
    parser.add_argument('--f_solver', default='anderson', type=str,
                        choices=['anderson', 'broyden'],
                        help='forward solver to use (only anderson and broyden supported now)')
    parser.add_argument('--b_solver', default='broyden', type=str,
                        choices=['anderson', 'broyden', 'None'],
                        help='backward solver to use (if None, then set it to f_solver)')
    parser.add_argument('--stop_mode', type=str, default="rel",
                        choices=['abs', 'rel'],
                        help='stop criterion absolute or relative')
    parser.add_argument('--rand_f_thres_delta', type=int, default=0,
                        help='use (f_thres + U(-delta, 0)) for forward threshold (delta default to 0)')    
    parser.add_argument('--f_thres', type=int, default=40,
                        help='forward pass Broyden threshold')
    parser.add_argument('--b_thres', type=int, default=40,
                        help='backward pass Broyden threshold')

    # Jacobian regularization related [Bai et al. 2021]
    parser.add_argument('--jac_loss_weight', type=float, default=0.0,
                        help='jacobian regularization loss weight (default to 0)')
    parser.add_argument('--jac_loss_freq', type=float, default=0.0,
                        help='the frequency of applying the jacobian regularization (default to 0)')
    parser.add_argument('--jac_incremental', type=int, default=0,
                        help='if positive, increase jac_loss_weight by 0.1 after this many steps')
    parser.add_argument('--spectral_radius_mode', action='store_true',
                        help='compute spectral radius at validation time')

    # Training techniques
    parser.add_argument('--not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation mode')
    parser.add_argument('--adaptive', action='store_true',
                        help='use adaptive softmax')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adapative input and softmax')
    parser.add_argument('--pre_lnorm', action='store_true',
                        help='apply LayerNorm to the input instead of the output')
    parser.add_argument('--wnorm', action='store_true',
                        help='apply WeightNorm to the weights')
    parser.add_argument('--varlen', action='store_true',
                        help='use variable length')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--eval-interval', type=int, default=4000,
                        help='evaluation interval')
    parser.add_argument('--work_dir', default='LM-TFM', type=str,
                        help='experiment directory.')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    parser.add_argument('--attn_type', type=int, default=0,
                        help='attention type. 0 for ours, 1 for Shaw et al,'
                        '2 for Vaswani et al, 3 for Al Rfou et al. (Only 0 supported now)')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--gpu0_bsz', type=int, default=-1,
                        help='batch size on gpu 0')
    parser.add_argument('--max_eval_steps', type=int, default=-1,
                        help='max eval steps')
    parser.add_argument('--pretrain_steps', type=int, default=0,
                        help='number of pretrain steps (default to 0')
    parser.add_argument('--start_train_steps', type=int, default=0,
                        help='starting training step count (default to 0)')
    parser.add_argument('--patience', type=int, default=0,
                        help='patience')
    parser.add_argument('--load', type=str, required=True,
                        help='path to load weight')
    parser.add_argument('--name', type=str, default='N/A',
                        help='name of the trial')
    parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
    parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
    # Lossland params
    # parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--resolution', type=int, nargs=2, required=True)
    parser.add_argument('--rect', type=float, nargs=4, required=True)
    parser.add_argument('--rank', type=int, default=-1, required=True)
    parser.add_argument('--nproc', type=int, default=-1, required=True)

    return parser

parser = add_parser()
args = parser.parse_args()
args.tied = not args.not_tied
args.pretrain_steps += args.start_train_steps
assert args.mem_len > 0, "For now you must set mem_len > 0 when using deq"
args.work_dir += f"deq_rank{args.rank}_nproc{args.nproc}"
args.cuda = torch.cuda.is_available()
    
if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.batch_size % args.batch_chunk == 0

# logging = create_logger(args.work_dir,
    # scripts_to_save=['train_transformer.py', 'models/deq_transformer.py', '../lib/solvers.py'], debug=args.debug)

logger, output_dir, tb_log_dir = create_logger(args.work_dir, args.work_dir,
                                                     args.dataset, 'transformer', 'default', 'valid', 
                                                     scripts_to_save=['lossland.py', 'models/deq_transformer.py', '../lib/solvers.py'])
args.work_dir = output_dir
logging = logger.info

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = max(16, torch.cuda.device_count())
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len, device=device)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len, device=device)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len, device=device)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103']
    cutoffs = [20000, 40000, 200000]
    tie_projs += [True] * len(cutoffs)

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i].weight, 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i].weight, 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('WeightShareSelfAttention') != -1:
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout
        else:
            m.dropout = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        if hasattr(m, 'p'):
            m.dropatt.p = args.dropatt
        else:
            m.dropatt.dropout = args.dropatt


model = DEQTransformerLM(ntokens, args.n_layer, args.eval_n_layer, args.n_head, args.d_model, args.d_head, args.d_inner,
                            args.dropout, args.dropatt, tie_weights=args.tied, d_embed=args.d_embed,
                            div_val=args.div_val, tie_projs=tie_projs, pre_lnorm=args.pre_lnorm,
                            wnorm=args.wnorm, local_size=args.local_size, pretrain_steps=args.pretrain_steps,
                            tgt_len=args.tgt_len, mem_len=args.mem_len, cutoffs=cutoffs, load=args.load,
                            f_solver=eval(args.f_solver), b_solver=eval(args.b_solver), stop_mode=args.stop_mode, logging=logging)

args.n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])

args.multi_gpu=False
if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0 and args.batch_size != args.gpu0_bsz*torch.cuda.device_count():
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)


logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)

###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter, model):
    train_step = 1e9
    epoch = -1
    model.eval()
    model.reset_length(args.eval_tgt_len, args.mem_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    rho_list = []
    if args.spectral_radius_mode:
        print("WARNING: You are evaluating with the power method at val. time. This may make things extremely slow.")
    with torch.no_grad():
        mems = []
        for i, (data, target, seq_len) in enumerate(eval_iter):
            data, target = data.to(device), target.to(device)
            if 0 < args.max_eval_steps <= i:
                break
            ret = model(data, target, mems, train_step=train_step, f_thres=args.f_thres, 
                             b_thres=args.b_thres, compute_jac_loss=False,
                             spectral_radius_mode=args.spectral_radius_mode, writer=None)
            loss, _, sradius, mems = ret[0], ret[1], ret[2], ret[3:]
            loss = loss.mean()
            if args.spectral_radius_mode:
                rho_list.append(sradius.mean().item())
            total_loss += seq_len * loss.float().item()
            total_len += seq_len
    if rho_list:
        logging(f"(Estimated) Spectral radius over validation set: {np.mean(rho_list)}")
    model.train()
    return total_loss / total_len

###############################################################################
# Lossland code
###############################################################################
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Create surface file if not exist
dir_file = os.path.join(args.work_dir, 'dir.h5')
surf_file = os.path.join(args.work_dir, name_surface_file(args.rect, args.resolution, 'surf'))
layers = ('loss',)
create_surfile(model, layers, dir_file, surf_file, args.rect, args.resolution, logger)

# Load surface and prepair sampler
model = model.to(device)
surface = Surface.load(surf_file)
dir2d = surface.dirs
logger.info('cosine similarity between x-axis and y-axis: %f' % dir2d.similarity())
sampler = Sampler(model, surface, layers, None, comm=None, rank=0, logger=logger)
sampler.prepair()

# Get the job
inds, coords, inds_nums = scheduler.get_job_indices(*surface.get_unplotted_indices('loss'), args.rank, args.nproc)

# Exec
surface.open('r+')
sampler.run(lambda model: (evaluate(va_iter, model), ), inds, coords, max(inds_nums))
surface.close()
