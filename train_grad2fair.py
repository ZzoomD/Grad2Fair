# %%
# import dgl
import ipdb
import time
import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import warnings
# from torch_geometric.loader import DataLoader
from datetime import datetime

warnings.filterwarnings('ignore')

from load_data import *
from utils import *
import torch.nn as nn
from torch_sparse import SparseTensor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from trainer import *
import json
import wandb

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs for vanilla, bias amplification epochs for Grad2Fair.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hid_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='bail',
                        choices=['bail', 'pokec_z', 'pokec_n', 'credit'])
    parser.add_argument("--layer_num", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--encoder', type=str, default='gcn', choices=['gcn', 'gin'])
    parser.add_argument('--enable_shortcut', type=str2bool, default=True)
    parser.add_argument('--save_results', type=str2bool, default=True)
    parser.add_argument('--log_dir', type=str, default='./wandb')
    parser.add_argument('--run_type', type=str, default='vanilla', choices=['vanilla', 'grad2fair'])
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for spurious dimension identification.')
    parser.add_argument('--upweight_epochs', type=int, default=100, help='Upweighting epochs for Grad2Fair.')
    parser.add_argument('--alpha', type=float, default=0.5, help='coefficient for spurious loss.')
    parser.add_argument('--to_wandb', action='store_true', help='whether to log training process to wandb.')

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def run(args):
    torch.set_printoptions(threshold=float('inf'))
    """
    Load data
    """
    data = FairDataset(args.dataset, args.device)
    data.load_data()

    num_class = 1
    args.in_dim = data.features.shape[1]
    args.nnode = data.features.shape[0]
    args.out_dim = num_class

    """
    Build model, optimizer, and loss fuction
    """
    trainer = Trainer(args)

    """
    Train model
    """
    if args.run_type == 'vanilla':
        trainer.train_vanilla(data, pbar=args.pbar)
        load_weight_path = f'./weights/{args.run_type}_{args.encoder}.pt'
    elif args.run_type == 'grad2fair':
        trainer.train_grad2fair(data, pbar=args.pbar, enable_shortcut=args.enable_shortcut, seed=int(args.seed.split('seed')[1]))
        load_weight_path = f'./weights/{args.run_type}_{args.encoder}.pt'

    """
    evaluation
    """
    state_dict = torch.load(load_weight_path)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('classifier.mask')}
    trainer.load_state_dict(filtered_state_dict, strict=False)
    trainer.eval()
    with torch.no_grad():
        output = trainer(data.features, data.edge_index)
        if isinstance(output, tuple):
            emb, output = output[0], output[1]

    pred = (output.squeeze() > 0).type_as(data.labels)
    # utility performance
    auc_test = roc_auc_score(data.labels[data.idx_test].cpu(), output[data.idx_test].cpu())
    f1_test = f1_score(data.labels[data.idx_test].cpu(), pred[data.idx_test].cpu())
    acc_test = accuracy_score(data.labels[data.idx_test].cpu(), pred[data.idx_test].cpu())
    # fairness performance
    parity_test, equality_test = fair_metric(pred[data.idx_test].cpu().numpy(),
                                            data.labels[data.idx_test].cpu().numpy(),
                                            data.sens[data.idx_test].cpu().numpy())
    return auc_test, f1_test, acc_test, parity_test, equality_test

class LogWriter:
    def __init__(self, project_name='Grad2Fair', group_name=None, run_name=None, config=None, logdir='./logs'):
        self.run = wandb.init(
            project=project_name,
            config=config,
            dir=logdir,
            group=group_name,
            name=run_name,
            reinit=True
        )

    def record(self, loss_item: dict):
        wandb.log(loss_item)


if __name__ == '__main__':
    # Training settings
    args = args_parser()

    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')

    model_num = 1
    results = Results(args.seed_num, model_num, args)

    group_name = f'{args.run_type}-{args.dataset}-{args.encoder}-{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    args.log_dir = os.path.join(args.log_dir, group_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("./weights", exist_ok=True)

    for seed in range(args.seed_num):
        args.seed = f"seed{seed}"
        # set seeds
        args.pbar = tqdm(total=args.epochs if args.run_type == 'vanilla' else args.upweight_epochs, desc=f"Seed {seed + 1}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        set_seed(seed)
        
        if args.to_wandb:
            args.wandb_writer = LogWriter(config=args.__dict__, group_name=group_name, run_name=args.seed, logdir=args.log_dir)

        # running train
        results.auc[seed, :], results.f1[seed, :], results.acc[seed, :], results.parity[seed, :], results.equality[seed, :] = run(args)

    if hasattr(args, 'wandb_writer'):
        args.wandb_writer.run.finish()

    # reporting results
    results.report_results()
    if args.save_results:
        results.save_results(args)