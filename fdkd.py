#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from load_data import *
from trainer import ConstructModel
from utils import *
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
import torch.nn as nn
from torch_sparse import SparseTensor
from typing import List

from tqdm import tqdm
import os


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


def train_fdkd_teacher(model, optimizer, criterion, epochs, data, save_name):
    best_loss = 100
    save_model = 0
    for epoch in tqdm(range(epochs), desc='Training teacher'):
        if isinstance(model, list):
            backbone = model[0]
            classifier = model[1]

            backbone.train()
            classifier.train()
            optimizer.zero_grad()

            h = backbone(data.features, data.edge_index)
            output = classifier(h)

            loss_train = criterion(output[data.idx_train],
                                   data.labels[data.idx_train])
            loss_train.backward()
            optimizer.step()

            backbone.eval()
            classifier.eval()

            h = backbone(data.features, data.edge_index)
            output = classifier(h)
            loss_val = criterion(output[data.idx_val],
                                 data.labels[data.idx_val])

            if loss_val.item() < best_loss:
                save_model += 1
                best_loss = loss_val.item()
                torch.save({
                    'backbone': backbone.state_dict(),
                    'classifier': classifier.state_dict()
                }, save_name)
            elif save_model == 0 and epoch == epochs-1:
                torch.save({
                    'backbone': backbone.state_dict(),
                    'classifier': classifier.state_dict()
                }, save_name)
            
        else:
            model.train()
            optimizer.zero_grad()
            h, output = model(data.features, data.edge_index)
            loss_train = criterion(output[data.idx_train], data.labels[data.idx_train])
            loss_train.backward()
            optimizer.step()

            model.eval()
            h, output = model(data.features, data.edge_index)
            loss_val = criterion(output[data.idx_val], data.labels[data.idx_val])

            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(model.state_dict(), save_name)


def run_fdkd(args):
    torch.set_printoptions(threshold=float('inf'))
    """
    Load data
    """
    data = FairDataset(args.dataset, args.device)
    data.load_data()

    num_class = 2

    """
    Construct model and optimizer
    """
    tea_backbone = ConstructModel(in_dim=data.features.shape[1], hid_dim=args.hidden, encoder=args.model, layer_num=3)
    tea_cls = nn.Linear(in_features=args.hidden, out_features=num_class)
    optimizer_tea = optim.Adam(list(tea_backbone.parameters())+list(tea_cls.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    tea_backbone = tea_backbone.to(args.device)
    tea_cls = tea_cls.to(args.device)
    
    # student model and optimizer
    stu_backbone = ConstructModel(in_dim=data.features.shape[1], hid_dim=args.hidden, encoder=args.model, layer_num=2) #2层对应的backbone为1层
    stu_cls = nn.Linear(in_features=args.hidden, out_features=num_class)
    optimizer_stu = optim.Adam(list(stu_backbone.parameters())+list(stu_cls.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    stu_backbone = stu_backbone.to(args.device)
    stu_cls = stu_cls.to(args.device)

    """
    Training model
    """
    # train teacher model
    criterion = torch.nn.CrossEntropyLoss()
    train_fdkd_teacher([tea_backbone, tea_cls], optimizer_tea, criterion, args.tea_epochs, data, save_name=f'./weights/{args.model}_fdkd_teacher.pt')

    ckpt = torch.load(f'./weights/{args.model}_fdkd_teacher.pt', map_location=args.device)
    tea_backbone.load_state_dict(ckpt['backbone'])
    tea_cls.load_state_dict(ckpt['classifier'])

    tea_backbone.eval()
    tea_cls.eval()
    with torch.no_grad():
        h_t = tea_backbone(data.features, data.edge_index)
        logits_t = tea_cls(h_t)
        output_soft_train = F.softmax(logits_t[data.idx_train] / args.tem, dim=1)
        output_soft_val = F.softmax(logits_t[data.idx_val] / args.tem, dim=1)

    best_loss = 100
    save_model = 0
    for epoch in tqdm(range(args.epochs), desc='Training teacher'):
        stu_backbone.train()
        stu_cls.train()
        
        optimizer_stu.zero_grad()
        h_s = stu_backbone(data.features, data.edge_index)
        logits_s = stu_cls(h_s)
        log_p_s_train = F.log_softmax(logits_s[data.idx_train] / args.tem, dim=1)

        kd_loss = F.kl_div(log_p_s_train, output_soft_train, reduction='batchmean') * (args.tem ** 2)

        ce_loss = criterion(logits_s[data.idx_train], data.labels[data.idx_train])
        loss_train = args.alpha * kd_loss + (1 - args.alpha) * ce_loss
        
        loss_train.backward()
        optimizer_stu.step()

        # validation
        stu_backbone.eval()
        stu_cls.eval()
        
        h_s = stu_backbone(data.features, data.edge_index)
        logits_s = stu_cls(h_s)
        log_p_s_val = F.log_softmax(logits_s[data.idx_val] / args.tem, dim=1)
        
        kd_loss = F.kl_div(log_p_s_val, output_soft_val, reduction='batchmean') * (args.tem ** 2)
        ce_loss = criterion(logits_s[data.idx_val], data.labels[data.idx_val])
        loss_val = args.alpha * kd_loss + (1 - args.alpha) * ce_loss

        if loss_val.item() < best_loss:
            save_model += 1
            best_loss = loss_val.item()
            torch.save({
                    'backbone': stu_backbone.state_dict(),
                    'classifier': stu_cls.state_dict()
                    }, f'./weights/{args.model}_fdkd_student.pt')
        elif save_model == 0 and epoch == args.epochs-1:
            torch.save({
                    'backbone': stu_backbone.state_dict(),
                    'classifier': stu_cls.state_dict()
                    }, f'./weights/{args.model}_fdkd_student.pt')
    """
    Evaluating model
    """
    ckpt = torch.load(f'./weights/{args.model}_fdkd_student.pt', map_location=args.device)
    stu_backbone.load_state_dict(ckpt['backbone'])
    stu_cls.load_state_dict(ckpt['classifier'])
    
    h_s = stu_backbone(data.features, data.edge_index)
    logits_s = stu_cls(h_s)
    output_preds = logits_s.argmax(dim=1).type_as(data.labels)
    output = torch.max(logits_s, dim=1).values
    auc_roc_test = roc_auc_score(data.labels.cpu().numpy()[data.idx_test.cpu()],
                                    output.detach().cpu().numpy()[data.idx_test.cpu()])
    f1_s = f1_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    acc = accuracy_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    parity, equality = fair_metric(output_preds[data.idx_test].cpu().numpy(),
                                    data.labels[data.idx_test].cpu().numpy(),
                                    data.sens[data.idx_test].cpu().numpy())

    return auc_roc_test, f1_s, acc, parity, equality


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--tea_epochs', type=int, default=1500, help='Number of epochs to train the teacher models')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument("--num_heads", type=int, default=1, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--alpha', type=float, default=0.5, help='coefficient for loss function')
    parser.add_argument('--tem', type=float, default=0.5, help='temperature of the Softmax function')


    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    auc, f1, acc, parity, equality = np.zeros(shape=(args.seed_num, 1)), np.zeros(shape=(args.seed_num, 1)), \
                                     np.zeros(shape=(args.seed_num, 1)), np.zeros(shape=(args.seed_num, 1)), \
                                     np.zeros(shape=(args.seed_num, 1))
    
    os.makedirs('./weights', exist_ok=True)
    os.makedirs('./results/fdkd', exist_ok=True)

    for seed in range(args.seed_num):
        # set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        # torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        auc[seed, 0], f1[seed, 0], acc[seed, 0], parity[seed, 0], equality[seed, 0] = run_fdkd(args)

        print(f"========finish seed {seed}========")

    # print report
    print("============" + "FDKD" + "============")
    print(f"AUCROC: {np.around(np.mean(auc[:, 0]) * 100, 2)} ± {np.around(np.std(auc[:, 0]) * 100, 2)}")
    print(f'F1-score: {np.around(np.mean(f1[:, 0]) * 100, 2)} ± {np.around(np.std(f1[:, 0]) * 100, 2)}')
    print(f'ACC: {np.around(np.mean(acc[:, 0]) * 100, 2)} ± {np.around(np.std(acc[:, 0]) * 100, 2)}')
    print(f'Parity: {np.around(np.mean(parity[:, 0]) * 100, 2)} ± {np.around(np.std(parity[:, 0]) * 100, 2)}')
    print(f'Equality: {np.around(np.mean(equality[:, 0]) * 100, 2)} ± {np.around(np.std(equality[:, 0]) * 100, 2)}')
    
    file_path = f"./results/fdkd/{args.model}_{args.dataset}.txt"
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass
    
    with open(file_path, 'a') as f:
        f.write(f"τ={args.tem}, alpha={args.alpha}, lr={args.lr}, encoder={args.model}\n")
        f.write(f"AUCROC: {np.around(np.mean(auc[:, 0]) * 100, 2)} ± {np.around(np.std(auc[:, 0]) * 100, 2)}\n")
        f.write(f'F1-score: {np.around(np.mean(f1[:, 0]) * 100, 2)} ± {np.around(np.std(f1[:, 0]) * 100, 2)}\n')
        f.write(f'ACC: {np.around(np.mean(acc[:, 0]) * 100, 2)} ± {np.around(np.std(acc[:, 0]) * 100, 2)}\n')
        f.write(f'Parity: {np.around(np.mean(parity[:, 0]) * 100, 2)} ± {np.around(np.std(parity[:, 0]) * 100, 2)}\n')
        f.write(f'Equality: {np.around(np.mean(equality[:, 0]) * 100, 2)} ± {np.around(np.std(equality[:, 0]) * 100, 2)}\n')


# %%
