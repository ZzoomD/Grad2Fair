import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import grad
from torch.autograd import Function
import torch.distributed as dist
import os
from torch_geometric.nn import GCNConv, GINConv
import math

from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import fair_metric
import numpy as np


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.in_dim = args.in_dim
        self.hid_dim = args.hid_dim
        self.out_dim = args.out_dim
        self.args = args
        gnn_backbone = ConstructModel(args.in_dim, args.hid_dim, args.encoder, args.layer_num)
        self.gnn_backbone = gnn_backbone.to(args.device)
        self.classifier = nn.Linear(args.hid_dim, args.out_dim).to(args.device)
        
        self.optimizer = torch.optim.Adam(list(self.gnn_backbone.parameters())+list(self.classifier.parameters()),
                                          lr=args.lr, weight_decay=args.weight_decay)
        self.criterion_cls = nn.BCEWithLogitsLoss()
           
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def train_vanilla(self, data, **kwargs):
        best_loss = 100
        pbar = kwargs.get('pbar', None)
        enable_log = kwargs.get('enable_log', True)
        enable_shortcut = kwargs.get('enable_shortcut', False)
        select_ratio = kwargs.get('select_ratio', 0.5)
        save_path = kwargs.get('save_path', f'./weights/vanilla_{self.args.encoder}.pt')
        
        shortcut_start = 50 if self.args.epochs<=100 else 100
        for epoch in range(self.args.epochs):
            self.gnn_backbone.train()
            self.classifier.train()

            self.optimizer.zero_grad()

            emb = self.gnn_backbone(data.features, data.edge_index)
            output = self.classifier(emb)

            if enable_shortcut and epoch > shortcut_start:
                selected_num_half = int((select_ratio/2)*len(data.idx_train))
                train_out_sorted, ids = torch.sort(output[data.idx_train], dim=0, descending=True)
                ids = ids.squeeze()
                selected_ids = torch.cat((ids[:selected_num_half], ids[-selected_num_half:]))
                idx_train = data.idx_train[selected_ids.to(data.idx_train.device)]
                loss_cls = self.criterion_cls(output[idx_train], data.labels[idx_train].unsqueeze(1).float())
            else:
                loss_cls = self.criterion_cls(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())

            loss_cls.backward()
            self.optimizer.step()

            self.gnn_backbone.eval()
            self.classifier.eval()
            with torch.no_grad():
                emb_val = self.gnn_backbone(data.features, data.edge_index)
                output_val = self.classifier(emb_val)

            loss_cls_val = self.criterion_cls(output_val[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())
            pred = (output_val.squeeze() > 0).type_as(data.labels)
            # utility performance
            auc_val = roc_auc_score(data.labels[data.idx_val].cpu(), output_val[data.idx_val].cpu())
            f1_val = f1_score(data.labels[data.idx_val].cpu(), pred[data.idx_val].cpu())
            acc_val = accuracy_score(data.labels[data.idx_val].cpu(), pred[data.idx_val].cpu())
            # fairness performance
            parity_val, equality_val = fair_metric(pred[data.idx_val].cpu().numpy(),
                                                   data.labels[data.idx_val].cpu().numpy(),
                                                   data.sens[data.idx_val].cpu().numpy())
            
            if loss_cls_val.item() < best_loss:
                best_loss = loss_cls_val.item()
                torch.save(self.state_dict(), save_path)

            if hasattr(self.args, 'wandb_writer') and enable_log:
                # log training set loss
                self.args.wandb_writer.record({'epoch': epoch, 'train/loss_cls_'+self.args.seed: loss_cls.item()})
                # log validation set performance
                self.args.wandb_writer.record({'epoch': epoch, 'val/auc_'+self.args.seed: auc_val, 'val/f1_'+self.args.seed: f1_val, 'val/acc_'+self.args.seed: acc_val,
                                             'val/dp_'+self.args.seed: parity_val, 'val/eo_'+self.args.seed: equality_val})

            if pbar is not None:
                pbar.set_postfix({'loss_train': "{:.2f}".format(loss_cls.item())})
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
    
    def train_grad2fair(self, data, **kwargs):
        best_loss = 100
        pbar = kwargs.get('pbar', None)
        select_ratio = kwargs.get('select_ratio', 0.5)
        enable_shortcut = kwargs.get('enable_shortcut', True)

        self.train_vanilla(data, pbar=None, enable_log=False, enable_shortcut=enable_shortcut, select_ratio=select_ratio)
        pretrained_weight_path = f'./weights/vanilla_{self.args.encoder}.pt'
        
        full_state_dict = torch.load(pretrained_weight_path)

        gnn_backbone_state_dict = {k.replace('gnn_backbone.', ''): v for k, v in full_state_dict.items() if 'gnn_backbone.' in k}
        self.gnn_backbone.load_state_dict(gnn_backbone_state_dict, strict=False)
        
        classifier_state_dict = {k.replace('classifier.', ''): v for k, v in full_state_dict.items() if 'classifier.' in k}
        classifier_state_dict = {k: v for k, v in classifier_state_dict.items() if k != 'mask'}
        self.classifier.load_state_dict(classifier_state_dict, strict=False)

        data.features.requires_grad_(True)
        self.gnn_backbone.eval()
        self.classifier.eval()

        if data.features.grad is not None:
            data.features.grad.zero_()

        emb = self.gnn_backbone(data.features, data.edge_index)
        output = self.classifier(emb)
        pred = (output > 0).type_as(data.labels)

        idx = data.idx_train
        loss_train = self.criterion_cls(output[idx], data.labels[idx].unsqueeze(1).float())
        loss_train.backward()

        sam_grad_contr = torch.norm(data.features.grad[idx], p=2, dim=1)
        correct_mask = (pred[idx].squeeze() == data.labels[idx])
        incorrect_mask = ~correct_mask
        
        sam_grad_contr_scale = (sam_grad_contr - sam_grad_contr.min()) / (sam_grad_contr.max() - sam_grad_contr.min())
        sam_grad_contr_scale[correct_mask] = 1
        sam_grad_contr_scale[incorrect_mask] = 1 + self.args.alpha * sam_grad_contr_scale[incorrect_mask]
        criterion = nn.BCEWithLogitsLoss(weight=sam_grad_contr_scale.unsqueeze(1))

        for epoch in range(self.args.upweight_epochs):
            self.gnn_backbone.train()
            self.classifier.train()
            self.optimizer.zero_grad()

            emb = self.gnn_backbone(data.features, data.edge_index)
            output = self.classifier(emb)

            loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            loss_train.backward()
            self.optimizer.step()

            self.gnn_backbone.eval()
            self.classifier.eval()
            with torch.no_grad():
                emb_val = self.gnn_backbone(data.features, data.edge_index)
                output_val = self.classifier(emb_val)
                pred_val = (output_val.squeeze() > 0).type_as(data.labels)

            loss_val = self.criterion_cls(output_val[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())

            # utility performance
            auc_val = roc_auc_score(data.labels[data.idx_val].cpu(), output_val[data.idx_val].cpu())
            f1_val = f1_score(data.labels[data.idx_val].cpu(), pred_val[data.idx_val].cpu())
            acc_val = accuracy_score(data.labels[data.idx_val].cpu(), pred_val[data.idx_val].cpu())
            # fairness performance
            parity_val, equality_val = fair_metric(pred_val[data.idx_val].cpu().numpy(),
                                                   data.labels[data.idx_val].cpu().numpy(),
                                                   data.sens[data.idx_val].cpu().numpy())
            
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.state_dict(), f'./weights/{self.args.run_type}_{self.args.encoder}.pt')

            if hasattr(self.args, 'wandb_writer'):
                # log training set loss
                self.args.wandb_writer.record({'epoch': epoch, 'train/loss_train_'+self.args.seed: loss_train.item()})
                # log validation set performance
                self.args.wandb_writer.record({'epoch': epoch, 'val/loss_val_'+self.args.seed: loss_val.item(),
                                             'val/auc_'+self.args.seed: auc_val, 'val/f1_'+self.args.seed: f1_val, 'val/acc_'+self.args.seed: acc_val,
                                             'val/dp_'+self.args.seed: parity_val, 'val/eo_'+self.args.seed: equality_val})

            if pbar is not None:
                pbar.set_postfix({'loss_train': "{:.2f}".format(loss_train.item())})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    def forward(self, x, edge_index):
        emb = self.gnn_backbone(x, edge_index)
        output = self.classifier(emb)
        return emb, output

class ConstructModel(nn.Module):
    def __init__(self, in_dim, hid_dim, encoder, layer_num):
        super(ConstructModel, self).__init__()
        self.encoder = encoder
        
        if encoder == 'gcn':
            self.model = nn.ModuleList()
            for i in range(layer_num-1):
                if i == 0:
                    self.model.append(GCNConv(in_dim, hid_dim))
                else:
                    self.model.append(GCNConv(hid_dim, hid_dim))
        elif encoder == 'gin':
            self.model = GIN(nfeat=in_dim, nhid=hid_dim, dropout=0.5)

    def forward(self, x, edge_index):
        if self.encoder == 'gcn':
            h = x
            for i, layer in enumerate(self.model):
                h = layer(h, edge_index)
        elif self.encoder == 'gin':
            h = self.model(x, edge_index)
        return h

class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout): 
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(nfeat, nhid), 
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nhid), 
        )
        self.conv1 = GINConv(self.mlp1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, x, edge_index): 
        x = self.conv1(x, edge_index)
        return x
