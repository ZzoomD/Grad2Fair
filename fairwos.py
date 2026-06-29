import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import Data
from load_data import *
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import argparse
import os
import json
import time

torch.set_float32_matmul_precision("high")


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=True)
        # self.conv2 = GCNConv(hidden_channels, out_channels, cached=True, normalize=True)
        # self.dropout = dropout
    
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = F.relu(x, inplace=True)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.conv2(x, edge_index)
        return x

class EncoderClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, enc_out_channels, num_classes, dropout=0.3):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, enc_out_channels, dropout)
        self.cls = nn.Linear(enc_out_channels, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        logits = self.cls(z)
        return logits, z

class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_channels, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=True)
        # self.conv2 = GCNConv(hidden_channels, emb_channels, cached=True, normalize=True)
        self.cls = nn.Linear(emb_channels, num_classes)
        # self.dropout = dropout
    
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, return_emb=False):
        h = self.conv1(x, edge_index)
        # h = F.relu(h, inplace=True)
        # h = F.dropout(h, p=self.dropout, training=self.training)
        # h = self.conv2(h, edge_index)
        logits = self.cls(h)
        if return_emb:
            return logits, h
        return logits

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

class GINClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_channels, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GIN(in_channels, hidden_channels, dropout)
        # self.conv2 = GCNConv(hidden_channels, emb_channels, cached=True, normalize=True)
        self.cls = nn.Linear(emb_channels, num_classes)
        # self.dropout = dropout
    
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, return_emb=False):
        h = self.conv1(x, edge_index)
        # h = F.relu(h, inplace=True)
        # h = F.dropout(h, p=self.dropout, training=self.training)
        # h = self.conv2(h, edge_index)
        logits = self.cls(h)
        if return_emb:
            return logits, h
        return logits

def accuracy(logits, y, mask):
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == y[mask]).sum().item()
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return correct / total

def train_encoder(data, enc_hidden=64, enc_out=8, lr=1e-3, epochs=200, weight_decay=5e-4, dropout=0.3, device="cpu"):
    model = EncoderClassifier(data.features.shape[1], enc_hidden, enc_out, 1, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x, ei, y = data.features, data.edge_index, data.labels
    tr, va = data.idx_train, data.idx_val
    best, best_state = 100, None
    criterion_cls = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _ = model(x, ei)
        loss = criterion_cls(logits[tr], y[tr].unsqueeze(1).float())
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, ei)
            loss_val = criterion_cls(logits[va], y[va].unsqueeze(1).float())
        if loss_val < best:
            best, best_state = loss_val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, z = model(x, ei)
    return z.detach(), model

def discretize_by_median(X):
    med = X.median(dim=0).values
    S = (X > med).long()
    return S, med

def pretrain_classifier(data, X0, hidden=64, emb_dim=32, lr=1e-3, epochs=400, weight_decay=5e-4, dropout=0.3, device="cpu", encoder="gcn"):
    if encoder == "gcn":
        model = GCNClassifier(X0.shape[1], hidden, emb_dim, 1, dropout).to(device)
    elif encoder == "gin":
        model = GINClassifier(X0.shape[1], hidden, emb_dim, 1, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y = data.labels
    ei = data.edge_index
    tr, va = data.idx_train, data.idx_val
    best, best_state = 100, None
    criterion_cls = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X0, ei)
        loss = criterion_cls(logits[tr], y[tr].unsqueeze(1).float())
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(X0, ei)
            loss_val = criterion_cls(logits[va], y[va].unsqueeze(1).float())
        if loss_val < best:
            best, best_state = loss_val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, emb = model(X0, ei, return_emb=True)
        pred = (logits.squeeze() > 0).type_as(y)
    return model, emb.detach(), pred.detach()

@torch.no_grad()
def build_counterfactual_indices(emb, pred, Sbin, K=5, batch_size=4096):
    """
    Build counterfactual indices in batches to avoid OOM error.
    
    Args:
        emb: Embedding tensor of shape (N, D)
        pred: Predicted labels of shape (N,)
        Sbin: Sensitive attributes of shape (N, I)
        K: Number of counterfactuals to find per instance
        batch_size: Number of instances to process in each batch
    
    Returns:
        cf_idx: Counterfactual indices of shape (I, N, K)
    """
    N, I = Sbin.shape[0], Sbin.shape[1]
    cf_idx = torch.full((I, N, K), -1, dtype=torch.long, device=emb.device)
    
    # Process in batches to reduce memory usage
    for d in range(I):
        sd = Sbin[:, d]
        # Process each batch of instances
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_indices = torch.arange(batch_start, batch_end, device=emb.device)
            
            # Compute distances for this batch only
            batch_emb = emb[batch_start:batch_end]
            # Compute distances between batch embeddings and all embeddings
            # This creates a (batch_size, N) matrix instead of (N, N)
            batch_dist = torch.cdist(batch_emb, emb, p=2)
            
            # Process each instance in the batch
            for local_i, global_i in enumerate(batch_indices):
                # Find candidates with same prediction but different sensitive attribute
                cand = (pred == pred[global_i]) & (sd != sd[global_i])
                cand[global_i] = False  # Exclude self
                
                if cand.any():
                    # Get distances for valid candidates
                    dists = batch_dist[local_i, cand]
                    # Find K nearest neighbors
                    topk_size = min(K, dists.size(0))
                    topk = dists.topk(k=topk_size, largest=False)
                    # Map back to global indices
                    idx_pool = torch.where(cand)[0]
                    chosen = idx_pool[topk.indices]
                    # Store the chosen indices
                    cf_idx[d, global_i, :chosen.size(0)] = chosen
    
    return cf_idx

def kkt_update_lambdas(D_per_attr):
    I = D_per_attr.numel()
    D = D_per_attr.clone().detach()
    D_sorted, idx = torch.sort(D, descending=True)
    b = None
    for j in range(1, I + 1):
        num = 2.0 + D_sorted[j - 1 :].sum().item()
        denom = (I - j + 1)
        bj = -num / denom
        left = -float("inf") if j == 1 else -D_sorted[j - 2].item()
        right = -D_sorted[j - 1].item()
        if left <= bj <= right:
            b = bj
            break
    if b is None:
        num = 2.0 + D_sorted[-1].item()
        b = -num / 1.0
    lam = torch.clamp((-b - D) / 2.0, min=0.0)
    s = lam.sum()
    if s.item() <= 1e-12:
        lam = torch.full_like(D, 1.0 / I)
    else:
        lam = lam / s
    return lam

def fair_loss_and_dist(model, X0, edge_index, cf_idx, lambdas, train_mask=None):
    logits, emb = model(X0, edge_index, return_emb=True)
    N = emb.size(0)
    I, _, K = cf_idx.size()
    device = emb.device
    D_attr = torch.zeros(I, device=device)
    reg = torch.tensor(0.0, device=device)
    if train_mask is None:
        active = torch.ones(N, dtype=torch.bool, device=device)
    else:
        active = train_mask
    for d in range(I):
        cnt = 0
        s = torch.tensor(0.0, device=device)
        for i in train_mask.tolist():
            js = cf_idx[d, i]
            js = js[js >= 0]
            if js.numel() == 0:
                continue
            dif = emb[i].unsqueeze(0) - emb[js]
            dists = (dif * dif).sum(dim=1)
            s = s + dists.sum()
            reg = reg + lambdas[d] * dists.sum()
            cnt += js.numel()
        D_attr[d] = s / max(cnt, 1)
    return logits, reg, D_attr

def finetune_with_fairness(data, X0, base_model, Sbin, K=5, alpha=1.0, epochs=50, lam_l2=1.0, lr=1e-3, weight_decay=5e-4, device="cpu"):
    model = base_model.to(device)
    ei, y = data.edge_index, data.labels
    tr, va, te = data.idx_train, data.idx_val, data.idx_test
    criterion_cls = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        model.eval()
        logits, emb0 = model(X0, ei, return_emb=True)
        pred0 = (logits.squeeze() > 0).type_as(y)
    cf_idx = build_counterfactual_indices(emb0, pred0, Sbin, K=K)
    I = X0.shape[1]
    lambdas = torch.full((I,), 1.0 / I, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state = 100, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits, reg, D_attr = fair_loss_and_dist(model, X0, ei, cf_idx, lambdas, train_mask=tr)
        Lu = criterion_cls(logits[tr], y[tr].unsqueeze(1).float())
        loss = Lu + alpha * reg + lam_l2 * (lambdas @ lambdas)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.eval()
            logits = model(X0, ei)
            loss_val = criterion_cls(logits[va], y[va].unsqueeze(1).float())
            if loss_val < best:
                best, best_state = loss_val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            lambdas = kkt_update_lambdas(D_attr)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X0, ei)
        # tr_acc = accuracy(logits, y, tr)
        # va_acc = accuracy(logits, y, va)
        pred = (logits.cpu().squeeze() > 0).type_as(y)
        te_acc = accuracy_score(y[te].cpu(), pred[te].cpu())
        te_auc = roc_auc_score(y[te].cpu(), logits[te].cpu())
        te_f1 = f1_score(y[te].cpu(), pred[te].cpu())
        te_dp, te_eo = fair_metric(pred[te].cpu().numpy(), y[te].cpu().numpy(), data.sens[te].cpu().numpy())

    return model, lambdas, (te_acc, te_auc, te_f1, te_dp, te_eo)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def run_fairwos(args):
    seed_num = 5
    results = {"test_acc":[], "test_auc":[], "test_f1":[], "test_dp":[], "test_eo":[], "training_time":[]}
    for seed in range(seed_num):
        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = FairDataset(dataset=args.dataset_name, device=torch.device(device))
        dataset.load_data()
        X0, enc_model = train_encoder(dataset, enc_hidden=16, enc_out=16, lr=args.lr, epochs=200, weight_decay=1e-5, dropout=0.3, device=device)
        Sbin, _ = discretize_by_median(X0)
        clf, emb_init, pred_init = pretrain_classifier(dataset, X0, hidden=16, emb_dim=16, lr=args.lr, epochs=1000, weight_decay=1e-5, dropout=0.3, device=device, encoder=args.encoder)
        clf, lambdas, (te_acc, te_auc, te_f1, te_dp, te_eo) = finetune_with_fairness(dataset, X0, base_model=clf, Sbin=Sbin, K=args.K, alpha=args.alpha, epochs=15, lam_l2=1.0, lr=1e-3, weight_decay=1e-5, device=device)
        results["test_acc"].append(te_acc)
        results["test_auc"].append(te_auc)
        results["test_f1"].append(te_f1)
        results["test_dp"].append(te_dp)
        results["test_eo"].append(te_eo)
        del dataset
    print(f"test_acc: {np.mean(results['test_acc']):2f}±{np.std(results['test_acc']):2f}\n"
            f"test_auc: {np.mean(results['test_auc']):2f}±{np.std(results['test_auc']):2f}\n"
            f"test_f1: {np.mean(results['test_f1']):2f}±{np.std(results['test_f1']):2f}\n"
            f"test_dp: {np.mean(results['test_dp']):2f}±{np.std(results['test_dp']):2f}\n"
            f"test_eo: {np.mean(results['test_eo']):2f}±{np.std(results['test_eo']):2f}\n"
            f"lambdas: {lambdas.detach().cpu().tolist()}")

    save_path = os.path.join("./results/fairwos", f"{args.dataset_name}_alpha{args.alpha}_K{args.K}_lr{args.lr}_encoder{args.encoder}.json")
    args_dict = vars(args)
    with open(save_path, 'w') as file:
        json.dump(args_dict, file, indent=4)
        file.write('\n')

    with open(save_path, 'a') as file:
        ret_dict = {"AUC": f"{np.around(np.mean(results['test_auc']) * 100, 2)} ± {np.around(np.std(results['test_auc']) * 100, 2)}",
                    "F1": f"{np.around(np.mean(results['test_f1']) * 100, 2)} ± {np.around(np.std(results['test_f1']) * 100, 2)}",
                    "ACC": f"{np.around(np.mean(results['test_acc']) * 100, 2)} ± {np.around(np.std(results['test_acc']) * 100, 2)}",
                    "Parity": f"{np.around(np.mean(results['test_dp']) * 100, 2)} ± {np.around(np.std(results['test_dp']) * 100, 2)}",
                    "Equality": f"{np.around(np.mean(results['test_eo']) * 100, 2)} ± {np.around(np.std(results['test_eo']) * 100, 2)}"
                    }
        json.dump(ret_dict, file, indent=4, ensure_ascii=False)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'german', 'credit'])
    parser.add_argument('--alpha', type=float, default=0.01, help='The weights of the regularization term.')
    parser.add_argument('--K', type=int, default=1, help='The Number of graph counterfactuals.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--encoder', type=str, default='gcn', choices=['gcn', 'gin'], help='The encoder to use.')

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    args = args_parser()
    
    os.makedirs('./weights', exist_ok=True)
    os.makedirs('./results/fairwos', exist_ok=True)
    
    run_fairwos(args)