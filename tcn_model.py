"""
Groundwater Level Prediction - TCN (Temporal Convolutional Network)
Dataset: 7 Hydrogeological Clusters, 3 Wells per Cluster (21 Wells Total)
Method: Dilated Causal Convolutions for Temporal Pattern Learning
"""

import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT_DIR = r'C:\Users\Amir\OneDrive - City University of Hong Kong - Student\Desktop\Input'
CLUSTERS_DIR = os.path.join(ROOT_DIR, 'Clusters')
METEO_DIR = os.path.join(ROOT_DIR, 'meteo_cache')
NQ_DIR = os.path.join(ROOT_DIR, 'nq_extracted')
OUTPUT_DIR = os.path.join(ROOT_DIR, '..', 'outputs_tcn')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 7
N_WELLS = 3
NQ_CSV_CLUSTERS = {1, 2, 5, 6}
USWWD_CLUSTERS = {0, 3, 4}

TARGET_COL = 'sl_lev_navd88'
METEO_COLS = ['precip_mm', 'tmax_C', 'tmin_C', 'pet_mm', 'srad_Wm2', 'vpd_kPa']

SEQ_FEAT_COLS = [TARGET_COL] + METEO_COLS + ['NQ_total', 'month_sin', 'month_cos']
N_SEQ_FEAT = len(SEQ_FEAT_COLS)

MEM_FEAT_COLS = ['WB_roll6', 'WB_roll12', 'WB_roll24',
                 'P_roll6', 'P_roll12',
                 'API',
                 'NQ_roll6', 'NQ_roll12']
N_MEM_FEAT = len(MEM_FEAT_COLS)

SEQ_LEN = 6
PRED_H = 6
MAX_WELLS = 7

NUM_FILTERS = 64
KERNEL_SIZE = 3
N_LAYERS = 2
DROPOUT = 0.2
LR = 3e-4
BATCH_SIZE = 32
EPOCHS = 200
GRAD_CLIP = 1.0
PATIENCE = 30

SEEDS = [42, 123, 456, 789, 999]
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def load_cluster_excel(cluster_id):
    file_path = os.path.join(CLUSTERS_DIR, f'cluster_{cluster_id}.xlsx')
    xls = pd.ExcelFile(file_path)
    
    wells = {}
    for i in range(3):
        sheet_name = f'USGS_level_well_{i}'
        if sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            wells[i] = df
    
    meteo = None
    if 'Meteo' in xls.sheet_names:
        meteo = pd.read_excel(file_path, sheet_name='Meteo')
    
    uswwd = None
    if USWWD_CLUSTERS and cluster_id in USWWD_CLUSTERS:
        try:
            sheets = [s for s in xls.sheet_names if s.startswith('USWWD_withdrawal')]
            if sheets:
                uswwd_dfs = [pd.read_excel(file_path, sheet_name=s) for s in sheets]
                uswwd = pd.concat(uswwd_dfs, axis=1)
        except:
            pass
    
    return wells, meteo, uswwd

def load_nq_csv(cluster_id):
    file_path = os.path.join(NQ_DIR, f'cluster_{cluster_id}_nq.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=[0])
    return None

class ClusterDataset(Dataset):
    def __init__(self, Xs, Xm, ys, cid):
        self.Xs = torch.tensor(Xs, dtype=torch.float32)
        self.Xm = torch.tensor(Xm, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.float32)
        self.cid = torch.tensor(cid, dtype=torch.long)
    
    def __len__(self):
        return len(self.ys)
    
    def __getitem__(self, idx):
        return self.Xs[idx], self.Xm[idx], self.ys[idx], self.cid[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=padding, dilation=dilation, bias=True)
        self.conv = nn.utils.weight_norm(self.conv)
        self.chomp = nn.ConstantPad1d((-padding, 0), 0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_final = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=3, dropout=0.5):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_outputs)
    
    def forward(self, x):
        y = self.network(x)
        y = y[:, :, -1]
        return self.fc(y)

class TCNMODEL(nn.Module):
    def __init__(self, seq_input_dim, mem_input_dim, num_filters, n_layers, dropout):
        super().__init__()
        
        num_channels = [num_filters] * n_layers
        self.tcn = TemporalConvNet(seq_input_dim, num_filters, num_channels, 
                                   kernel_size=3, dropout=dropout)
        
        self.head = nn.Sequential(
            nn.Linear(num_filters + mem_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, PRED_H)
        )
    
    def forward(self, xs, xm):
        B, W, T, F = xs.shape
        xs_flat = xs.reshape(B*W, T, F)
        xs_flat = xs_flat.transpose(1, 2)
        
        tcn_out = self.tcn(xs_flat)
        
        xm_flat = xm.reshape(B*W, -1)
        h_final = torch.cat([tcn_out, xm_flat], dim=1)
        
        out = self.head(h_final)
        return out.reshape(B, W, PRED_H)

def build_dataset(cluster_id):
    wells, meteo, uswwd = load_cluster_excel(cluster_id)
    nq = load_nq_csv(cluster_id) if cluster_id in NQ_CSV_CLUSTERS else None
    
    well_data = {}
    for w_id in range(3):
        if w_id not in wells:
            continue
        
        df = wells[w_id].copy()
        date_col = [c for c in df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in df.columns) else df.columns[0]
        level_col = [c for c in df.columns if 'level' in c.lower() or 'lev' in c.lower()][0] if any(c.lower() in TARGET_COL.lower() for c in df.columns) else df.columns[1]
        
        df['date'] = pd.to_datetime(df[date_col])
        df[TARGET_COL] = pd.to_numeric(df[level_col], errors='coerce')
        df = df.dropna(subset=[TARGET_COL])
        
        well_data[w_id] = df[['date', TARGET_COL]].set_index('date')
    
    if meteo is not None:
        date_col = [c for c in meteo.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in meteo.columns) else meteo.columns[0]
        meteo['date'] = pd.to_datetime(meteo[date_col])
        meteo = meteo.set_index('date')
    
    common_idx = meteo.index
    for w_id in well_data:
        common_idx = common_idx.intersection(well_data[w_id].index)
    
    for w_id in well_data:
        well_data[w_id] = well_data[w_id].reindex(common_idx).fillna(method='ffill')
    meteo = meteo.reindex(common_idx).fillna(method='ffill')
    
    for met_col in METEO_COLS:
        if met_col not in meteo.columns:
            meteo[met_col] = 0.0
    
    if nq is not None:
        nq['date'] = pd.to_datetime(nq.iloc[:, 0])
        nq = nq.set_index('date')
        nq_total = nq.sum(axis=1)
        nq_total = nq_total.reindex(common_idx).fillna(0)
    elif uswwd is not None:
        nq_total = uswwd.sum(axis=1)
        nq_total = nq_total.reindex(common_idx).fillna(0)
    else:
        nq_total = pd.Series(0.0, index=common_idx)
    
    meteo['NQ_total'] = nq_total
    meteo['month_sin'] = np.sin(2 * np.pi * meteo.index.month / 12)
    meteo['month_cos'] = np.cos(2 * np.pi * meteo.index.month / 12)
    
    for col in ['WB_roll6', 'WB_roll12', 'WB_roll24', 'P_roll6', 'P_roll12', 'API', 'NQ_roll6', 'NQ_roll12']:
        if col not in meteo.columns:
            meteo[col] = 0.0
    
    Xs, Xm, ys = [], [], []
    
    for t in range(len(meteo) - SEQ_LEN):
        seq = meteo.iloc[t:t+SEQ_LEN][SEQ_FEAT_COLS].values
        mem = meteo.iloc[t:t+SEQ_LEN][MEM_FEAT_COLS].values.mean(axis=0)
        
        targets = []
        valid = True
        for w_id in range(3):
            if w_id in well_data:
                target = well_data[w_id].iloc[t+SEQ_LEN:t+SEQ_LEN+PRED_H][TARGET_COL].values
                if len(target) == PRED_H:
                    targets.append(target)
                else:
                    valid = False
                    break
        
        if valid and len(targets) == 3:
            seq_padded = np.zeros((MAX_WELLS, SEQ_LEN, N_SEQ_FEAT))
            for w_id, target in enumerate(targets):
                seq_padded[w_id] = seq
            
            Xs.append(seq_padded)
            Xm.append(mem)
            ys.append(np.array(targets))
    
    return np.array(Xs), np.array(Xm), np.array(ys)

def train():
    all_data = []
    
    for cid in range(N_CLUSTERS):
        print(f'Loading cluster {cid}...')
        Xs, Xm, ys = build_dataset(cid)
        all_data.append((Xs, Xm, ys, cid))
    
    results = []
    
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f'\nSeed {seed}')
        
        for cid, (Xs, Xm, ys, _) in enumerate(all_data):
            if len(Xs) == 0:
                continue
            
            n = len(Xs)
            n_tr = int(n * TRAIN_RATIO)
            n_val = int(n * VAL_RATIO)
            
            X_tr, Xm_tr, y_tr = Xs[:n_tr], Xm[:n_tr], ys[:n_tr]
            X_val, Xm_val, y_val = Xs[n_tr:n_tr+n_val], Xm[n_tr:n_tr+n_val], ys[n_tr:n_tr+n_val]
            X_te, Xm_te, y_te = Xs[n_tr+n_val:], Xm[n_tr+n_val:], ys[n_tr+n_val:]
            
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            
            train_ds = ClusterDataset(X_tr, Xm_tr, y_tr, np.full(len(y_tr), cid))
            val_ds = ClusterDataset(X_val, Xm_val, y_val, np.full(len(y_val), cid))
            test_ds = ClusterDataset(X_te, Xm_te, y_te, np.full(len(y_te), cid))
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
            
            model = TCNMODEL(N_SEQ_FEAT, N_MEM_FEAT, NUM_FILTERS, N_LAYERS, DROPOUT).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model = None
            
            for epoch in range(EPOCHS):
                model.train()
                for Xs_b, Xm_b, ys_b, _ in train_loader:
                    Xs_b = Xs_b.to(device)
                    Xm_b = Xm_b.to(device)
                    ys_b = ys_b.to(device)
                    
                    optimizer.zero_grad()
                    preds = model(Xs_b, Xm_b)
                    loss = nn.MSELoss()(preds, ys_b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                
                scheduler.step()
                
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for Xs_b, Xm_b, ys_b, _ in val_loader:
                        Xs_b = Xs_b.to(device)
                        Xm_b = Xm_b.to(device)
                        ys_b = ys_b.to(device)
                        preds = model(Xs_b, Xm_b)
                        val_loss += nn.MSELoss()(preds, ys_b).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= PATIENCE:
                    break
            
            if best_model:
                model.load_state_dict(best_model)
            
            model.eval()
            test_preds = []
            test_targets = []
            with torch.no_grad():
                for Xs_b, Xm_b, ys_b, _ in test_loader:
                    Xs_b = Xs_b.to(device)
                    Xm_b = Xm_b.to(device)
                    preds = model(Xs_b, Xm_b).cpu().numpy()
                    test_preds.append(preds)
                    test_targets.append(ys_b.numpy())
            
            test_preds = np.concatenate(test_preds, axis=0)
            test_targets = np.concatenate(test_targets, axis=0)
            
            nse_scores = []
            for h in range(PRED_H):
                y_true = test_targets[:, :, h].flatten()
                y_pred = test_preds[:, :, h].flatten()
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999
                nse_scores.append(nse)
            
            mean_nse = np.mean(nse_scores)
            results.append({'cluster': cid, 'seed': seed, 'nse': mean_nse})
            print(f'  Cluster {cid}: NSE={mean_nse:.4f}')
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'tcn_results.csv'), index=False)
    print(f'\nResults saved to {OUTPUT_DIR}')

if __name__ == '__main__':
    train()
