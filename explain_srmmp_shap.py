import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import shap
import matplotlib.pyplot as plt

class SRMMPCharQMBinary(nn.Module):
    def __init__(self, vocab_size, qm_dim=2, emb_dim=64, rnn_hidden=128, mlp_hidden=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.qm_mlp = nn.Sequential(
            nn.Linear(qm_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * rnn_hidden + mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, smiles_ids, qm_feats):
        emb = self.embedding(smiles_ids)
        out, h_n = self.gru(emb)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h_smiles = torch.cat([h_fwd, h_bwd], dim=-1)
        h_qm = self.qm_mlp(qm_feats)
        h = torch.cat([h_smiles, h_qm], dim=-1)
        logit = self.head(h).squeeze(-1)
        return logit


class QM_Part_Model(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.qm_mlp = original_model.qm_mlp
        self.head = original_model.head
    
    def forward(self, h_smiles, qm_feats):
        h_qm = self.qm_mlp(qm_feats)
        h = torch.cat([h_smiles, h_qm], dim=-1)
        return self.head(h)

def build_smiles_vocab(smiles_list):
    charset = set()
    for s in smiles_list:
        for ch in s: charset.add(ch)
    charset = sorted(list(charset))
    stoi = {"<PAD>": 0}
    for i, ch in enumerate(charset, start=1):
        stoi[ch] = i
    return stoi

def encode_smiles(s, stoi, max_len):
    ids = [stoi.get(ch, 0) for ch in s]
    if len(ids) >= max_len: ids = ids[:max_len]
    else: ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

class SimpleDataset(Dataset):
    def __init__(self, df, stoi, max_len, qm_mean, qm_std, qm_cols, use_smiles_std=True):
        smiles_col = "smiles_std" if use_smiles_std and "smiles_std" in df.columns else "smiles"
        self.smiles_list = df[smiles_col].astype(str).tolist()
        
        qm_raw = df[qm_cols].astype(float).values
        col_mean = np.nanmean(qm_raw, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        idx_nan = ~np.isfinite(qm_raw)
        if idx_nan.any():
            qm_raw[idx_nan] = np.take(col_mean, np.where(idx_nan)[1])
        
        qm_std_safe = qm_std.copy()
        qm_std_safe[qm_std_safe == 0] = 1.0
        self.qm_values = ((qm_raw - qm_mean) / qm_std_safe).astype(np.float32)
        
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self): return len(self.smiles_list)
    def __getitem__(self, idx):
        s = self.smiles_list[idx]
        ids = encode_smiles(s, self.stoi, self.max_len)
        qm = self.qm_values[idx]
        return torch.from_numpy(ids), torch.from_numpy(qm)

def main():
    parser = argparse.ArgumentParser(description="SR-MMP Quantum Features Explainability Analysis")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--model_path", type=str, default="./output_srmmp_smiles_qm_char/srmmp_smiles_qm_char_best.pt")
    parser.add_argument("--background_size", type=int, default=100, help="Number of samples for background")
    parser.add_argument("--test_size", type=int, default=200, help="Number of samples to explain")
    args = parser.parse_args()

    device = torch.device("cpu") # SHAP usually runs on CPU for compatibility or CUDA if handled carefully
    print(f"Running analysis on {device}...")

    df = pd.read_csv(args.csv)
    train_df = df[df['split'] == 'train'] if 'split' in df.columns else df.sample(frac=0.7, random_state=42)
    qm_cols = ["xtb_gap_eV", "xtb_Etot_Ha"]
    
    qm_train = train_df[qm_cols].astype(float).values
    qm_mean = np.nanmean(qm_train, axis=0)
    qm_std = np.nanstd(qm_train, axis=0)
    qm_mean = np.where(np.isfinite(qm_mean), qm_mean, 0.0)
    qm_std = np.where(np.isfinite(qm_std) & (qm_std > 0), qm_std, 1.0)

    smiles_col = "smiles_std" if "smiles_std" in df.columns else "smiles"
    stoi = build_smiles_vocab(train_df[smiles_col].astype(str).tolist())
    vocab_size = len(stoi)
    max_len = 120 

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}. Please train the model first.")
        return

    model = SRMMPCharQMBinary(vocab_size=vocab_size, qm_dim=2).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    bg_dataset = SimpleDataset(train_df.iloc[:args.background_size], stoi, max_len, qm_mean, qm_std, qm_cols)
    bg_loader = DataLoader(bg_dataset, batch_size=args.background_size, shuffle=False)
    bg_ids, bg_qm = next(iter(bg_loader))
    
    test_df = df[df['split'] == 'test'] if 'split' in df.columns else df.drop(train_df.index).sample(n=args.test_size)
    n_explain = min(len(test_df), args.test_size)
    explain_dataset = SimpleDataset(test_df.iloc[:n_explain], stoi, max_len, qm_mean, qm_std, qm_cols)
    explain_loader = DataLoader(explain_dataset, batch_size=n_explain, shuffle=False)
    explain_ids, explain_qm = next(iter(explain_loader))

    print("Pre-calculating SMILES embeddings...")
    with torch.no_grad():
        emb = model.embedding(bg_ids.to(device))
        _, h_n = model.gru(emb)
        bg_h_smiles = torch.cat([h_n[-2], h_n[-1]], dim=-1) # (N_bg, 2*hidden)

        emb = model.embedding(explain_ids.to(device))
        _, h_n = model.gru(emb)
        explain_h_smiles = torch.cat([h_n[-2], h_n[-1]], dim=-1) # (N_test, 2*hidden)

    part_model = QM_Part_Model(model).to(device)
    
    print("Running SHAP DeepExplainer (this might take a minute)...")
    explainer = shap.DeepExplainer(part_model, [bg_h_smiles, bg_qm.to(device)])
    
    shap_values = explainer.shap_values([explain_h_smiles, explain_qm.to(device)])

    qm_shap_vals = shap_values[1] 
    
    if len(qm_shap_vals.shape) == 3:
        qm_shap_vals = qm_shap_vals[:, :, 0] 
    qm_display = explain_qm.numpy() * qm_std + qm_mean
    
    feature_names = ["HOMO-LUMO Gap (eV)", "Total Energy (Ha)"]
    
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Summary Plot: Impact of Quantum Features on Toxicity")
    shap.summary_plot(qm_shap_vals, qm_display, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_qm.png", dpi=300)
    print("Saved shap_summary_qm.png")


    plt.figure(figsize=(8, 6))
    shap.dependence_plot(0, qm_shap_vals, qm_display, feature_names=feature_names, show=False)
    plt.title("SHAP Dependence: HOMO-LUMO Gap")
    plt.tight_layout()
    plt.savefig("shap_dependence_gap.png", dpi=300)
    print("Saved shap_dependence_gap.png")

if __name__ == "__main__":
    main()