#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_smiles_vocab(smiles_list: List[str]) -> Dict[str, int]:
    charset = set()
    for s in smiles_list:
        for ch in s:
            charset.add(ch)
    charset = sorted(list(charset))
    stoi = {"<PAD>": 0}
    for i, ch in enumerate(charset, start=1):
        stoi[ch] = i
    return stoi


def encode_smiles(s: str, stoi: Dict[str, int], max_len: int) -> np.ndarray:
    ids = [stoi.get(ch, 0) for ch in s]
    if len(ids) >= max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)




class SRMMPCharQMDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        stoi: Dict[str, int],
        max_len: int,
        qm_mean: np.ndarray,
        qm_std: np.ndarray,
        qm_cols: List[str],
        use_smiles_std: bool = True,
    ):

        self.stoi = stoi
        self.max_len = max_len
        self.qm_cols = qm_cols

        if use_smiles_std and "smiles_std" in df.columns:
            self.smiles_col = "smiles_std"
        elif "smiles" in df.columns:
            self.smiles_col = "smiles"
        else:
            raise ValueError("数据集中既没有 'smiles_std' 也没有 'smiles' 列。")

        df = df.copy()
        df = df[~df["SR-MMP"].isna()]

        self.labels = df["SR-MMP"].astype(float).values
        self.smiles_list = df[self.smiles_col].astype(str).tolist()

        qm_raw = df[self.qm_cols].astype(float).values  # (N, 2)
        col_mean = np.nanmean(qm_raw, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        idx_nan = ~np.isfinite(qm_raw)
        if idx_nan.any():
            qm_raw[idx_nan] = np.take(col_mean, np.where(idx_nan)[1])

        qm_std_safe = qm_std.copy()
        qm_std_safe[~np.isfinite(qm_std_safe)] = 1.0
        qm_std_safe[qm_std_safe == 0] = 1.0

        self.qm_values = ((qm_raw - qm_mean) / qm_std_safe).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        s = self.smiles_list[idx]
        y = self.labels[idx]

        ids = encode_smiles(s, self.stoi, self.max_len)
        qm = self.qm_values[idx]

        return (
            torch.from_numpy(ids),                   # (L,)
            torch.from_numpy(qm),                    # (2,)
            torch.tensor(y, dtype=torch.float32),    # ()
        )



class SRMMPCharQMBinary(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        qm_dim: int = 2,
        emb_dim: int = 64,
        rnn_hidden: int = 128,
        mlp_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
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

    def forward(self, smiles_ids: torch.Tensor, qm_feats: torch.Tensor) -> torch.Tensor:

        emb = self.embedding(smiles_ids)      # (B, L, E)
        out, h_n = self.gru(emb)             # h_n: (2, B, H)
        h_fwd = h_n[-2]                      # (B, H)
        h_bwd = h_n[-1]                      # (B, H)
        h_smiles = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2H)

        h_qm = self.qm_mlp(qm_feats)         # (B, mlp_hidden)

        h = torch.cat([h_smiles, h_qm], dim=-1)
        logit = self.head(h).squeeze(-1)
        return logit



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for smiles_ids, qm, y in loader:
        smiles_ids = smiles_ids.to(device)
        qm = qm.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(smiles_ids, qm)

        if not torch.isfinite(logits).all():
            print("[WARN] logits contains NaN/Inf, skip this batch.")
            continue

        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            print("[WARN] loss is NaN/Inf, skip this batch.")
            continue

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    if n_samples == 0:
        return float("nan")
    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    all_y = []
    all_p = []

    for smiles_ids, qm, y in loader:
        smiles_ids = smiles_ids.to(device)
        qm = qm.to(device)
        y = y.to(device)

        logits = model(smiles_ids, qm)
        prob = torch.sigmoid(logits)

        all_y.append(y.cpu().numpy())
        all_p.append(prob.cpu().numpy())

    if not all_y:
        return np.nan, np.nan

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)

    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if y_true.size == 0:
        return np.nan, np.nan

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = np.nan

    return auc, auprc


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    print(f"Loading CSV: {args.csv}")
    try:
        df = pd.read_csv(args.csv)
    except Exception:
        df = pd.read_csv(args.csv, sep=None, engine="python")
    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))

    required_cols = ["SR-MMP", "split", "xtb_gap_eV", "xtb_Etot_Ha"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"缺少必需列: {c}")

    df = df[~df["SR-MMP"].isna()].copy()

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        valid_df = df[df["split"] == "valid"].copy()
        test_df = df[df["split"] == "test"].copy()
    else:
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=args.seed, stratify=df["SR-MMP"]
        )
        valid_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df["SR-MMP"]
        )

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    qm_cols = ["xtb_gap_eV", "xtb_Etot_Ha"]

    print("\nNaN counts in QM columns:")
    for name, subdf in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        nan_cnt = subdf[qm_cols].isna().sum()
        print(f"  {name}:")
        for c in qm_cols:
            print(f"    {c}: {nan_cnt[c]} NaN / {len(subdf)} rows")

    qm_train = train_df[qm_cols].astype(float).values
    qm_mean = np.nanmean(qm_train, axis=0)
    qm_std = np.nanstd(qm_train, axis=0)

    qm_mean = np.where(np.isfinite(qm_mean), qm_mean, 0.0)
    qm_std = np.where(np.isfinite(qm_std) & (qm_std > 0), qm_std, 1.0)

    print("\nQM mean (train):", qm_mean)
    print("QM std  (train):", qm_std)

    if "smiles_std" in train_df.columns:
        smiles_train = train_df["smiles_std"].astype(str).tolist()
        use_smiles_std = True
    elif "smiles" in train_df.columns:
        smiles_train = train_df["smiles"].astype(str).tolist()
        use_smiles_std = False
    else:
        raise ValueError("既没有 'smiles_std' 也没有 'smiles' 列，无法进行 SMILES 编码。")

    stoi = build_smiles_vocab(smiles_train)
    vocab_size = len(stoi)
    print(f"\nSMILES vocab size: {vocab_size}")

    lengths = [len(s) for s in smiles_train]
    p95 = int(np.percentile(lengths, 95))
    max_len = min(args.max_len, max(p95, 10))
    print(f"SMILES max_len: {max_len} (95th percentile={p95}, user_max={args.max_len})")

    train_ds = SRMMPCharQMDataset(
        train_df, stoi, max_len, qm_mean, qm_std, qm_cols, use_smiles_std=use_smiles_std
    )
    valid_ds = SRMMPCharQMDataset(
        valid_df, stoi, max_len, qm_mean, qm_std, qm_cols, use_smiles_std=use_smiles_std
    )
    test_ds = SRMMPCharQMDataset(
        test_df, stoi, max_len, qm_mean, qm_std, qm_cols, use_smiles_std=use_smiles_std
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 模型
    model = SRMMPCharQMBinary(
        vocab_size=vocab_size,
        qm_dim=len(qm_cols),
        emb_dim=args.emb_dim,
        rnn_hidden=args.rnn_hidden,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    ).to(device)

    y_train = train_df["SR-MMP"].astype(float).values
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    if n_pos > 0:
        pos_weight = n_neg / max(n_pos, 1)
    else:
        pos_weight = 1.0
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    print(f"\npos_weight = {pos_weight:.4f} (neg={n_neg}, pos={n_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, "srmmp_smiles_qm_char_best.pt")

    best_val_auprc = -np.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auc, val_auprc = evaluate(model, valid_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auc={val_auc:.4f} | val_auprc={val_auprc:.4f}"
        )

        if np.isfinite(val_auprc) and (val_auprc > best_val_auprc):
            best_val_auprc = val_auprc
            best_state = model.state_dict()
            torch.save(best_state, best_model_path)

    print(f"\nBest val AUPRC = {best_val_auprc:.4f}, model saved to: {best_model_path}")

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    test_auc, test_auprc = evaluate(model, test_loader, device)
    print("\n=== Test metrics (SR-MMP, SMILES-char + QM) ===")
    print(f"AUC   : {test_auc:.4f}")
    print(f"AUPRC: {test_auprc:.4f}")

    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for smiles_ids, qm, y in test_loader:
            smiles_ids = smiles_ids.to(device)
            qm = qm.to(device)
            y = y.to(device)
            logits = model(smiles_ids, qm)
            prob = torch.sigmoid(logits)
            all_y.append(y.cpu().numpy())
            all_p.append(prob.cpu().numpy())
    y_test = np.concatenate(all_y)
    p_test = np.concatenate(all_p)

    npz_path = os.path.join(args.out_dir, "srmmp_smiles_qm_char_test_outputs.npz")
    np.savez(npz_path, y_true=y_test, y_prob=p_test)
    print(f"Saved test predictions to: {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SR-MMP 单任务：SMILES（字符编码）+ 量子特征 模型训练（无 RDKit，NaN 安全处理）"
    )
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 路径")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./output_srmmp_smiles_qm_char",
        help="输出目录（模型与预测）",
    )

    parser.add_argument("--emb_dim", type=int, default=64, help="SMILES 字符 embedding 维度")
    parser.add_argument("--rnn_hidden", type=int, default=128, help="BiGRU 隐状态维度")
    parser.add_argument("--mlp_hidden", type=int, default=128, help="MLP 隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 训练")

    parser.add_argument("--max_len", type=int, default=120, help="SMILES 最大长度（字符）")

    args = parser.parse_args()
    main(args)
    
