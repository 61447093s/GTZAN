#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib.resources import path
import os
import json
import math
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) Model definition
# ============================================================
class SpectrumClassifier(nn.Module):
    def __init__(self, num_classes,dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # hardcode spectrogram size: 128 x 128
        self.fc = nn.Sequential(
            nn.Linear(128 * (128//8) * (128//8), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # expect input shape: (batch, 1, 128, 128)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @staticmethod
    def accuracy(logits, target_idx):
        pred = torch.argmax(logits, dim=1)
        return (pred == target_idx).float().mean().item() * 100.0


# ============================================================
# 2) Feature extraction (audio -> fixed-length sequence)
# ============================================================
def extract_spectrogram(path, hop_length=512, target_sr=22050):
    # load audio
    y, sr = librosa.load(path, sr=target_sr)

    # compute mel spectrogram with fixed 128 bands
    S = librosa.feature.melspectrogram(y=y, sr=sr,
                                       n_mels=128,
                                       hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)

    # pad or truncate to fixed width 128
    if log_S.shape[1] < 128:
        pad_width = 128 - log_S.shape[1]
        log_S = np.pad(log_S, ((0,0),(0,pad_width)), mode='constant')
    else:
        log_S = log_S[:, :128]

    log_S = (log_S - log_S.mean()) / (log_S.std() + 1e-6)
    # final shape: (128, 128)
    return log_S.astype(np.float32)

# ============================================================
# 3) Dataset: reading CSV rows and loading audio
# ============================================================
class AudioCSVDataset(Dataset):
    """
    A PyTorch Dataset that:
      - reads one row at a time from a dataframe
      - resolves the audio file path
      - extracts spectrogram features (128 x 128)
      - returns tensors for training or inference

    For train/val:
      returns (audio_id, x, y)
    For test:
      returns (audio_id, x)   (no label)
    """
    def __init__(
        self,
        df: "pd.DataFrame",
        audio_root: str,
        label2idx: Dict[str, int] | None,
        hop_length: int,
        target_sr: int | None,
        cache_features: bool = False,
        fail_on_missing: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.audio_root = audio_root
        self.label2idx = label2idx
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.cache_features = cache_features
        self.fail_on_missing = fail_on_missing

        # simple in-memory cache: idx -> torch.Tensor (1,128,128)
        self._cache: Dict[int, torch.Tensor] = {}

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, audio_id: str) -> str:
        return os.path.join(self.audio_root, audio_id)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_id = str(row["ID"]).strip()

        label_str = None if pd.isna(row.get("label", np.nan)) else str(row.get("label", "")).strip()
        path = self._resolve_path(audio_id)

        if not os.path.isfile(path):
            if self.fail_on_missing:
                raise FileNotFoundError(f"Missing audio file: {path}")
            feats = np.zeros((128, 128), dtype=np.float32)
            x = torch.from_numpy(feats).unsqueeze(0)  # (1, 128, 128)
        else:
            if self.cache_features and idx in self._cache:
                x = self._cache[idx]
            else:
                feats = extract_spectrogram(path, hop_length=self.hop_length, target_sr=self.target_sr)
                x = torch.from_numpy(feats).unsqueeze(0)  # (1, 128, 128)
                if self.cache_features:
                    self._cache[idx] = x

        if self.label2idx is None:
            return audio_id, x

        if label_str not in self.label2idx:
            raise ValueError(f"Label '{label_str}' not in label2idx. Check CSV label normalization.")
        y = torch.tensor(self.label2idx[label_str], dtype=torch.long)

        return audio_id, x, y

# ============================================================
# 4) Collate functions: how to batch variable items
# ============================================================
def collate_train(batch):
    """
    Batch builder for train/val.

    Input batch items: List[(id, x, y)]
    x: (1, 128, 128)

    Output:
    ids: list of strings length B
    xs: tensor (B, 1, 128, 128)  <- CNN expects batch-first format
    ys: tensor (B,)
    """
    ids = [b[0] for b in batch]
    xs = torch.stack([b[1] for b in batch], dim=0)  # (B, 1, 128, 128)
    ys = torch.stack([b[2] for b in batch], dim=0)  # (B,)
    return ids, xs, ys


def collate_test(batch):
    """
    Batch builder for test/inference.

    Input items: List[(id, x)]
    x: (1, 128, 128)

    Output:
    ids: list[str]
    xs: tensor (B, 1, 128, 128)
    """
    ids = [b[0] for b in batch]
    xs = torch.stack([b[1] for b in batch], dim=0)  # (B, 1, 128, 128)
    return ids, xs

# ============================================================
# 5) Training utilities and hyper-parameter container
# ============================================================
@dataclass
class HParams:
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    validate_every: int
    hop_length: int
    target_sr: int | None
    num_workers: int
    seed: int


def set_seed(seed: int):
    """
    Make training deterministic-ish (still not perfectly deterministic on GPU, but better).
    Fixes random seeds for:
      - python random
      - numpy
      - torch CPU/GPU
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_map(df: "pd.DataFrame") -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build mapping from string labels to integer indices.

    IMPORTANT:
    We only build from TRAIN labels.
    Why?
      - Avoid "data leakage" from val/test labels
      - In competitions, test labels may be hidden

    Returns:
      label2idx: e.g. {"disco":0, "jazz":1, ...}
      idx2label: reverse mapping
    """
    labels = sorted({str(x).strip() for x in df["label"].dropna().tolist()})
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label


def run_eval(model, loader, device):
    """
    Evaluate on validation set.

    Steps:
      - model.eval() disables dropout etc.
      - torch.no_grad() disables gradient tracking (faster + less memory)
      - compute average loss and average accuracy across batches
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for _, x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            total_acc += SpectrumClassifier.accuracy(logits, y)
            n_batches += 1

    if n_batches == 0:
        return math.nan, math.nan
    return total_loss / n_batches, total_acc / n_batches


# ============================================================
# 6) Main training script (argument parsing + training loop)
# ============================================================
def main():
    # --------------------------
    # (A) Parse command line args
    # --------------------------
    ap = argparse.ArgumentParser()

    # data paths
    ap.add_argument("--csv_path", type=str, required=True, help="CSV with columns: ID,label,set")
    ap.add_argument("--audio_root", type=str, required=True, help="Root folder containing audio files")
    ap.add_argument("--out_dir", type=str, default="checkpoint_csv", help="Output directory for checkpoints & maps")

    # model hyperparams (external adjustable)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=100)

    # optimization hyperparams
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--validate_every", type=int, default=1, help="Validate every N epochs")

    # feature hyperparams (external adjustable)
    ap.add_argument("--hop_length", type=int, default=512)

    # For target_sr:
    # - If 0: use librosa default behavior (sr=None in our code becomes None => default 22050)
    # - Else: resample audio to this sampling rate
    ap.add_argument("--target_sr", type=int, default=0, help="0 means librosa default; else resample to this SR")

    # dataloader settings
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_features", action="store_true", help="Cache extracted features in RAM (faster, more RAM)")
    ap.add_argument("--fail_on_missing", action="store_true", help="Raise error if any audio file missing")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Convert target_sr from int argument to either None or actual sr
    target_sr = None if args.target_sr == 0 else int(args.target_sr)

    # Store all hyperparams in a single object (easy to save)
    hp = HParams(
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        validate_every=args.validate_every,
        hop_length=args.hop_length,
        target_sr=target_sr,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # --------------------------
    # (B) Setup output folder + seed
    # --------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(hp.seed)

    # --------------------------
    # (C) Read CSV and split sets
    # --------------------------
    df = pd.read_csv(args.csv_path)

    # Normalize column names (trim spaces)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    if "ID" not in df.columns or "set" not in df.columns:
        raise ValueError("CSV must include columns: ID, set, and (for train/val) label.")
    if "label" not in df.columns:
        # If label column missing, create it (mostly for inference-only cases)
        df["label"] = np.nan

    # Normalize string values
    df["ID"] = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip()

    # Split by "set" column
    train_df = df[df["set"] == "train"].copy()
    val_df = df[df["set"].isin(["val", "dev", "valid", "validation"])].copy()
    test_df = df[df["set"] == "test"].copy()

    if len(train_df) == 0:
        raise ValueError("No train rows found (set=train).")
    if len(val_df) == 0:
        print("[WARN] No val/dev rows found. Validation will be skipped, and best ckpt won't be meaningful.")

    # Build label mapping from training data
    label2idx, idx2label = build_label_map(train_df)
    num_classes = len(label2idx)

    # Save label mapping + hyperparams for reproducibility
    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(hp) | {"num_classes": num_classes}, f, ensure_ascii=False, indent=2)

    # --------------------------
    # (D) Device selection (GPU if available)
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Classes] {num_classes}: {list(label2idx.keys())}")

    # --------------------------
    # (E) Build datasets + dataloaders
    # --------------------------
    train_ds = AudioCSVDataset(
        train_df,
        args.audio_root,
        label2idx,
        hop_length=hp.hop_length,
        target_sr=hp.target_sr,
        cache_features=args.cache_features,
        fail_on_missing=args.fail_on_missing
    )

    # shuffle=True for training (important!)
    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=hp.num_workers,
        collate_fn=collate_train,
        drop_last=True  # drop last incomplete batch (keeps batch size consistent)
    )

    val_loader = None
    if len(val_df) > 0:
        val_ds = AudioCSVDataset(
            val_df,
            args.audio_root,
            label2idx,
            hop_length=hp.hop_length,
            target_sr=hp.target_sr,
            cache_features=args.cache_features,
            fail_on_missing=args.fail_on_missing
        )

        # shuffle=False for validation
        val_loader = DataLoader(
            val_ds,
            batch_size=hp.batch_size,
            shuffle=False,
            num_workers=hp.num_workers,
            collate_fn=collate_train,
            drop_last=True
        )

    # --------------------------
    # (F) Build model, loss, optimizer
    # --------------------------
    model = SpectrumClassifier(num_classes=num_classes, dropout=hp.dropout).to(device)

    # CrossEntropyLoss expects:
    # - logits (B, C) (no softmax needed)
    # - targets (B,) class indices
    loss_fn = nn.CrossEntropyLoss()

    # Adam optimizer is a strong default for many problems
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay
    )

    # Track best validation accuracy and save best checkpoint
    best_val_acc = -1.0
    best_path = os.path.join(args.out_dir, "checkpoint_best.pt")

    # --------------------------
    # (G) Training loop (epoch-based)
    # --------------------------
    for epoch in range(1, hp.epochs + 1):
        model.train()  # enable training behaviors like dropout
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        # Iterate batches
        for _, x, y in train_loader:
            x = x.to(device)  # (128, B, 33)
            y = y.to(device)  # (B,)

            optimizer.zero_grad()

            logits = model(x)        # logits: (B, num_classes)
            loss = loss_fn(logits, y)   # scalar loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += SpectrumClassifier.accuracy(logits, y)
            n_batches += 1

        # Average metrics over batches
        train_loss = running_loss / max(n_batches, 1)
        train_acc = running_acc / max(n_batches, 1)

        msg = f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_acc={train_acc:.2f}"

        # --------------------------
        # (H) Validation (optional)
        # --------------------------
        do_val = (val_loader is not None) and (epoch % hp.validate_every == 0)
        if do_val:
            val_loss, val_acc = run_eval(model, val_loader, device)
            msg += f" | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}"

            # If this epoch achieves better val accuracy, save it as "best"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "label2idx": label2idx,
                        "idx2label": idx2label,
                        "hparams": asdict(hp) | {"num_classes": num_classes},
                        "best_val_acc": best_val_acc,
                        "epoch": epoch,
                    },
                    best_path,
                )
                msg += f"\n[BEST] -> saved {os.path.basename(best_path)}"

        print(msg)

        # --------------------------
        # (I) Periodic checkpoint saving (every 10 epochs)
        # --------------------------
        # Useful when:
        # - training crashes midway
        # - you want to compare intermediate models
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)

    # --------------------------
    # (J) Training summary
    # --------------------------
    print(f"[Done] Best val acc: {best_val_acc:.2f} | ckpt: {best_path}")
    print(f"[Note] test rows in CSV = {len(test_df)} (test is not used during training).")


if __name__ == "__main__":
    # Entry point: running "python train.py ..." will call main()
    main()