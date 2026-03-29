#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import numpy as np
import librosa

import torch
import torch.nn as nn


# ============================================================
# 1) Model definition (must match training architecture)
# ============================================================
class SpectrumClassifier(nn.Module):
    def __init__(self, num_classes, dropout):
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
        self.fc = nn.Sequential(
            nn.Linear(128 * (128 // 8) * (128 // 8), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
# ============================================================
# 2) Feature extraction (must match training feature pipeline)
# ============================================================
def extract_audio_features(file_path: str, hop_length=512, target_sr=None):
    """
    Extract spectrogram features from audio file.

    Output shape:
      (1, 128, 128)  -> single-channel spectrogram image

    This matches the CNN training pipeline.
    """
    # Load audio waveform
    y, sr = librosa.load(file_path, sr=target_sr)

    # Compute mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)

    # Convert to log scale (dB)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Ensure fixed size (128 x 128)
    if S_db.shape[1] < 128:
        S_db = np.pad(S_db, ((0, 0), (0, 128 - S_db.shape[1])), mode="constant")
    else:
        S_db = S_db[:, :128]

    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    # Add channel dimension -> (1, 128, 128)
    return S_db[np.newaxis, :, :].astype(np.float32)
# ============================================================
# 3) Main inference flow
# ============================================================
def main():
    # --------------------------
    # (A) Parse command line args
    # --------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True, help="e.g. checkpoint_csv/checkpoint_best.pt")
    ap.add_argument("--out_csv", type=str, default="pred.csv")
    ap.add_argument("--dropout", type=float, default=0.5)
    # Feature parameters:
    # MUST match training configuration, otherwise model sees different input distribution/shape.
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--target_sr", type=int, default=0)  # 0 means librosa default (we convert to None)

    # Model parameters:

    # If True, missing audio files will raise an error and stop inference.
    # If False, we will provide a fallback prediction.
    ap.add_argument("--fail_on_missing", action="store_true")
    args = ap.parse_args()

    # Convert target_sr argument:
    # - args.target_sr == 0 => None (librosa default behavior)
    # - else => resample to that sampling rate
    target_sr = None if args.target_sr == 0 else int(args.target_sr)

    # --------------------------
    # (B) Load checkpoint and label mapping
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # map_location ensures checkpoint can be loaded on CPU/GPU safely
    ckpt = torch.load(args.ckpt_path, map_location=device)

    # The training script saves idx2label inside checkpoint_best.pt.
    # We use it to translate predicted class index back to human-readable label.
    if isinstance(ckpt, dict) and "idx2label" in ckpt:
        # JSON-like dict sometimes stores keys as strings, so we force int keys here
        idx2label = {int(k): v for k, v in ckpt["idx2label"].items()}
        num_classes = len(idx2label)
    else:
        raise ValueError("Checkpoint missing idx2label. Please use checkpoint produced by train_from_csv.py")

    # --------------------------
    # (C) Rebuild model and load weights
    # --------------------------
    model = SpectrumClassifier(num_classes=num_classes, dropout=args.dropout).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # --------------------------
    # (D) Read CSV and select test rows
    # --------------------------
    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]

    # normalize strings to avoid issues like extra spaces
    df["ID"] = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()

    # Only predict for set == "test"
    test_df = df[df["set"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test rows found (set=test).")

    # --------------------------
    # (E) Predict each test file
    # --------------------------
    preds = []
    for audio_id in test_df["ID"].tolist():
        path = os.path.join(args.audio_root, audio_id)

        if not os.path.isfile(path):
            if args.fail_on_missing:
                raise FileNotFoundError(f"Missing audio file: {path}")
            pred_label = idx2label[0]
            preds.append((audio_id, pred_label))
            continue

    # Extract spectrogram features -> (1,128,128)
        feats = extract_audio_features(path, hop_length=args.hop_length, target_sr=target_sr)

    # Convert to tensor and add batch dimension -> (B,1,128,128)
        x = torch.from_numpy(feats).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_label = idx2label[pred_idx]

        preds.append((audio_id, pred_label))

    # --------------------------
    # (F) Save prediction CSV
    # --------------------------
    out = pd.DataFrame(preds, columns=["ID", "label"])

    # Kaggle usually expects: ID,label (no index column)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] saved: {args.out_csv}  rows={len(out)}")


if __name__ == "__main__":
    # Script entry point
    main()