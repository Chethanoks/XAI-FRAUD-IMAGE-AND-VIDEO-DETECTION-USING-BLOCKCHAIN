"""
Training Script — AltFreezing + LipForensics Video Detector
Implements the alternating-freeze strategy per epoch.

Usage:
  python -m ai.training.train_video \
    --video_dir  datasets/videos \
    --label_csv  datasets/labels.csv \
    --epochs     30 \
    --batch      4 \
    --output_alt checkpoints/altfreezing.pt \
    --output_lip checkpoints/lipforensics.pt

label_csv format (no header):
  video_path, label        (label: 0=real, 1=fake)
"""

import os
import argparse
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai.models.altfreezing_video  import AltFreezingDetector
from ai.models.lipforensics        import LipForensicsNet, MouthROIExtractor
from ai.preprocessing.face_detector import FaceDetector
from ai.preprocessing.frame_extractor import FrameExtractor


# ─── Dataset ──────────────────────────────────────────────────────────────────

class VideoDeepfakeDataset(Dataset):
    """
    Loads video clips and returns:
      - rep_frame  : single representative frame tensor (3, 224, 224)
      - frame_seq  : sequence tensor (T, 3, 224, 224) for temporal stream
      - mouth_seq  : mouth sequence tensor (3, T, 88, 88) for LipForensics
      - label      : 0.0 or 1.0
    """

    CLIP_LEN = 8   # frames per clip for AltFreezing
    LIP_LEN  = 25  # frames per clip for LipForensics

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, label_csv: str, max_videos: int = None):
        import cv2 as _cv2
        self._cv2 = _cv2

        self.items = []
        with open(label_csv, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    self.items.append((row[0].strip(), int(row[1].strip())))

        if max_videos:
            self.items = self.items[:max_videos]

        random.shuffle(self.items)
        self.extractor     = FrameExtractor(max_frames=60)
        self.face_detector = FaceDetector()
        self.mouth_extract = MouthROIExtractor()
        print(f"Video dataset: {len(self.items)} videos")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        zero_alt    = (torch.zeros(3, 224, 224), torch.zeros(self.CLIP_LEN, 3, 224, 224))
        zero_lip    = torch.zeros(3, self.LIP_LEN, 88, 88)
        zero_label  = torch.tensor(float(label))

        try:
            extraction  = self.extractor.extract(path)
            raw_frames  = extraction["frames"]
            if len(raw_frames) < 2:
                return zero_alt[0], zero_alt[1], zero_lip, zero_label

            face_crops = []
            for f in raw_frames:
                face = self.face_detector.detect_and_align(f)
                if face is not None:
                    face_crops.append(face)

            if len(face_crops) < max(self.CLIP_LEN, self.LIP_LEN):
                # Pad by repeating last frame
                while len(face_crops) < max(self.CLIP_LEN, self.LIP_LEN):
                    face_crops.append(face_crops[-1])

            # AltFreezing inputs
            clip       = face_crops[:self.CLIP_LEN]
            tensors    = [self._to_tensor(f) for f in clip]
            frame_seq  = torch.stack(tensors)          # (T, 3, 224, 224)
            rep_frame  = tensors[len(tensors) // 2]    # (3, 224, 224)

            # LipForensics mouth inputs
            lip_clip   = face_crops[:self.LIP_LEN]
            mouth_seq  = self.mouth_extract.extract_sequence(
                [f[:, :, ::-1].copy() for f in lip_clip]  # BGR→RGB for extractor
            )  # (T, 88, 88, 3)
            lip_tensor = self._mouth_to_tensor(mouth_seq)  # (3, T, 88, 88)

            return rep_frame, frame_seq, lip_tensor, zero_label.fill_(label)

        except Exception:
            return zero_alt[0], zero_alt[1], zero_lip, zero_label

    def _to_tensor(self, face_bgr):
        import cv2
        rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res  = cv2.resize(rgb, (224, 224))
        t    = torch.from_numpy(res).permute(2, 0, 1).float() / 255.0
        return (t - self.MEAN) / self.STD

    def _mouth_to_tensor(self, mouth_seq):
        """mouth_seq: (T, 88, 88, 3) RGB uint8 → (3, T, 88, 88) normalized"""
        MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        t    = torch.from_numpy(mouth_seq).float() / 255.0  # (T, 88, 88, 3)
        t    = t.permute(3, 0, 1, 2)                        # (3, T, 88, 88)
        return (t - MEAN) / STD


# ─── AltFreezing Trainer ──────────────────────────────────────────────────────

class AltFreezingTrainer:
    def __init__(self, args, train_dl, val_dl):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = AltFreezingDetector().to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion = nn.BCELoss()
        self.train_dl  = train_dl
        self.val_dl    = val_dl

        Path(args.output_alt).parent.mkdir(parents=True, exist_ok=True)

    def train(self):
        best_acc = 0.0
        for epoch in range(1, self.args.epochs + 1):
            # Alternate freezing per epoch
            self.model.set_training_mode(epoch)
            self.model.train()

            loss_sum, correct, total = 0.0, 0, 0
            for rep, seq, _, y in tqdm(self.train_dl, desc=f"AltFreezing Epoch {epoch}"):
                rep, seq, y = rep.to(self.device), seq.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                prob, _, _ = self.model(rep, seq)
                loss       = self.criterion(prob, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                loss_sum += loss.item() * rep.size(0)
                correct  += ((prob > 0.5).float() == y).sum().item()
                total    += rep.size(0)

            val_acc = self._validate()
            self.scheduler.step()

            stream = "spatial" if epoch % 2 == 0 else "temporal"
            print(f"  [AltFreezing] Epoch {epoch} | stream={stream} | trn_acc={correct/total:.4f} | val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model_state_dict": self.model.state_dict()}, self.args.output_alt)
                print(f"    ✓ Saved (val_acc={val_acc:.4f})")

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        correct, total = 0, 0
        for rep, seq, _, y in self.val_dl:
            rep, seq, y = rep.to(self.device), seq.to(self.device), y.to(self.device)
            prob, _, _  = self.model(rep, seq)
            correct     += ((prob > 0.5).float() == y).sum().item()
            total       += rep.size(0)
        return correct / max(total, 1)


# ─── LipForensics Trainer ─────────────────────────────────────────────────────

class LipForensicsTrainer:
    def __init__(self, args, train_dl, val_dl):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = LipForensicsNet().to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion = nn.BCELoss()
        self.train_dl  = train_dl
        self.val_dl    = val_dl

        Path(args.output_lip).parent.mkdir(parents=True, exist_ok=True)

    def train(self):
        best_acc = 0.0
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            loss_sum, correct, total = 0.0, 0, 0

            for _, _, lip, y in tqdm(self.train_dl, desc=f"LipForensics Epoch {epoch}"):
                lip, y = lip.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                prob, _ = self.model(lip)
                loss    = self.criterion(prob, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                loss_sum += loss.item() * lip.size(0)
                correct  += ((prob > 0.5).float() == y).sum().item()
                total    += lip.size(0)

            val_acc = self._validate()
            self.scheduler.step()
            print(f"  [LipForensics] Epoch {epoch} | trn_acc={correct/total:.4f} | val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model_state_dict": self.model.state_dict()}, self.args.output_lip)
                print(f"    ✓ Saved (val_acc={val_acc:.4f})")

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        correct, total = 0, 0
        for _, _, lip, y in self.val_dl:
            lip, y  = lip.to(self.device), y.to(self.device)
            prob, _ = self.model(lip)
            correct += ((prob > 0.5).float() == y).sum().item()
            total   += lip.size(0)
        return correct / max(total, 1)


# ─── Entry ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train video deepfake detectors")
    p.add_argument("--label_csv",   required=True, help="CSV file: video_path,label")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch",       type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--output_alt",  default="checkpoints/altfreezing.pt")
    p.add_argument("--output_lip",  default="checkpoints/lipforensics.pt")
    p.add_argument("--max_videos",  type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Shared dataset + split
    full_ds  = VideoDeepfakeDataset(args.label_csv, max_videos=args.max_videos)
    val_size = max(1, int(len(full_ds) * 0.15))
    trn_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [trn_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    print("\n=== Training AltFreezing ===")
    AltFreezingTrainer(args, train_dl, val_dl).train()

    print("\n=== Training LipForensics ===")
    LipForensicsTrainer(args, train_dl, val_dl).train()

    print("\nAll models trained successfully.")
