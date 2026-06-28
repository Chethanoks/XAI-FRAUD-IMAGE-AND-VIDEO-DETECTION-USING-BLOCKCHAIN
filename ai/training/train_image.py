"""
Training Script — CLIP Image Deepfake Detector
Uses SBI (Self-Blended Images) strategy for robustness.

Usage:
  python -m ai.training.train_image \
    --real_dir  datasets/real_faces \
    --fake_dir  datasets/fake_faces \
    --epochs    20 \
    --batch     16 \
    --output    checkpoints/clip_detector.pt

Dataset layout expected:
  datasets/
    real_faces/   (jpg/png images of real faces)
    fake_faces/   (jpg/png images of deepfake faces)
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import cv2
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai.models.clip_image_detector import CLIPDeepfakeDetector, SBIAugmenter
from ai.preprocessing.face_detector import FaceDetector, ImagePreprocessor


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DeepfakeImageDataset(Dataset):
    """
    Loads real + fake face images.
    Applies SBI augmentation on-the-fly to real images (50% probability)
    to create extra synthetic training samples.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, real_dir: str, fake_dir: str, augment_sbi: bool = True):
        self.real_paths   = self._collect(real_dir)
        self.fake_paths   = self._collect(fake_dir)
        self.augment_sbi  = augment_sbi
        self.sbi          = SBIAugmenter()
        self.face_detect  = FaceDetector()
        self.preprocessor = ImagePreprocessor()

        print(f"Dataset: {len(self.real_paths)} real, {len(self.fake_paths)} fake")

        # Build balanced index: 0 = real, 1 = fake
        self.items = (
            [(p, 0) for p in self.real_paths] +
            [(p, 1) for p in self.fake_paths]
        )
        random.shuffle(self.items)

    def _collect(self, directory: str):
        d = Path(directory)
        return [p for p in d.rglob("*") if p.suffix.lower() in self.IMG_EXTS]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]

        image = cv2.imread(str(path))
        if image is None:
            # Return zero tensor on bad file
            return torch.zeros(3, 224, 224), torch.tensor(0.0)

        # Face detection + alignment
        face = self.face_detect.detect_and_align(image)
        if face is None:
            face = cv2.resize(image, (224, 224))

        # SBI augmentation: randomly blend real face to create extra fake sample
        if label == 0 and self.augment_sbi and random.random() < 0.5:
            face  = self.sbi.generate_blended_fake(face)
            label = 1  # It's now a synthetic fake

        tensor = self.preprocessor.preprocess(face).squeeze(0)
        return tensor, torch.tensor(float(label))


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model
        self.model = CLIPDeepfakeDetector().to(self.device)

        # Dataset + split
        full_ds  = DeepfakeImageDataset(args.real_dir, args.fake_dir)
        val_size = int(len(full_ds) * 0.15)
        trn_size = len(full_ds) - val_size
        self.train_ds, self.val_ds = random_split(full_ds, [trn_size, val_size])

        self.train_dl = DataLoader(self.train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
        self.val_dl   = DataLoader(self.val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

        # Optimizer + scheduler
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        self.criterion = nn.BCELoss()

        # Output path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def train(self):
        best_val_acc = 0.0

        for epoch in range(1, self.args.epochs + 1):
            # ── Train ───────────────────────────────────────────────────────
            self.model.train()
            trn_loss, trn_correct, trn_total = 0.0, 0, 0

            for x, y in tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.args.epochs} [train]"):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                trn_loss    += loss.item() * x.size(0)
                trn_correct += ((pred > 0.5).float() == y).sum().item()
                trn_total   += x.size(0)

            trn_loss /= trn_total
            trn_acc   = trn_correct / trn_total

            # ── Validate ─────────────────────────────────────────────────────
            val_loss, val_acc = self._validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d} | "
                f"trn_loss={trn_loss:.4f} trn_acc={trn_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc":          val_acc,
                }, self.args.output)
                print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

        print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
        print(f"Model saved to: {self.args.output}")

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        loss_total, correct, total = 0.0, 0, 0

        for x, y in self.val_dl:
            x, y   = x.to(self.device), y.to(self.device)
            pred   = self.model(x)
            loss   = self.criterion(pred, y)
            loss_total += loss.item() * x.size(0)
            correct    += ((pred > 0.5).float() == y).sum().item()
            total      += x.size(0)

        return loss_total / total, correct / total


# ─── Entry ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CLIP image deepfake detector")
    p.add_argument("--real_dir", required=True,  help="Directory of real face images")
    p.add_argument("--fake_dir", required=True,  help="Directory of fake face images")
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--output",   default="checkpoints/clip_detector.pt")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    trainer = Trainer(args)
    trainer.train()
