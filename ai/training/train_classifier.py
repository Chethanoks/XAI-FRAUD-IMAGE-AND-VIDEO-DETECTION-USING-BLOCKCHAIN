"""
Train deepfake image classifier using EfficientNet-B4.
Fixed: handles truncated/corrupted images gracefully.
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile

# ── Fix truncated images (corrupted files won't crash training) ───────────────
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# ── Model ─────────────────────────────────────────────────────────────────────

class DeepfakeClassifier(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b4(
            weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.backbone(x)).squeeze(-1)


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms():
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ── Safe dataset that skips bad images ────────────────────────────────────────

class SafeImageFolder(ImageFolder):
    """ImageFolder that skips corrupted/unreadable images instead of crashing."""

    def __getitem__(self, index):
        # Try up to 5 times with different indices if current one fails
        for attempt in range(5):
            try:
                return super().__getitem__((index + attempt) % len(self))
            except Exception as e:
                if attempt == 0:
                    print(f"\n[Warning] Skipping corrupted image at index {index}: {e}")
                continue
        # Return a black image as last resort
        dummy = torch.zeros(3, 224, 224)
        return dummy, 0


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda")
        print(f"Device: {self.device} ({torch.cuda.get_device_name(0)})")

        train_tf, val_tf = get_transforms()

        # Use SafeImageFolder instead of ImageFolder
        full_ds = SafeImageFolder(args.dataset, transform=train_tf)
        print(f"Classes: {full_ds.classes}")
        print(f"Total images: {len(full_ds)}")

        class_counts = np.bincount(full_ds.targets)
        print(f"Class distribution: {dict(zip(full_ds.classes, class_counts))}")

        val_size = max(50, int(len(full_ds) * 0.15))
        trn_size = len(full_ds) - val_size
        self.train_ds, self.val_ds = random_split(full_ds, [trn_size, val_size])
        self.val_ds.dataset.transform = val_tf

        targets = [full_ds.targets[i] for i in self.train_ds.indices]
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = weights[targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # num_workers=0 required on Windows
        self.train_dl = DataLoader(self.train_ds, batch_size=args.batch,
                                   sampler=sampler, num_workers=0, pin_memory=True)
        self.val_dl   = DataLoader(self.val_ds, batch_size=args.batch,
                                   shuffle=False, num_workers=0, pin_memory=True)

        self.model = DeepfakeClassifier().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        self.criterion = nn.BCELoss()

        self._freeze_early_layers()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def _freeze_early_layers(self):
        for name, param in self.model.backbone.named_parameters():
            if any(f"features.{i}" in name for i in range(6)):
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,}")

    def train(self):
        best_val_acc = 0.0

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            trn_loss, trn_correct, trn_total = 0.0, 0, 0

            for x, y in tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.args.epochs} [train]"):
                x = x.to(self.device)
                y = y.float().to(self.device)

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

            val_loss, val_acc, val_auc = self._validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d} | "
                f"trn_loss={trn_loss:.4f} trn_acc={trn_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_auc={val_auc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_acc":          val_acc,
                    "val_auc":          val_auc,
                }, self.args.output)
                print(f"  ✓ Saved best model (val_acc={val_acc:.3f}, auc={val_auc:.3f})")

        print(f"\nTraining done! Best val_acc={best_val_acc:.3f}")
        print(f"Model saved to: {self.args.output}")

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []

        for x, y in self.val_dl:
            x = x.to(self.device)
            y = y.float().to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss_sum  += loss.item() * x.size(0)
            correct   += ((pred > 0.5).float() == y).sum().item()
            total     += x.size(0)
            all_probs.extend(pred.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(all_labels, all_probs))
        except:
            auc = 0.0

        return loss_sum/total, correct/total, auc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--epochs",  type=int,   default=5)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--output",  default="checkpoints/deepfake_classifier.pth")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    trainer = Trainer(args)
    trainer.train()