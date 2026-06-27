"""
AltFreezing Video Deepfake Detector
Implements the AltFreezing training strategy that alternates between:
  - Spatial stream:   detects per-frame artifacts (blending, texture)
  - Temporal stream:  detects cross-frame inconsistencies (motion, identity)

This alternating freeze strategy forces the model to learn BOTH types
of deepfake artifacts independently, then fuses them at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Spatial Stream — Frame-level artifact detection
# ---------------------------------------------------------------------------

class SpatialStream(nn.Module):
    """
    Processes individual frames to find spatial artifacts:
    blending boundaries, GAN fingerprints, texture inconsistencies.
    Uses a ResNet-like backbone with attention.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
        super().__init__()

        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # Channel attention (SE block)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 512),
            nn.Sigmoid(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, feature_dim)"""
        feat     = self.backbone(x)                      # (B, 512, 7, 7)
        attn     = self.channel_attn(feat).unsqueeze(-1).unsqueeze(-1)
        feat     = feat * attn                           # channel-wise reweight
        pooled   = self.pool(feat).flatten(1)            # (B, 512)
        return self.proj(pooled)                         # (B, feature_dim)


# ---------------------------------------------------------------------------
# Temporal Stream — Cross-frame inconsistency detection
# ---------------------------------------------------------------------------

class TemporalStream(nn.Module):
    """
    Processes sequences of frames to detect temporal inconsistencies.
    Uses a GRU over per-frame spatial features to model dynamics.
    Deepfakes often have flickering, identity drift, or unnatural motion.
    """

    def __init__(self, spatial_feature_dim: int = 512, hidden_dim: int = 256, feature_dim: int = 512):
        super().__init__()

        # Per-frame spatial encoder (lightweight)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.gru = nn.GRU(
            input_size  = 256,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.2,
            bidirectional = True,
        )

        self.proj = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, H, W)  — T frames per video clip
        returns: (B, feature_dim)
        """
        B, T, C, H, W = frames.shape
        frames_flat    = frames.view(B * T, C, H, W)
        frame_feats    = self.frame_encoder(frames_flat)   # (B*T, 256)
        frame_feats    = frame_feats.view(B, T, -1)        # (B, T, 256)

        gru_out, _     = self.gru(frame_feats)             # (B, T, 512)
        # Use last hidden state from both directions
        temporal_feat  = gru_out[:, -1, :]                 # (B, 512)
        return self.proj(temporal_feat)                    # (B, feature_dim)


# ---------------------------------------------------------------------------
# AltFreezing Fusion Model
# ---------------------------------------------------------------------------

class AltFreezingDetector(nn.Module):
    """
    Fuses spatial + temporal streams.

    During training:
      - Even epochs → freeze temporal, train spatial
      - Odd epochs  → freeze spatial, train temporal
    This alternating strategy prevents one stream from dominating.

    At inference: both streams run and features are concatenated.
    """

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.spatial_stream  = SpatialStream(feature_dim=feature_dim)
        self.temporal_stream = TemporalStream(feature_dim=feature_dim)

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def set_training_mode(self, epoch: int):
        """
        Alternating freeze: call at the start of each training epoch.
        """
        if epoch % 2 == 0:
            # Train spatial only
            self._set_stream_grad(self.spatial_stream, True)
            self._set_stream_grad(self.temporal_stream, False)
        else:
            # Train temporal only
            self._set_stream_grad(self.spatial_stream, False)
            self._set_stream_grad(self.temporal_stream, True)

        # Fusion head always trains
        for p in self.fusion.parameters():
            p.requires_grad = True

    def _set_stream_grad(self, stream: nn.Module, requires_grad: bool):
        for p in stream.parameters():
            p.requires_grad = requires_grad

    def forward(
        self,
        frame: torch.Tensor,              # single representative frame (B, 3, H, W)
        frames_seq: torch.Tensor,         # sequence (B, T, 3, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          fake_prob     : (B,) — final fused prediction
          spatial_feat  : (B, feature_dim) — for XAI
          temporal_feat : (B, feature_dim) — for XAI
        """
        spatial_feat  = self.spatial_stream(frame)
        temporal_feat = self.temporal_stream(frames_seq)
        fused         = torch.cat([spatial_feat, temporal_feat], dim=1)
        logit         = self.fusion(fused)
        fake_prob     = torch.sigmoid(logit).squeeze(-1)
        return fake_prob, spatial_feat, temporal_feat


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

@dataclass
class VideoFrameResult:
    frame_index:    int
    fake_prob:      float
    is_suspicious:  bool


class AltFreezingInference:
    """
    Runs AltFreezing inference on extracted video frames.
    Returns per-clip and aggregated results.
    """

    CLIP_LENGTH = 8   # frames per clip fed to temporal stream

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = AltFreezingDetector()
        self.model.to(self.device)

        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

        self.model.eval()

    @torch.no_grad()
    def predict_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        frames: list of BGR face crops (224x224 each)
        Returns aggregated detection result with per-clip breakdown.
        """
        if len(frames) < 2:
            return {"error": "Need at least 2 frames for video analysis"}

        preprocessed = self._preprocess_frames(frames)
        clip_results: List[VideoFrameResult] = []

        # Slide a window of CLIP_LENGTH over the frames
        for start in range(0, len(preprocessed) - self.CLIP_LENGTH + 1, self.CLIP_LENGTH // 2):
            clip     = preprocessed[start: start + self.CLIP_LENGTH]
            if len(clip) < self.CLIP_LENGTH:
                break

            # Representative frame = middle of clip
            rep_frame   = clip[len(clip) // 2].unsqueeze(0).to(self.device)
            frames_seq  = torch.stack(clip).unsqueeze(0).to(self.device)   # (1, T, 3, H, W)

            fake_prob, _, _ = self.model(rep_frame, frames_seq)
            prob = float(fake_prob.cpu().item())

            clip_results.append(VideoFrameResult(
                frame_index   = start,
                fake_prob     = prob,
                is_suspicious = prob > 0.5,
            ))

        if not clip_results:
            return {"error": "Not enough frames for analysis"}

        probs          = [r.fake_prob for r in clip_results]
        final_prob     = float(np.mean(probs))
        suspicious_clips = [r for r in clip_results if r.is_suspicious]

        return {
            "fake_probability":   round(final_prob, 4),
            "real_probability":   round(1 - final_prob, 4),
            "is_fake":            final_prob > 0.5,
            "verdict":            "FAKE" if final_prob > 0.5 else "REAL",
            "clips_analyzed":     len(clip_results),
            "suspicious_clips":   len(suspicious_clips),
            "clip_breakdown": [
                {
                    "clip_start_frame": r.frame_index,
                    "fake_prob":        round(r.fake_prob, 4),
                    "suspicious":       r.is_suspicious,
                }
                for r in clip_results
            ],
        }

    def _preprocess_frames(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """Convert BGR numpy frames to normalized tensors."""
        import cv2
        MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tensors = []
        for frame in frames:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224))
            t     = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            t     = (t - MEAN) / STD
            tensors.append(t)
        return tensors
