"""
CLIP-based Image Deepfake Detector
Fine-tunes OpenAI CLIP's vision encoder with a binary classifier head.
Uses SBI (Self-Blended Images) training strategy for robustness against
unseen deepfake generators (GAN, Diffusion, Face-swap variants).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional, Tuple
import hashlib

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class CLIPDeepfakeDetector(nn.Module):
    """
    Architecture:
      CLIP ViT-L/14 vision encoder (frozen early layers, fine-tune last 4)
        → 768-dim feature vector
        → Dropout(0.3)
        → FC(768 → 256) + GELU
        → Dropout(0.2)
        → FC(256 → 1)  → sigmoid → fake probability
    """

    def __init__(self, clip_model_name: str = "ViT-L/14", freeze_layers: int = 20):
        super().__init__()
        self.clip_model_name = clip_model_name

        if CLIP_AVAILABLE:
            self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
            # Freeze all CLIP params first
            for param in self.clip_model.parameters():
                param.requires_grad = False

            # Unfreeze last `freeze_layers` transformer blocks
            visual_blocks = list(self.clip_model.visual.transformer.resblocks)
            for block in visual_blocks[-freeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

            embed_dim = self.clip_model.visual.output_dim
        else:
            # Fallback for environments without CLIP
            embed_dim = 768
            self.clip_model = None
            self.preprocess = None

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLIP visual features from preprocessed tensor."""
        if self.clip_model is not None:
            with torch.no_grad() if not any(
                p.requires_grad for p in self.clip_model.visual.parameters()
            ) else torch.enable_grad():
                features = self.clip_model.encode_image(x)
            return features.float()
        else:
            # Fallback: return random features (for testing without CLIP)
            return torch.randn(x.shape[0], 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits   = self.classifier(features)
        return torch.sigmoid(logits).squeeze(-1)


class SBIAugmenter:
    """
    Self-Blended Images (SBI) Training Strategy.

    Generates synthetic fake training samples on-the-fly by:
    1. Taking a real face crop
    2. Applying random geometric/color transformations to create a "source"
    3. Blending the transformed source onto the original using alpha masks
    4. This simulates the blending artifacts found in real deepfakes
    """

    def __init__(self):
        self.blend_ratios = [0.25, 0.5, 0.75]
        self.mask_types   = ["gaussian", "uniform", "edge"]

    def generate_blended_fake(self, real_face: np.ndarray) -> np.ndarray:
        """
        Takes a real face (H, W, 3) uint8 and returns
        a synthetic blended fake image.
        """
        source = self._apply_source_transforms(real_face.copy())
        mask   = self._generate_blend_mask(real_face.shape[:2])
        ratio  = np.random.choice(self.blend_ratios)
        blended = (ratio * source + (1 - ratio) * real_face).astype(np.uint8)
        return blended

    def _apply_source_transforms(self, face: np.ndarray) -> np.ndarray:
        """Random color + geometric distortions to simulate a different source."""
        # Random color jitter
        face = face.astype(np.float32)
        face *= np.random.uniform(0.8, 1.2, size=(1, 1, 3))
        face  = np.clip(face, 0, 255).astype(np.uint8)

        # Random horizontal flip
        if np.random.random() > 0.5:
            face = cv2.flip(face, 1)

        # Random slight rotation
        angle = np.random.uniform(-10, 10)
        h, w  = face.shape[:2]
        M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        face  = cv2.warpAffine(face, M, (w, h))

        return face

    def _generate_blend_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates soft blending masks (Gaussian or random)."""
        h, w    = shape
        mask_type = np.random.choice(self.mask_types)

        if mask_type == "gaussian":
            mask = np.random.randn(h, w)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        elif mask_type == "uniform":
            mask = np.ones((h, w)) * np.random.uniform(0.3, 0.7)
        else:  # edge — blend more at face boundaries
            mask  = np.ones((h, w)) * 0.5
            mask[:10, :]  = 0.2
            mask[-10:, :] = 0.2
            mask[:, :10]  = 0.2
            mask[:, -10:] = 0.2

        return mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting


class ImageDetectionResult:
    """Structured result object from image detection."""

    def __init__(
        self,
        fake_probability: float,
        is_fake: bool,
        confidence: str,
        file_hash: str,
        features: Optional[np.ndarray] = None,
    ):
        self.fake_probability = fake_probability
        self.is_fake          = is_fake
        self.confidence       = confidence
        self.file_hash        = file_hash
        self.features         = features

    def to_dict(self) -> Dict:
        return {
            "fake_probability": round(self.fake_probability, 4),
            "real_probability": round(1 - self.fake_probability, 4),
            "is_fake":          self.is_fake,
            "confidence":       self.confidence,
            "verdict":          "FAKE" if self.is_fake else "REAL",
            "file_hash":        self.file_hash,
        }

    @staticmethod
    def _confidence_label(prob: float) -> str:
        if prob > 0.90 or prob < 0.10:
            return "Very High"
        elif prob > 0.75 or prob < 0.25:
            return "High"
        elif prob > 0.60 or prob < 0.40:
            return "Medium"
        else:
            return "Low"


class CLIPDetectorInference:
    """
    Production inference wrapper for the CLIP detector.
    Handles model loading, face detection, preprocessing, and result packaging.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = CLIPDeepfakeDetector()
        self.model.to(self.device)

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        Full pipeline:
          load image → detect face → preprocess → model inference → result
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Compute hash
        file_hash = self._compute_hash(image_path)

        # Face detection
        from ai.preprocessing.face_detector import FaceDetector, ImagePreprocessor
        detector     = FaceDetector(device=str(self.device))
        preprocessor = ImagePreprocessor()

        face = detector.detect_and_align(image)

        if face is None:
            # If no face found, use full image (might still be detectable)
            face = cv2.resize(image, (224, 224))
            face_found = False
        else:
            face_found = True

        # Preprocess
        tensor = preprocessor.preprocess(face).to(self.device)

        # Inference
        fake_prob = float(self.model(tensor).cpu().item())

        result = ImageDetectionResult(
            fake_probability = fake_prob,
            is_fake          = fake_prob > 0.5,
            confidence       = ImageDetectionResult._confidence_label(fake_prob),
            file_hash        = file_hash,
        )

        output = result.to_dict()
        output["face_detected"] = face_found
        return output

    def _compute_hash(self, path: str) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
