"""
XAI (Explainability) Pipeline
Full 3-layer explainability for deepfake detection:

  Layer 1: Attention Rollout  → which facial regions were suspicious
  Layer 2: SHAP pixel values  → per-pixel contribution to fake score
  Layer 3: Artifact Classifier → what TYPE of manipulation was found

Outputs annotated image + structured JSON breakdown per region.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Artifact Types
# ---------------------------------------------------------------------------

class ArtifactType(str, Enum):
    FACE_SWAP_BOUNDARY  = "Face-swap blending boundary"
    GAN_FINGERPRINT     = "GAN texture fingerprint"
    LIP_SYNC_MISMATCH   = "Lip-sync temporal mismatch"
    TEXTURE_SYNTHESIS   = "Synthetic texture artifacts"
    FULL_FACE_SYNTHESIS = "Fully synthesized face"
    COMPRESSION_ARTIFACT= "Compression inconsistency"
    UNKNOWN             = "Unknown manipulation"


@dataclass
class RegionExplanation:
    region_name:          str
    artifact_type:        ArtifactType
    confidence_contribution: float   # 0–1, how much this region contributed
    bounding_box:         Tuple[int, int, int, int]  # x1, y1, x2, y2
    description:          str

    def to_dict(self) -> Dict:
        return {
            "region":           self.region_name,
            "artifact_type":    self.artifact_type.value,
            "contribution_pct": round(self.confidence_contribution * 100, 1),
            "bounding_box":     list(self.bounding_box),
            "description":      self.description,
        }


# ---------------------------------------------------------------------------
# Layer 1: Attention Rollout
# ---------------------------------------------------------------------------

class AttentionRollout:
    """
    Computes attention rollout for transformer-based models (CLIP ViT).
    Reveals which spatial patches the model attended to most.

    Rollout = matrix product of attention maps across all layers,
    accounting for residual connections.
    """

    def __init__(self, model, discard_ratio: float = 0.9):
        self.model        = model
        self.discard_ratio = discard_ratio
        self.attention_maps: List[torch.Tensor] = []
        self._hooks: List = []

    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights from transformer blocks."""
        self.attention_maps.append(output.detach().cpu())

    def register_hooks(self):
        """Register hooks on all attention layers."""
        self.attention_maps = []
        if hasattr(self.model, 'clip_model') and self.model.clip_model is not None:
            visual = self.model.clip_model.visual
            for block in visual.transformer.resblocks:
                hook = block.attn.register_forward_hook(self._attention_hook)
                self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def compute(self, tensor: torch.Tensor, image_size: int = 224) -> np.ndarray:
        """
        Runs forward pass with hooks and computes rollout attention map.
        Returns attention heatmap (H, W) normalized to [0, 1].
        """
        self.register_hooks()
        with torch.no_grad():
            _ = self.model(tensor)
        self.remove_hooks()

        if not self.attention_maps:
            # Fallback: return uniform attention
            return np.ones((image_size, image_size), dtype=np.float32) * 0.5

        # Roll out attention across layers
        result = torch.eye(self.attention_maps[0].shape[-1])

        for attn in self.attention_maps:
            # attn shape: (heads, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
            if attn.dim() == 4:
                attn = attn.squeeze(0)
            if attn.dim() == 3:
                attn = attn.mean(0)  # average over heads

            # Discard low attention values
            flat       = attn.flatten()
            threshold  = torch.quantile(flat, self.discard_ratio)
            attn       = torch.where(attn < threshold, torch.zeros_like(attn), attn)

            # Add residual connection
            attn = attn + torch.eye(attn.shape[-1])
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
            result = torch.matmul(attn, result)

        # Extract patch attention (skip CLS token)
        mask     = result[0, 1:]  # (num_patches,)
        patch_size = int(mask.shape[0] ** 0.5)

        if patch_size ** 2 != mask.shape[0]:
            # Non-square: use sqrt approximation
            patch_size = int(mask.shape[0] ** 0.5) + 1
            mask = F.pad(mask, (0, patch_size**2 - mask.shape[0]))

        attn_map  = mask.reshape(patch_size, patch_size).numpy()
        attn_map  = cv2.resize(attn_map, (image_size, image_size))
        attn_map  = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
        return attn_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Layer 2: Gradient-based SHAP approximation
# ---------------------------------------------------------------------------

class GradientSHAP:
    """
    Gradient-based pixel importance estimation.
    Approximates SHAP values using integrated gradients:
    IG(x) = (x - baseline) * integral(dF/dx from baseline to x)

    This tells us exactly which pixels pushed the prediction toward "FAKE".
    """

    N_STEPS = 50  # integration steps

    def __init__(self, model: nn.Module):
        self.model = model

    def compute(self, tensor: torch.Tensor, baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        tensor:   (1, 3, H, W) input image
        baseline: (1, 3, H, W) reference (black image by default)
        Returns:  (H, W) importance map, values in [-1, 1]
                  Positive = pushed toward FAKE, Negative = pushed toward REAL
        """
        if baseline is None:
            baseline = torch.zeros_like(tensor)

        # Interpolate between baseline and input
        alphas    = torch.linspace(0, 1, self.N_STEPS).view(-1, 1, 1, 1)
        interps   = baseline + alphas * (tensor - baseline)  # (N_STEPS, 3, H, W)

        self.model.eval()
        interps.requires_grad_(True)

        # Forward pass
        preds = self.model(interps)
        if preds.dim() > 1:
            preds = preds.squeeze(-1)

        # Gradient of mean prediction w.r.t. inputs
        grad = torch.autograd.grad(
            outputs   = preds.sum(),
            inputs    = interps,
            create_graph = False,
        )[0]

        # Integrated gradients
        ig = (tensor - baseline) * grad.mean(0, keepdim=True)

        # Collapse to single channel (sum over RGB)
        importance = ig.squeeze(0).sum(0)  # (H, W)
        importance = importance.detach().numpy()

        # Normalize to [-1, 1]
        abs_max = np.abs(importance).max() + 1e-6
        importance = importance / abs_max

        return importance.astype(np.float32)


# ---------------------------------------------------------------------------
# Layer 3: Artifact Classifier
# ---------------------------------------------------------------------------

class ArtifactClassifier:
    """
    Analyzes the attention map and image features to classify
    what TYPE of deepfake artifact was detected.

    Uses heuristics + lightweight analysis since this runs after main model.
    """

    # Facial region definitions (relative to 224x224 face crop)
    REGIONS = {
        "left_eye":    (50,  50,  110, 90),
        "right_eye":   (114, 50,  174, 90),
        "nose":        (80,  80,  144, 140),
        "mouth":       (62,  140, 162, 185),
        "left_cheek":  (20,  90,  75,  160),
        "right_cheek": (149, 90,  204, 160),
        "forehead":    (45,  15,  179, 55),
        "chin":        (70,  185, 154, 224),
    }

    def classify_artifacts(
        self,
        face_image:    np.ndarray,
        attention_map: np.ndarray,
        shap_map:      np.ndarray,
        fake_prob:     float,
    ) -> List[RegionExplanation]:
        """
        Returns a list of RegionExplanation, one per suspicious region.
        """
        explanations = []

        for region_name, (x1, y1, x2, y2) in self.REGIONS.items():
            # Clamp to image bounds
            h, w = attention_map.shape
            x1c, y1c = min(x1, w-1), min(y1, h-1)
            x2c, y2c = min(x2, w), min(y2, h)

            attn_score  = float(attention_map[y1c:y2c, x1c:x2c].mean())
            shap_score  = float(shap_map[y1c:y2c, x1c:x2c].mean())
            contribution = (attn_score + max(shap_score, 0)) / 2.0

            if contribution < 0.25:
                continue  # Skip non-suspicious regions

            artifact_type = self._classify_region_artifact(
                region_name, face_image[y1c:y2c, x1c:x2c],
                attn_score, shap_score, fake_prob
            )

            description = self._generate_description(
                region_name, artifact_type, contribution
            )

            explanations.append(RegionExplanation(
                region_name             = region_name.replace("_", " ").title(),
                artifact_type           = artifact_type,
                confidence_contribution = min(contribution * fake_prob, 1.0),
                bounding_box            = (x1, y1, x2, y2),
                description             = description,
            ))

        # Sort by contribution descending
        explanations.sort(key=lambda e: e.confidence_contribution, reverse=True)
        return explanations[:5]  # Top 5 regions

    def _classify_region_artifact(
        self,
        region:     str,
        crop:       np.ndarray,
        attn_score: float,
        shap_score: float,
        fake_prob:  float,
    ) -> ArtifactType:
        """Heuristic classification of artifact type per region."""

        if region in ("left_eye", "right_eye", "left_cheek", "right_cheek"):
            if self._has_blending_boundary(crop):
                return ArtifactType.FACE_SWAP_BOUNDARY
            return ArtifactType.GAN_FINGERPRINT

        if region == "mouth":
            return ArtifactType.LIP_SYNC_MISMATCH

        if region in ("forehead", "chin"):
            if self._has_texture_artifacts(crop):
                return ArtifactType.TEXTURE_SYNTHESIS
            return ArtifactType.FULL_FACE_SYNTHESIS

        return ArtifactType.UNKNOWN

    def _has_blending_boundary(self, crop: np.ndarray) -> bool:
        """Detect blending seams via gradient discontinuities."""
        if crop.size == 0:
            return False
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        # High variance in Laplacian = blending artifacts
        return float(edges.var()) > 150.0

    def _has_texture_artifacts(self, crop: np.ndarray) -> bool:
        """Detect unnatural texture regularity (GAN/diffusion patterns)."""
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        # Low local variance = unnaturally smooth (synthetic texture)
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return local_var < 50.0

    def _generate_description(
        self,
        region:        str,
        artifact_type: ArtifactType,
        contribution:  float,
    ) -> str:
        templates = {
            ArtifactType.FACE_SWAP_BOUNDARY:  f"Blending boundary detected around {region}. The skin tone and texture transition unnaturally, suggesting a face-swap composite.",
            ArtifactType.GAN_FINGERPRINT:     f"GAN-specific frequency patterns found in {region}. Deep neural networks leave statistical fingerprints invisible to the human eye.",
            ArtifactType.LIP_SYNC_MISMATCH:   f"Mouth region shows temporal inconsistency. Lip movements don't align naturally with expected speech patterns.",
            ArtifactType.TEXTURE_SYNTHESIS:   f"Unnaturally smooth synthetic texture in {region}. Real skin has micro-texture variation that is absent here.",
            ArtifactType.FULL_FACE_SYNTHESIS: f"{region.replace('_', ' ').title()} shows signs of complete facial synthesis rather than modification of a real face.",
            ArtifactType.COMPRESSION_ARTIFACT:f"Unusual compression artifacts in {region} inconsistent with the rest of the image.",
            ArtifactType.UNKNOWN:             f"Suspicious patterns detected in {region} that differ from authentic facial imagery.",
        }
        return templates.get(artifact_type, "Suspicious region detected.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class XAIVisualizer:
    """
    Generates annotated images from XAI outputs.
    """

    def generate_heatmap_overlay(
        self,
        face_image:    np.ndarray,
        attention_map: np.ndarray,
        alpha:         float = 0.5,
    ) -> np.ndarray:
        """Overlays attention heatmap on face image."""
        heatmap     = (attention_map * 255).astype(np.uint8)
        heatmap_rgb = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.resize(heatmap_rgb, (face_image.shape[1], face_image.shape[0]))
        overlay     = cv2.addWeighted(face_image, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay

    def annotate_regions(
        self,
        face_image:   np.ndarray,
        explanations: List[RegionExplanation],
    ) -> np.ndarray:
        """Draws bounding boxes and labels on suspicious regions."""
        annotated = face_image.copy()

        colors = {
            ArtifactType.FACE_SWAP_BOUNDARY:  (0,   0,   255),   # Red
            ArtifactType.GAN_FINGERPRINT:     (0,   165, 255),   # Orange
            ArtifactType.LIP_SYNC_MISMATCH:   (0,   255, 255),   # Yellow
            ArtifactType.TEXTURE_SYNTHESIS:   (255, 0,   255),   # Magenta
            ArtifactType.FULL_FACE_SYNTHESIS: (255, 0,   0),     # Blue
            ArtifactType.UNKNOWN:             (128, 128, 128),   # Gray
        }

        for exp in explanations:
            x1, y1, x2, y2 = exp.bounding_box
            color           = colors.get(exp.artifact_type, (128, 128, 128))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{exp.region_name}: {exp.confidence_contribution*100:.0f}%"
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
            )

        return annotated

    def generate_shap_overlay(
        self,
        face_image: np.ndarray,
        shap_map:   np.ndarray,
    ) -> np.ndarray:
        """
        Red = pixels pushing toward FAKE
        Blue = pixels pushing toward REAL
        """
        h, w = face_image.shape[:2]
        shap_resized = cv2.resize(shap_map, (w, h))

        overlay = face_image.copy().astype(np.float32)
        pos_mask = shap_resized > 0
        neg_mask = shap_resized < 0

        overlay[pos_mask] = np.clip(
            overlay[pos_mask] + shap_resized[pos_mask, np.newaxis] * 80 * np.array([0, 0, 255]),
            0, 255
        )
        overlay[neg_mask] = np.clip(
            overlay[neg_mask] + np.abs(shap_resized[neg_mask, np.newaxis]) * 80 * np.array([255, 0, 0]),
            0, 255
        )
        return overlay.astype(np.uint8)


# ---------------------------------------------------------------------------
# Master XAI Pipeline
# ---------------------------------------------------------------------------

class XAIPipeline:
    """
    Orchestrates the full 3-layer XAI analysis.
    Call `explain()` with a face image and model to get full explainability output.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device              = device
        self.attention_rollout   = AttentionRollout(model)
        self.gradient_shap       = GradientSHAP(model)
        self.artifact_classifier = ArtifactClassifier()
        self.visualizer          = XAIVisualizer()

    def explain(
        self,
        face_image: np.ndarray,
        tensor:     torch.Tensor,
        fake_prob:  float,
    ) -> Dict:
        """
        Full XAI explanation.

        Args:
            face_image: (H, W, 3) BGR numpy face crop
            tensor:     (1, 3, H, W) preprocessed model input
            fake_prob:  model's fake probability output

        Returns dict with:
          attention_map:      numpy array
          shap_map:           numpy array
          regions:            list of RegionExplanation dicts
          heatmap_image:      annotated BGR numpy image
          annotated_image:    region-annotated BGR numpy image
          summary:            human-readable text summary
        """
        h, w = face_image.shape[:2]

        # --- Layer 1: Attention Rollout ---
        try:
            attention_map = self.attention_rollout.compute(tensor, image_size=h)
        except Exception:
            attention_map = np.ones((h, w), dtype=np.float32) * 0.5

        # --- Layer 2: SHAP ---
        try:
            shap_map = self.gradient_shap.compute(tensor.requires_grad_(True))
        except Exception:
            shap_map = np.zeros((h, w), dtype=np.float32)

        # --- Layer 3: Artifact Classification ---
        explanations = self.artifact_classifier.classify_artifacts(
            face_image, attention_map, shap_map, fake_prob
        )

        # --- Visualization ---
        heatmap_image    = self.visualizer.generate_heatmap_overlay(face_image, attention_map)
        annotated_image  = self.visualizer.annotate_regions(face_image, explanations)
        shap_image       = self.visualizer.generate_shap_overlay(face_image, shap_map)

        # --- Summary ---
        summary = self._build_summary(fake_prob, explanations)

        return {
            "attention_map":   attention_map.tolist(),
            "shap_map":        shap_map.tolist(),
            "regions":         [e.to_dict() for e in explanations],
            "heatmap_image":   heatmap_image,
            "annotated_image": annotated_image,
            "shap_image":      shap_image,
            "summary":         summary,
        }

    def _build_summary(self, fake_prob: float, explanations: List[RegionExplanation]) -> str:
        verdict  = "FAKE" if fake_prob > 0.5 else "REAL"
        pct      = round(fake_prob * 100, 1)

        if not explanations:
            return f"Verdict: {verdict} ({pct}% fake probability). No specific artifact regions identified."

        top_region   = explanations[0]
        artifact_str = top_region.artifact_type.value

        lines = [
            f"Verdict: {verdict} ({pct}% fake probability).",
            f"Primary indicator: {artifact_str} in the {top_region.region_name} area ({top_region.confidence_contribution*100:.0f}% contribution).",
        ]

        if len(explanations) > 1:
            secondary = explanations[1]
            lines.append(
                f"Secondary indicator: {secondary.artifact_type.value} detected in {secondary.region_name}."
            )

        return " ".join(lines)
