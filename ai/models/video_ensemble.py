"""
Video Detection Ensemble
Combines AltFreezing + LipForensics + Temporal Consistency Analysis
with weighted voting for final verdict.

Weights (tunable):
  AltFreezing          : 0.45  (best for general spatial+temporal fakes)
  LipForensics         : 0.35  (best for lip-sync fakes)
  TemporalConsistency  : 0.20  (catches motion jitter artifacts)
"""

import numpy as np
from typing import List, Dict, Optional
import hashlib

from ai.models.altfreezing_video import AltFreezingInference
from ai.models.lipforensics import LipForensicsInference
from ai.preprocessing.frame_extractor import FrameExtractor, TemporalConsistencyAnalyzer
from ai.preprocessing.face_detector import FaceDetector


class VideoEnsemble:
    """
    Orchestrates full video deepfake detection pipeline:
    1. Extract frames adaptively
    2. Detect & align faces per frame
    3. Run AltFreezing (spatial + temporal)
    4. Run LipForensics (lip-sync)
    5. Run Temporal Consistency Analysis (optical flow)
    6. Ensemble weighted vote
    7. Return structured result
    """

    WEIGHTS = {
        "altfreezing":   0.45,
        "lipforensics":  0.35,
        "temporal":      0.20,
    }

    def __init__(
        self,
        altfreezing_path:  Optional[str] = None,
        lipforensics_path: Optional[str] = None,
        device:            str = "cpu",
    ):
        self.device       = device
        self.extractor    = FrameExtractor()
        self.face_detector = FaceDetector(device=device)
        self.temporal_analyzer = TemporalConsistencyAnalyzer()
        self.altfreezing  = AltFreezingInference(altfreezing_path, device)
        self.lipforensics = LipForensicsInference(lipforensics_path, device)

    def predict(self, video_path: str) -> Dict:
        """
        Full video analysis pipeline.
        Returns comprehensive result dict.
        """
        # Step 1: Extract frames
        extraction = self.extractor.extract(video_path)
        frames     = extraction["frames"]

        if not frames:
            return {"error": "No frames could be extracted from video"}

        # Step 2: Compute video hash
        video_hash = FrameExtractor.compute_video_hash(video_path)

        # Step 3: Detect faces in frames
        face_crops = self._extract_faces(frames)

        if not face_crops:
            return {
                "error":      "No faces detected in video",
                "file_hash":  video_hash,
                "duration":   extraction["duration"],
            }

        # Step 4: Run all models
        altfreezing_result  = self._run_altfreezing(face_crops)
        lipforensics_result = self._run_lipforensics(face_crops)
        temporal_result     = self._run_temporal(frames)

        # Step 5: Weighted ensemble
        ensemble = self._ensemble_vote(
            altfreezing_result,
            lipforensics_result,
            temporal_result,
        )

        # Step 6: Package result
        return {
            "file_hash":         video_hash,
            "duration_seconds":  round(extraction["duration"], 2),
            "frames_analyzed":   len(face_crops),
            "fps":               extraction["fps"],

            # Final verdict
            "fake_probability":  ensemble["fake_probability"],
            "real_probability":  ensemble["real_probability"],
            "is_fake":           ensemble["is_fake"],
            "verdict":           ensemble["verdict"],
            "confidence":        ensemble["confidence"],

            # Per-model breakdown
            "model_breakdown": {
                "altfreezing": {
                    "fake_probability": altfreezing_result.get("fake_probability", 0.5),
                    "weight":           self.WEIGHTS["altfreezing"],
                    "focus":            "Spatial + temporal artifacts",
                },
                "lipforensics": {
                    "fake_probability": lipforensics_result.get("fake_probability", 0.5),
                    "weight":           self.WEIGHTS["lipforensics"],
                    "focus":            "Lip-sync consistency",
                },
                "temporal": {
                    "fake_probability": temporal_result["fake_probability"],
                    "weight":           self.WEIGHTS["temporal"],
                    "focus":            "Optical flow consistency",
                    "consistency_score": temporal_result.get("consistency_score", 0.5),
                    "anomaly_frames":    temporal_result.get("anomaly_frames", []),
                },
            },

            # Clip-level timeline (from AltFreezing)
            "timeline": altfreezing_result.get("clip_breakdown", []),
        }

    def _extract_faces(self, frames: List) -> List:
        """Extract aligned face crops from frames. Skip frames without faces."""
        face_crops = []
        for frame in frames:
            face = self.face_detector.detect_and_align(frame)
            if face is not None:
                face_crops.append(face)
        return face_crops

    def _run_altfreezing(self, face_crops: List) -> Dict:
        try:
            return self.altfreezing.predict_frames(face_crops)
        except Exception as e:
            return {"fake_probability": 0.5, "error": str(e)}

    def _run_lipforensics(self, face_crops: List) -> Dict:
        try:
            return self.lipforensics.predict_frames(face_crops)
        except Exception as e:
            return {"fake_probability": 0.5, "error": str(e)}

    def _run_temporal(self, frames: List) -> Dict:
        try:
            result = self.temporal_analyzer.compute_flow_scores(frames)
            # Convert consistency score to fake probability
            # Low consistency (lots of motion jitter) → higher fake probability
            consistency  = result.get("consistency_score", 0.5)
            fake_prob    = 1.0 - consistency
            return {
                "fake_probability":  round(fake_prob, 4),
                "consistency_score": round(consistency, 4),
                "anomaly_frames":    result.get("anomaly_frames", []),
            }
        except Exception as e:
            return {"fake_probability": 0.5, "error": str(e)}

    def _ensemble_vote(
        self,
        altfreezing:  Dict,
        lipforensics: Dict,
        temporal:     Dict,
    ) -> Dict:
        """Weighted average ensemble."""
        p_alt  = altfreezing.get("fake_probability",  0.5)
        p_lip  = lipforensics.get("fake_probability", 0.5)
        p_temp = temporal.get("fake_probability",     0.5)

        w_alt  = self.WEIGHTS["altfreezing"]
        w_lip  = self.WEIGHTS["lipforensics"]
        w_temp = self.WEIGHTS["temporal"]

        final_prob = (
            w_alt  * p_alt  +
            w_lip  * p_lip  +
            w_temp * p_temp
        )

        confidence = self._confidence_label(final_prob)

        return {
            "fake_probability": round(float(final_prob), 4),
            "real_probability": round(float(1 - final_prob), 4),
            "is_fake":          final_prob > 0.5,
            "verdict":          "FAKE" if final_prob > 0.5 else "REAL",
            "confidence":       confidence,
        }

    @staticmethod
    def _confidence_label(prob: float) -> str:
        margin = abs(prob - 0.5) * 2  # 0 = no confidence, 1 = full
        if margin > 0.8:
            return "Very High"
        elif margin > 0.5:
            return "High"
        elif margin > 0.25:
            return "Medium"
        else:
            return "Low"
