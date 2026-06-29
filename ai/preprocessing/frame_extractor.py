"""
Video Frame Extractor
Adaptively samples frames from videos based on FPS and duration,
then tracks faces across frames using a simplified tracking approach.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import hashlib


class FrameExtractor:
    """
    Extracts frames from video adaptively:
    - Short videos  (<30s)  → every 5th frame
    - Medium videos (30–120s) → every 10th frame
    - Long videos   (>120s)  → every 20th frame
    Max 120 frames per video to keep inference fast.
    """

    MAX_FRAMES = 120
    SAMPLE_RULES = [
        (30,  5),
        (120, 10),
        (float("inf"), 20),
    ]

    def __init__(self, max_frames: int = MAX_FRAMES):
        self.max_frames = max_frames

    def extract(self, video_path: str) -> Dict:
        """
        Returns a dict with:
          frames      : List[np.ndarray] - BGR frames
          fps         : float
          total_frames: int
          duration    : float (seconds)
          sample_step : int
          frame_indices: List[int]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps

        sample_step = self._get_sample_step(duration)

        frames        = []
        frame_indices = []
        idx           = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_step == 0:
                frames.append(frame)
                frame_indices.append(idx)
            idx += 1

        cap.release()

        return {
            "frames":       frames,
            "fps":          fps,
            "total_frames": total_frames,
            "duration":     duration,
            "sample_step":  sample_step,
            "frame_indices": frame_indices,
        }

    def _get_sample_step(self, duration: float) -> int:
        for threshold, step in self.SAMPLE_RULES:
            if duration <= threshold:
                return step
        return self.SAMPLE_RULES[-1][1]

    @staticmethod
    def compute_video_hash(video_path: str) -> str:
        """SHA-256 hash of video file for blockchain storage."""
        sha256 = hashlib.sha256()
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class TemporalConsistencyAnalyzer:
    """
    Analyzes optical flow between consecutive frames to detect
    unnatural temporal inconsistencies — a key deepfake signal.
    Real videos have smooth motion; deepfakes often have jitter.
    """

    def __init__(self):
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    def compute_flow_scores(self, frames: List[np.ndarray]) -> Dict:
        """
        Returns:
          mean_magnitude   : average optical flow magnitude across all frame pairs
          std_magnitude    : std — high std = inconsistent motion (deepfake signal)
          anomaly_frames   : frame indices where flow jumps are suspicious
          consistency_score: 0–1 where 1 = perfectly consistent (likely real)
        """
        if len(frames) < 2:
            return {"consistency_score": 0.5, "anomaly_frames": []}

        magnitudes = []
        anomaly_frames = []

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i, frame in enumerate(frames[1:], 1):
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, **self.flow_params
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            magnitudes.append(mean_mag)
            prev_gray = curr_gray

        if not magnitudes:
            return {"consistency_score": 0.5, "anomaly_frames": []}

        mean_mag = np.mean(magnitudes)
        std_mag  = np.std(magnitudes)

        # Flag frames where flow deviates > 2 std from mean
        threshold = mean_mag + 2 * std_mag
        for i, m in enumerate(magnitudes):
            if m > threshold:
                anomaly_frames.append(i + 1)

        # Consistency score: lower std relative to mean = more consistent
        cv_coef = std_mag / (mean_mag + 1e-6)  # coefficient of variation
        consistency_score = float(np.clip(1.0 - cv_coef / 2.0, 0.0, 1.0))

        return {
            "mean_magnitude":    float(mean_mag),
            "std_magnitude":     float(std_mag),
            "anomaly_frames":    anomaly_frames,
            "consistency_score": consistency_score,
            "magnitudes":        [float(m) for m in magnitudes],
        }
