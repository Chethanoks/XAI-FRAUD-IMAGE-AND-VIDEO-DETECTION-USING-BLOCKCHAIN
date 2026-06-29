cle"""
Face Detection & Alignment
Uses MTCNN for face detection and alignment before passing to deepfake models.
"""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from typing import Optional, Tuple, List


class FaceDetector:
    def __init__(self, device: str = "cpu", image_size: int = 224):
        self.device = device
        self.image_size = image_size
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=20,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False,  # Only largest/most confident face
        )

    def detect_and_align(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects the primary face in an image and returns it aligned & cropped.
        Returns None if no face is found.
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        face_tensor = self.mtcnn(pil_image)

        if face_tensor is None:
            return None

        # Convert tensor back to numpy uint8 for downstream use
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_np = ((face_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        return face_np

    def detect_all_faces(self, image: np.ndarray) -> Tuple[Optional[List[np.ndarray]], Optional[List]]:
        """
        Returns all detected faces and their bounding boxes.
        Useful for multi-face images.
        """
        mtcnn_all = MTCNN(
            image_size=self.image_size,
            margin=20,
            min_face_size=60,
            device=self.device,
            keep_all=True,
        )
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes, probs = mtcnn_all.detect(pil_image)

        if boxes is None:
            return None, None

        faces = []
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (self.image_size, self.image_size))
                faces.append(face_resized)

        return faces, boxes.tolist()

    def get_face_landmarks(self, image: np.ndarray):
        """
        Returns facial landmarks (eyes, nose, mouth corners) for alignment.
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        _, _, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        return landmarks


class ImagePreprocessor:
    """Preprocesses images for CLIP-based detection model."""

    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD  = [0.26862954, 0.26130258, 0.27577711]

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def preprocess(self, face_np: np.ndarray) -> torch.Tensor:
        """
        Takes a numpy face crop (H, W, 3) uint8 and returns
        a normalized tensor (1, 3, H, W) ready for CLIP.
        """
        image = cv2.resize(face_np, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0

        mean = np.array(self.MEAN)
        std  = np.array(self.STD)
        image = (image - mean) / std

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    def preprocess_batch(self, faces: List[np.ndarray]) -> torch.Tensor:
        """Batch preprocess multiple face crops."""
        tensors = [self.preprocess(f).squeeze(0) for f in faces]
        return torch.stack(tensors)
