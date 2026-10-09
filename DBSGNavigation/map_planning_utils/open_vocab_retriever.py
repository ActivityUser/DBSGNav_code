"""
OpenVocabRetriever: CLIP-based instance retrieval for cross-modal instance navigation.
Returns Top-K candidates (instance_info, position, cosine_sim) for geometric verification.
"""

import json
import os
from typing import List, Tuple, Optional, Any

import numpy as np
import torch


def _try_parse_nav_xz(first_nav: Any) -> Optional[Tuple[float, float]]:
    """
    Parse navigable viewpoint to (x, z) in world coordinates.
    Handles nested [[x, z]], numpy arrays, and avoids crashes on scalars / one-element lists.
    """
    if first_nav is None:
        return None
    if isinstance(first_nav, np.ndarray):
        flat = np.asarray(first_nav, dtype=np.float64).reshape(-1)
        if flat.size >= 2:
            return float(flat[0]), float(flat[1])
        return None
    if isinstance(first_nav, (list, tuple)):
        if len(first_nav) == 0:
            return None
        if len(first_nav) >= 2:
            a, b = first_nav[0], first_nav[1]
            if isinstance(a, (list, tuple, np.ndarray)) and not isinstance(
                a, (str, bytes)
            ):
                inner = _try_parse_nav_xz(a)
                if inner is not None:
                    return inner
            try:
                return float(a), float(b)
            except (TypeError, ValueError):
                return None
        return _try_parse_nav_xz(first_nav[0])
    return None


class OpenVocabRetriever:
    """Retrieve instance candidates by CLIP semantic similarity (cosine)."""

    COCO_IDS = [56, 57, 58, 59, 61, 62]  # chair, sofa, plant, bed, toilet, tv_monitor

    def __init__(
        self,
        device: torch.device,
        clip_model_name: str = "ViT-B-32",
        clip_dim: int = 512,
        clip_pretrained_path: Optional[str] = None,
        clip_vectors_filename: str = "instance_clip_vectors.json",
    ):
        self.device = device
        self.clip_model_name = clip_model_name
        self.clip_dim = clip_dim
        self.clip_pretrained_path = (clip_pretrained_path or "").strip() or None
        # Per-scene filename used to look up the pre-computed CLIP vectors. Override
        # via config to switch between v1 (frame-level) and v2 (per-instance crop)
        # without renaming files in the dataset.
        self.clip_vectors_filename = (clip_vectors_filename or "instance_clip_vectors.json").strip() \
            or "instance_clip_vectors.json"
        self._clip_model = None
        self._clip_preprocess = None
        # Flat list of (instance_info, position, vector); position = [x, y, z] in env coords
        self._instance_infos: List[Tuple[Any, List[float], np.ndarray]] = []
        self._scene_path: Optional[str] = None

    def _load_clip(self):
        if self._clip_model is not None:
            return
        try:
            import open_clip
            pretrained = self.clip_pretrained_path if self.clip_pretrained_path and os.path.isfile(self.clip_pretrained_path) else "openai"
            _r = open_clip.create_model_and_transforms(self.clip_model_name, pretrained=pretrained)
            self._clip_model = _r[0].to(self.device).eval()
            self._clip_preprocess = _r[1]
        except Exception as e:
            raise RuntimeError(f"OpenVocabRetriever: failed to load CLIP ({e}). Install: pip install open-clip-torch. 若网络不可达可设置 clip_pretrained_path 为本地 .pt/.bin 路径或 HF_ENDPOINT=https://hf-mirror.com") from e

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode RGB image (H,W,3) uint8 or float [0,1] to L2-normalized 512-d vector."""
        self._load_clip()
        from PIL import Image
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
        x = self._clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._clip_model.encode_image(x)
            feat = feat.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        return (feat / norm).squeeze(0)

    def load_scene(
        self,
        scene_path: str,
        instance_retrieval_path: Optional[str] = None,
        instance_clip_vectors_path: Optional[str] = None,
    ) -> bool:
        """
        Load instance_retrieval.json and instance_clip_vectors.json for the scene.
        If instance_clip_vectors is missing, clears candidates and returns False.
        """
        ir_path = instance_retrieval_path or os.path.join(scene_path, "instance_retrieval.json")
        cv_path = instance_clip_vectors_path or os.path.join(scene_path, self.clip_vectors_filename)

        if not os.path.isfile(ir_path):
            self._instance_infos = []
            self._scene_path = None
            return False
        try:
            with open(ir_path, "r") as f:
                instance_retrieval = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self._instance_infos = []
            self._scene_path = scene_path
            print(f"[open_vocab] skip malformed instance_retrieval.json: {ir_path} ({e})")
            return False

        if not os.path.isfile(cv_path):
            self._instance_infos = []
            self._scene_path = scene_path
            return False

        try:
            with open(cv_path, "r") as f:
                clip_vectors_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self._instance_infos = []
            self._scene_path = scene_path
            print(f"[open_vocab] skip malformed instance_clip_vectors.json: {cv_path} ({e})")
            return False

        self._instance_infos = []
        self._scene_path = scene_path
        for coco_id in self.COCO_IDS:
            key = str(coco_id)
            if key not in instance_retrieval or key not in clip_vectors_data:
                continue
            entries = instance_retrieval[key]
            vectors = clip_vectors_data[key]
            if len(vectors) != len(entries):
                continue
            for idx, (entry, vec) in enumerate(zip(entries, vectors)):
                # entry = [bbox_list, frame_id, navigable_viewpoint]
                bbox_list = entry[0]
                if not bbox_list or not bbox_list[0]:
                    continue
                first_bbox = bbox_list[0]
                # Use bbox center in 3D instead of min corner.
                # Old behavior used [min_x, min_y, min_z], which introduces a systematic offset.
                center_x = float((first_bbox[0] + first_bbox[1]) * 0.5)
                center_y = float((first_bbox[2] + first_bbox[3]) * 0.5)
                center_z = float((first_bbox[4] + first_bbox[5]) * 0.5)
                position = [center_x, center_y, center_z]
                # Prefer precomputed navigable viewpoint (x,z) if available.
                # Keep y from bbox center since planner mainly relies on x/z projection.
                if len(entry) > 2 and entry[2]:
                    try:
                        raw_nav = entry[2]
                        if isinstance(raw_nav, (list, tuple)):
                            nav_points = [p for p in raw_nav if p is not None]
                        else:
                            nav_points = [raw_nav] if raw_nav is not None else []
                    except (TypeError, ValueError):
                        nav_points = []
                    if len(nav_points) > 0:
                        parsed = _try_parse_nav_xz(nav_points[0])
                        if parsed is not None:
                            nav_x, nav_z = parsed
                            position = [nav_x, center_y, nav_z]
                vec = np.array(vec, dtype=np.float32)
                if vec.size != self.clip_dim:
                    continue
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec = vec / norm
                self._instance_infos.append((entry, position, vec))
        return len(self._instance_infos) > 0

    def retrieve(
        self,
        target_image: np.ndarray,
        top_k: int,
    ) -> List[Tuple[Any, List[float], float]]:
        """
        Encode target image with CLIP, compute cosine similarity with all instance vectors,
        return top_k (instance_info, position, cosine_sim).
        instance_info is the raw entry [bbox_list, frame_id, navigable_viewpoint] for loading rgb.
        """
        if not self._instance_infos:
            return []
        v_g = self.encode_image(target_image)
        v_g = np.asarray(v_g, dtype=np.float32).reshape(1, -1)
        sims = []
        for _, _, vec in self._instance_infos:
            s = float(np.dot(v_g, vec.reshape(-1, 1)).squeeze())
            sims.append(s)
        sims = np.array(sims)
        order = np.argsort(-sims)[:top_k]
        return [
            (self._instance_infos[i][0], self._instance_infos[i][1], float(sims[i]))
            for i in order
        ]

    def get_scene_path(self) -> Optional[str]:
        return self._scene_path
