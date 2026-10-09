"""
语义高斯目标（创新点 1+2）：从 GaussianConstruction 产出的带 CLIP 的高斯地图中，
用目标图 CLIP 编码与各高斯 clip_vectors 相似度匹配，得到目标 3D 位置。
"""

import os
from glob import glob
from typing import Optional, Callable, Tuple, List, Dict, Any

import numpy as np


class SemanticGaussianGoal:
    """
    加载场景的 params.npz（means3D + clip_vectors），用目标图 CLIP 向量与高斯 CLIP 匹配，
    返回相似度最高的高斯（或 top-k 中心）的 3D 位置，作为导航目标。
    """

    def __init__(
        self,
        device,
        gaussian_map_root: str,
        clip_encode_image: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        clip_model_name: str = "ViT-B-32",
        clip_dim: int = 512,
        clip_pretrained_path: Optional[str] = None,
    ):
        self.device = device
        self.gaussian_map_root = (gaussian_map_root or "").strip()
        self.clip_encode_image = clip_encode_image
        self.clip_model_name = clip_model_name
        self.clip_dim = clip_dim
        self.clip_pretrained_path = (clip_pretrained_path or "").strip() or None
        self._means3D: Optional[np.ndarray] = None  # (N, 3)
        self._clip_vectors: Optional[np.ndarray] = None  # (N, clip_dim) L2-normalized
        self._rgb_colors: Optional[np.ndarray] = None  # (N, 3) float in [0,1] when available
        self._scene_name: Optional[str] = None
        self._clip_retriever = None  # 仅当未传入 clip_encode_image 时惰性创建
        self._last_error: Optional[str] = None  # 供调试：上次 load_scene 失败原因
        # 保护开关：若场景内 clip_vectors 近乎常量，直接判定为不可用（避免错误高相似度误导导航）。
        self._min_diversity_ratio = 0.0
        self._diversity_check_sample = 4000
        # 全局 floor_y 估计（场景级 1% 分位数，y-up），supplements agent_pos.y when given.
        self._floor_y_global: Optional[float] = None

    def _get_clip_encoder(self):
        if self.clip_encode_image is not None:
            return self.clip_encode_image
        if self._clip_retriever is None:
            from map_planning_utils.open_vocab_retriever import OpenVocabRetriever
            self._clip_retriever = OpenVocabRetriever(
                self.device,
                clip_model_name=self.clip_model_name,
                clip_dim=self.clip_dim,
                clip_pretrained_path=self.clip_pretrained_path,
            )
        return self._clip_retriever.encode_image

    def _resolve_params_path(self, scene_name: str) -> Optional[str]:
        """根据 scene_name 解析该场景的 params.npz 路径；优先含 _clip 的目录（带 CLIP 的建图）."""
        if not self.gaussian_map_root or not os.path.isdir(self.gaussian_map_root):
            return None
        candidates = [
            d for d in glob(os.path.join(self.gaussian_map_root, f"{scene_name}*"))
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "params.npz"))
        ]
        if not candidates:
            return None
        # 优先选名称含 _clip 的目录（use_clip 建图）
        candidates.sort(key=lambda d: ("_clip" not in os.path.basename(d), d))
        return os.path.join(candidates[0], "params.npz")

    def load_scene(self, scene_name: str) -> bool:
        """
        加载该场景的 params.npz，读取 means3D 与 clip_vectors。
        若缺少 clip_vectors 或全零则返回 False；失败原因见 get_last_error()。
        """
        self._last_error = None
        path = self._resolve_params_path(scene_name)
        if path is None:
            self._last_error = "no_params_npz (gaussian_map_root下无匹配目录或无params.npz)"
            self._means3D = None
            self._clip_vectors = None
            self._scene_name = None
            return False
        try:
            data = dict(np.load(path, allow_pickle=True))
        except Exception as e:
            self._last_error = f"params.npz load error: {e}"
            self._means3D = None
            self._clip_vectors = None
            self._scene_name = None
            return False
        if "means3D" not in data:
            self._last_error = "params.npz 中无 means3D"
            return False
        means3D = np.array(data["means3D"], dtype=np.float64)
        if means3D.ndim == 3:
            means3D = means3D.reshape(-1, 3)
        self._means3D = means3D
        try:
            self._floor_y_global = float(np.percentile(means3D[:, 1], 1.0))
        except Exception:
            self._floor_y_global = None

        if "clip_vectors" not in data:
            self._last_error = "params.npz 中无 clip_vectors (需 use_clip=True 建图)"
            self._clip_vectors = None
            self._scene_name = scene_name
            return False
        cv = np.array(data["clip_vectors"], dtype=np.float32)
        if cv.ndim == 1:
            cv = cv.reshape(1, -1)
        if cv.shape[0] != means3D.shape[0] or cv.shape[1] != self.clip_dim:
            self._last_error = f"clip_vectors shape 不符 (需 N×{self.clip_dim})"
            self._clip_vectors = None
            return False
        norm = np.linalg.norm(cv, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        self._clip_vectors = (cv / norm).astype(np.float32)
        self._scene_name = scene_name

        # 路径 1a：颜色直方图所需 — 读 rgb_colors（可选；若缺失记 None，后续 color prior 自动 skip）。
        self._rgb_colors = None
        try:
            if "rgb_colors" in data:
                rgb = np.array(data["rgb_colors"], dtype=np.float32)
                if rgb.ndim == 2 and rgb.shape[0] == means3D.shape[0] and rgb.shape[1] == 3:
                    if float(rgb.max()) > 1.5:
                        rgb = rgb / 255.0
                    self._rgb_colors = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        except Exception:
            self._rgb_colors = None
        if np.abs(self._clip_vectors).max() < 1e-6:
            self._last_error = "clip_vectors 全零 (需 use_clip=True 建图)"
            return False
        # Quick diversity check. Some maps may contain near-constant CLIP vectors due to
        # construction/export issues; such vectors can produce high but meaningless similarity.
        n = min(self._clip_vectors.shape[0], self._diversity_check_sample)
        if n > 16:
            probe = np.round(self._clip_vectors[:n, :16], 4)
            uniq = np.unique(probe, axis=0).shape[0]
            diversity_ratio = float(uniq) / float(n)
            if diversity_ratio < self._min_diversity_ratio:
                self._last_error = (
                    "clip_vectors 多样性过低 (ratio=%.4f < %.4f)"
                    % (diversity_ratio, self._min_diversity_ratio)
                )
                self._clip_vectors = None
                return False
        return True

    def get_last_error(self) -> Optional[str]:
        """返回上次 load_scene 失败原因，供调试."""
        return self._last_error

    def get_goal_position(
        self,
        target_image: np.ndarray,
        top_k: int = 1,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        用目标图 CLIP 与高斯 clip_vectors 做余弦相似度，取 top_k 高斯的中心作为目标位置。
        target_image: (H,W,3) uint8 或 [0,1] float。
        Returns:
            (position, max_sim) 或 None。position 为 (3,) 与 Habitat 世界坐标一致。
        """
        if self._means3D is None or self._clip_vectors is None:
            return None
        encode = self._get_clip_encoder()
        v_g = encode(target_image)
        v_g = np.asarray(v_g, dtype=np.float32).reshape(1, -1)
        if v_g.shape[1] != self._clip_vectors.shape[1]:
            return None
        sims = np.dot(self._clip_vectors, v_g.T).squeeze(-1)
        order = np.argsort(-sims)[:top_k]
        positions = self._means3D[order]
        position = np.array(
            [positions[:, 0].mean(), positions[:, 1].mean(), positions[:, 2].mean()],
            dtype=np.float64,
        )
        max_sim = float(sims[order[0]])
        return position, max_sim

    def get_navigable_goal_position(
        self,
        target_image: np.ndarray,
        agent_pos: Optional[np.ndarray] = None,
        sim_floor: float = 0.20,
        top_k_cluster: int = 200,
        eps_m: float = 0.6,
        min_cluster_size: int = 6,
        snap_distance_m: float = 0.8,
        floor_y_band: float = 1.5,
        floor_y_offset: float = 0.05,
    ) -> Optional[Tuple[np.ndarray, float, dict]]:
        """
        把"高 cosine 的 Gaussian 中心"转成"可导航的目标视点"：
          1) 从 sims >= sim_floor 中取 top_k_cluster 个 Gaussian 候选；
          2) 在 xyz 上做 DBSCAN(eps=eps_m, min_samples=min_cluster_size)；
          3) 选 sim 加权得分最高的簇，取 sim 加权 centroid 作为 raw_center；
          4) 把 y 投到地面（agent_pos.y 优先，否则 floor_y_global）；
          5) 若给了 agent_pos，在 xz 平面把 goal 朝 agent 方向 snap snap_distance_m
             米（落在 agent 这一侧的"物体边缘"，更接近视点而非物体中心）。

        Returns: (position(3,), max_sim, info_dict) 或 None。
        info_dict 包含: cluster_size, cluster_compactness, cluster_mean_sim,
        raw_center, snapped(bool)。
        """
        if self._means3D is None or self._clip_vectors is None:
            return None
        encode = self._get_clip_encoder()
        v_g = encode(target_image)
        v_g = np.asarray(v_g, dtype=np.float32).reshape(1, -1)
        if v_g.shape[1] != self._clip_vectors.shape[1]:
            return None
        sims = np.dot(self._clip_vectors, v_g.T).squeeze(-1)
        max_sim_global = float(sims.max())

        n_total = sims.shape[0]
        cand_mask = sims >= sim_floor
        if not bool(cand_mask.any()):
            cand_idx = np.argsort(-sims)[: max(min_cluster_size, 32)]
        else:
            cand_idx_full = np.where(cand_mask)[0]
            order_local = np.argsort(-sims[cand_idx_full])
            keep = order_local[: min(top_k_cluster, cand_idx_full.shape[0])]
            cand_idx = cand_idx_full[keep]

        cand_xyz = self._means3D[cand_idx].astype(np.float64)
        cand_sim = sims[cand_idx].astype(np.float64)

        # Floor estimate: 优先 agent.y，回退场景级 1% 分位数。
        if agent_pos is not None:
            floor_y = float(np.asarray(agent_pos, dtype=np.float64).reshape(-1)[1])
        elif self._floor_y_global is not None:
            floor_y = float(self._floor_y_global)
        else:
            floor_y = float(np.percentile(self._means3D[:, 1], 1.0))

        # （多层场景）只保留与当前层 y 接近的候选，避免 cluster 跨楼层。
        y_dev = np.abs(cand_xyz[:, 1] - floor_y)
        layer_mask = y_dev <= floor_y_band
        if int(layer_mask.sum()) >= max(min_cluster_size, 8):
            cand_xyz = cand_xyz[layer_mask]
            cand_sim = cand_sim[layer_mask]
            cand_idx = cand_idx[layer_mask]

        # 聚类：DBSCAN（如不可用则回退至简单 grid clustering）。
        labels = None
        try:
            from sklearn.cluster import DBSCAN  # type: ignore

            db = DBSCAN(eps=float(eps_m), min_samples=int(min_cluster_size))
            labels = db.fit_predict(cand_xyz)
        except Exception:
            labels = None
        if labels is None:
            # Grid fallback: round xyz to eps grid, group by cell, label by cell rank.
            cells = np.floor(cand_xyz / float(eps_m)).astype(np.int64)
            keys = cells[:, 0].astype(np.int64) * 73856093 \
                ^ cells[:, 1].astype(np.int64) * 19349663 \
                ^ cells[:, 2].astype(np.int64) * 83492791
            uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            labels = np.where(cnt[inv] >= int(min_cluster_size), inv, -1)

        # 选最强簇：sum(sim) 作为得分（密度 + 相似度兼顾）。
        info: dict = {
            "cluster_size": 0,
            "cluster_compactness": float("inf"),
            "cluster_mean_sim": float(cand_sim.mean()) if cand_sim.size else 0.0,
            "raw_center": None,
            "snapped": False,
            "n_candidates": int(cand_xyz.shape[0]),
            "n_total": int(n_total),
        }
        unique_labels = [lab for lab in np.unique(labels) if lab != -1]
        if len(unique_labels) > 0:
            best_lab = None
            best_score = -1.0
            for lab in unique_labels:
                m = labels == lab
                score = float(cand_sim[m].sum())
                if score > best_score:
                    best_score = score
                    best_lab = lab
            cl_mask = labels == best_lab
            cl_xyz = cand_xyz[cl_mask]
            cl_sim = cand_sim[cl_mask]
            w = cl_sim - cl_sim.min() + 1e-3
            w = w / float(w.sum())
            raw_center = (cl_xyz * w[:, None]).sum(axis=0)
            info["cluster_size"] = int(cl_xyz.shape[0])
            info["cluster_compactness"] = float(np.linalg.norm(cl_xyz - raw_center, axis=1).mean())
            info["cluster_mean_sim"] = float(cl_sim.mean())
        else:
            # 没有合格簇：回退到 sim 加权 top-K 中心。
            w = cand_sim - cand_sim.min() + 1e-3
            w = w / float(w.sum())
            raw_center = (cand_xyz * w[:, None]).sum(axis=0)
            info["cluster_size"] = int(cand_xyz.shape[0])
            info["cluster_compactness"] = float(
                np.linalg.norm(cand_xyz - raw_center, axis=1).mean()
            )
        info["raw_center"] = np.array(raw_center, dtype=np.float64).tolist()

        # 地面投影：goal.y = floor_y + offset
        position = np.array(raw_center, dtype=np.float64).copy()
        position[1] = floor_y + float(floor_y_offset)

        # 朝 agent snap：在 xz 平面把 goal 拉向 agent 一侧 snap_distance_m 米。
        if agent_pos is not None and snap_distance_m > 0.0:
            ag = np.asarray(agent_pos, dtype=np.float64).reshape(-1)
            if ag.size >= 3:
                dx = float(ag[0]) - float(position[0])
                dz = float(ag[2]) - float(position[2])
                d = float(np.sqrt(dx * dx + dz * dz))
                if d > 1e-3:
                    step = float(min(snap_distance_m, d * 0.5))
                    position[0] += dx / d * step
                    position[2] += dz / d * step
                    info["snapped"] = True

        return position, max_sim_global, info

    def _navigable_cluster_prep(
        self,
        target_image: np.ndarray,
        agent_pos: Optional[np.ndarray],
        sim_floor: float,
        top_k_cluster: int,
        eps_m: float,
        min_cluster_size: int,
        floor_y_band: float,
    ):
        """Shared prep for navigable clustering; returns (sims, max_sim_global, clusters) or None."""
        if self._means3D is None or self._clip_vectors is None:
            return None
        v_g = self._encode_target_clip(target_image)
        if v_g is None:
            return None
        sims = np.dot(self._clip_vectors, v_g.T).squeeze(-1)
        max_sim_global = float(sims.max())
        n_total = sims.shape[0]
        cand_mask = sims >= sim_floor
        if not bool(cand_mask.any()):
            cand_idx = np.argsort(-sims)[: max(min_cluster_size, 32)]
        else:
            cand_idx_full = np.where(cand_mask)[0]
            order_local = np.argsort(-sims[cand_idx_full])
            keep = order_local[: min(top_k_cluster, cand_idx_full.shape[0])]
            cand_idx = cand_idx_full[keep]
        cand_xyz = self._means3D[cand_idx].astype(np.float64)
        cand_sim = sims[cand_idx].astype(np.float64)
        if agent_pos is not None:
            floor_y = float(np.asarray(agent_pos, dtype=np.float64).reshape(-1)[1])
        elif self._floor_y_global is not None:
            floor_y = float(self._floor_y_global)
        else:
            floor_y = float(np.percentile(self._means3D[:, 1], 1.0))
        y_dev = np.abs(cand_xyz[:, 1] - floor_y)
        layer_mask = y_dev <= floor_y_band
        if int(layer_mask.sum()) >= max(min_cluster_size, 8):
            cand_xyz = cand_xyz[layer_mask]
            cand_sim = cand_sim[layer_mask]
            cand_idx = cand_idx[layer_mask]
        labels = None
        try:
            from sklearn.cluster import DBSCAN  # type: ignore

            db = DBSCAN(eps=float(eps_m), min_samples=int(min_cluster_size))
            labels = db.fit_predict(cand_xyz)
        except Exception:
            labels = None
        if labels is None:
            cells = np.floor(cand_xyz / float(eps_m)).astype(np.int64)
            keys = (
                cells[:, 0].astype(np.int64) * 73856093
                ^ cells[:, 1].astype(np.int64) * 19349663
                ^ cells[:, 2].astype(np.int64) * 83492791
            )
            _uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            labels = np.where(cnt[inv] >= int(min_cluster_size), inv, -1)
        clusters: List[Dict[str, Any]] = []
        unique_labels = [lab for lab in np.unique(labels) if lab != -1]
        for lab in unique_labels:
            m = labels == lab
            cl_xyz = cand_xyz[m]
            cl_sim = cand_sim[m]
            cl_idx = cand_idx[m]
            score = float(cl_sim.sum())
            w = cl_sim - cl_sim.min() + 1e-3
            w = w / float(w.sum())
            raw_center = (cl_xyz * w[:, None]).sum(axis=0)
            clusters.append(
                {
                    "label": int(lab),
                    "score": score,
                    "mean_sim": float(cl_sim.mean()),
                    "max_sim": float(cl_sim.max()),
                    "raw_center": np.array(raw_center, dtype=np.float64),
                    "gaussian_indices": cl_idx.astype(np.int64),
                    "cluster_size": int(cl_xyz.shape[0]),
                    "cluster_compactness": float(
                        np.linalg.norm(cl_xyz - raw_center, axis=1).mean()
                    ),
                }
            )
        if not clusters:
            w = cand_sim - cand_sim.min() + 1e-3
            w = w / float(w.sum())
            raw_center = (cand_xyz * w[:, None]).sum(axis=0)
            clusters.append(
                {
                    "label": -1,
                    "score": float(cand_sim.sum()),
                    "mean_sim": float(cand_sim.mean()),
                    "max_sim": float(cand_sim.max()),
                    "raw_center": np.array(raw_center, dtype=np.float64),
                    "gaussian_indices": cand_idx.astype(np.int64),
                    "cluster_size": int(cand_xyz.shape[0]),
                    "cluster_compactness": float(
                        np.linalg.norm(cand_xyz - raw_center, axis=1).mean()
                    ),
                }
            )
        clusters.sort(key=lambda c: -float(c["score"]))
        return sims, max_sim_global, int(n_total), float(floor_y), clusters

    def _finalize_navigable_position(
        self,
        raw_center: np.ndarray,
        floor_y: float,
        floor_y_offset: float,
        agent_pos: Optional[np.ndarray],
        snap_distance_m: float,
    ) -> Tuple[np.ndarray, bool]:
        position = np.array(raw_center, dtype=np.float64).copy()
        position[1] = floor_y + float(floor_y_offset)
        snapped = False
        if agent_pos is not None and snap_distance_m > 0.0:
            ag = np.asarray(agent_pos, dtype=np.float64).reshape(-1)
            if ag.size >= 3:
                dx = float(ag[0]) - float(position[0])
                dz = float(ag[2]) - float(position[2])
                d = float(np.sqrt(dx * dx + dz * dz))
                if d > 1e-3:
                    step = float(min(snap_distance_m, d * 0.5))
                    position[0] += dx / d * step
                    position[2] += dz / d * step
                    snapped = True
        return position, snapped

    def cluster_thumbnail_rgb(
        self,
        gaussian_indices: np.ndarray,
        patch_size: int = 128,
    ) -> Optional[np.ndarray]:
        """RGB thumbnail from mean color of cluster gaussians (for VLM candidate)."""
        if self._rgb_colors is None or gaussian_indices is None or gaussian_indices.size == 0:
            return None
        try:
            import cv2
        except Exception:
            return None
        idx = np.asarray(gaussian_indices, dtype=np.int64).reshape(-1)
        rgb = self._rgb_colors[idx]
        mean_rgb = np.clip(rgb.mean(axis=0), 0.0, 1.0)
        patch = (np.ones((patch_size, patch_size, 3), dtype=np.float32) * mean_rgb)
        return (patch * 255.0).astype(np.uint8)

    def get_topk_navigable_cluster_candidates(
        self,
        target_image: np.ndarray,
        top_k: int = 3,
        agent_pos: Optional[np.ndarray] = None,
        sim_floor: float = 0.20,
        top_k_cluster: int = 200,
        eps_m: float = 0.6,
        min_cluster_size: int = 6,
        snap_distance_m: float = 0.8,
        floor_y_band: float = 1.5,
        floor_y_offset: float = 0.05,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Top-k navigable SG clusters (discrete candidates), each with position + thumbnail.
        """
        prep = self._navigable_cluster_prep(
            target_image,
            agent_pos,
            sim_floor,
            top_k_cluster,
            eps_m,
            min_cluster_size,
            floor_y_band,
        )
        if prep is None:
            return None
        _sims, max_sim_global, n_total, floor_y, clusters = prep
        out: List[Dict[str, Any]] = []
        for cl in clusters[: max(1, int(top_k))]:
            position, snapped = self._finalize_navigable_position(
                cl["raw_center"],
                floor_y,
                floor_y_offset,
                agent_pos,
                snap_distance_m,
            )
            thumb = self.cluster_thumbnail_rgb(cl["gaussian_indices"])
            out.append(
                {
                    "position": position,
                    "max_sim": max_sim_global,
                    "mean_sim": float(cl["mean_sim"]),
                    "cluster_score": float(cl["score"]),
                    "cluster_size": int(cl["cluster_size"]),
                    "cluster_compactness": float(cl["cluster_compactness"]),
                    "thumbnail_rgb": thumb,
                    "snapped": bool(snapped),
                    "n_candidates": int(n_total),
                }
            )
        return out if out else None

    def get_navigable_goal_position_from_cluster(
        self,
        cluster: Dict[str, Any],
        agent_pos: Optional[np.ndarray],
        floor_y: float,
        snap_distance_m: float = 0.8,
        floor_y_offset: float = 0.05,
        max_sim_global: float = 0.0,
    ) -> Tuple[np.ndarray, float, dict]:
        position, snapped = self._finalize_navigable_position(
            cluster["raw_center"],
            floor_y,
            floor_y_offset,
            agent_pos,
            snap_distance_m,
        )
        info = {
            "cluster_size": int(cluster.get("cluster_size", 0)),
            "cluster_compactness": float(cluster.get("cluster_compactness", float("inf"))),
            "cluster_mean_sim": float(cluster.get("mean_sim", 0.0)),
            "raw_center": np.array(cluster["raw_center"], dtype=np.float64).tolist(),
            "snapped": bool(snapped),
        }
        return position, float(max_sim_global), info

    def _encode_target_clip(self, target_image: np.ndarray) -> Optional[np.ndarray]:
        """Internal helper to encode target image into 1×D L2-normalized CLIP."""
        if self._clip_vectors is None:
            return None
        encode = self._get_clip_encoder()
        v_g = encode(target_image)
        v_g = np.asarray(v_g, dtype=np.float32).reshape(1, -1)
        if v_g.shape[1] != self._clip_vectors.shape[1]:
            return None
        return v_g

    def score_positions_color(
        self,
        target_image: np.ndarray,
        positions: np.ndarray,
        radius_m: float = 1.5,
        bins_per_channel: int = 8,
        min_neighbors: int = 30,
        use_xz_only: bool = True,
    ) -> Optional[np.ndarray]:
        """
        路径 1a：颜色直方图先验。对每个候选 position：
          - 取 R 米内 Gaussian 的 rgb_colors，算 HSV 8×8×8 直方图 h_i
          - 与 target_image 的 HSV 直方图 h_t 做 Bhattacharyya 距离
          - score = 1 - bhattacharyya ∈ [0, 1]
        返回 (M,) np.ndarray，若 rgb_colors 不可用或全部候选周围邻居不足则返回 None。
        """
        if self._means3D is None or self._rgb_colors is None:
            return None
        try:
            import cv2
        except Exception:
            return None

        target = np.asarray(target_image)
        if target.ndim != 3 or target.shape[2] < 3:
            return None
        if target.dtype != np.uint8:
            target = (np.clip(target, 0, 1) * 255).astype(np.uint8)
        # target 默认 RGB；HSV 计算用 cv2.COLOR_RGB2HSV 保持口径一致。
        target_hsv = cv2.cvtColor(target[..., :3], cv2.COLOR_RGB2HSV)
        bins = int(bins_per_channel)
        # HSV ranges: H ∈ [0,180), S/V ∈ [0,256)
        h_t = cv2.calcHist(
            [target_hsv], [0, 1, 2], None,
            [bins, bins, bins], [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(h_t, h_t, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        m = positions.shape[0]
        out = np.zeros((m,), dtype=np.float64)

        means = self._means3D
        if use_xz_only:
            means_xz = means[:, [0, 2]]
        for i in range(m):
            p = positions[i]
            if use_xz_only:
                d2 = (means_xz[:, 0] - float(p[0])) ** 2 + (means_xz[:, 1] - float(p[2])) ** 2
            else:
                d2 = (
                    (means[:, 0] - float(p[0])) ** 2
                    + (means[:, 1] - float(p[1])) ** 2
                    + (means[:, 2] - float(p[2])) ** 2
                )
            mask = d2 <= float(radius_m) * float(radius_m)
            n_nb = int(mask.sum())
            if n_nb < int(min_neighbors):
                out[i] = 0.0
                continue
            rgb_local = (self._rgb_colors[mask] * 255.0).astype(np.uint8)
            # 视作 (n_nb, 1, 3) RGB 像素阵列，转 HSV 后算同 bin 直方图。
            rgb_block = rgb_local.reshape(-1, 1, 3)
            hsv_block = cv2.cvtColor(rgb_block, cv2.COLOR_RGB2HSV)
            h_i = cv2.calcHist(
                [hsv_block], [0, 1, 2], None,
                [bins, bins, bins], [0, 180, 0, 256, 0, 256],
            )
            cv2.normalize(h_i, h_i, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            dist = float(cv2.compareHist(h_t, h_i, cv2.HISTCMP_BHATTACHARYYA))
            out[i] = max(0.0, 1.0 - dist)
        return out

    def score_positions(
        self,
        target_image: np.ndarray,
        positions: np.ndarray,
        radius_m: float = 1.5,
        topk_local: int = 20,
        aggregator: str = "topk_mean",
        use_xz_only: bool = True,
    ) -> Optional[np.ndarray]:
        """
        给定一组 3D 候选位置（如 OV 的 instance positions），返回每个位置周围
        radius_m 米内 SG 语义响应作为 spatial prior 分数。

        Args:
            positions: (M, 3)
            radius_m: 取点 radius（在 xz 平面 / 默认）
            topk_local: 局部 sims 取 topk 后再聚合（避免被噪声拖低）
            aggregator: "topk_mean" / "max" / "mean"
            use_xz_only: True 时忽略 y（地面距离），更适合 IIN

        Returns: (M,) np.ndarray 或 None。
        """
        if self._means3D is None or self._clip_vectors is None:
            return None
        v_g = self._encode_target_clip(target_image)
        if v_g is None:
            return None
        sims = np.dot(self._clip_vectors, v_g.T).squeeze(-1)  # (N,)

        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        m = positions.shape[0]
        out = np.zeros((m,), dtype=np.float64)

        means = self._means3D
        if use_xz_only:
            means_xz = means[:, [0, 2]]
        for i in range(m):
            p = positions[i]
            if use_xz_only:
                d2 = (means_xz[:, 0] - float(p[0])) ** 2 + (means_xz[:, 1] - float(p[2])) ** 2
            else:
                d2 = (means[:, 0] - float(p[0])) ** 2 \
                     + (means[:, 1] - float(p[1])) ** 2 \
                     + (means[:, 2] - float(p[2])) ** 2
            mask = d2 <= float(radius_m) * float(radius_m)
            if not bool(mask.any()):
                out[i] = 0.0
                continue
            s_local = sims[mask]
            if aggregator == "max":
                out[i] = float(s_local.max())
            elif aggregator == "mean":
                out[i] = float(s_local.mean())
            else:  # topk_mean
                k = int(min(topk_local, s_local.size))
                if k <= 0:
                    out[i] = 0.0
                else:
                    idx = np.argpartition(-s_local, k - 1)[:k]
                    out[i] = float(s_local[idx].mean())
        return out
