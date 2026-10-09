"""
创新点4：语义-几何融合匹配策略

原策略：仅几何相似度（DISK+LightGlue 特征匹配）
本文创新：加权融合  总分 = α·语义相似度 + β·几何相似度

- 语义相似度：CLIP 余弦相似度，已大致在 [0,1] 或 [-1,1]
- 几何相似度：匹配特征点数量等，需归一化到可比范围
- 支持 minmax / max 归一化，可配置 α、β（或通过网格搜索确定）
"""

import numpy as np
from typing import List, Tuple, Optional, Literal


def normalize_geometric(
    geo_scores: np.ndarray,
    mode: Literal["minmax", "max", "none"] = "minmax",
    max_possible: Optional[float] = None,
) -> np.ndarray:
    """
    将几何得分归一化到 [0, 1] 左右，便于与语义相似度加权融合。

    Args:
        geo_scores: 几何得分数组，如 DISK+LightGlue 匹配点数量
        mode: "minmax" 按当前 batch 最小/最大做线性缩放；
              "max" 用 max_possible 或当前 max 除；
              "none" 不归一化（调用方需保证量纲一致）
        max_possible: mode=="max" 时使用的分母；为 None 则用 geo_scores.max() 或 1.0

    Returns:
        与 geo_scores 同 shape 的归一化数组
    """
    geo = np.asarray(geo_scores, dtype=np.float64)
    if mode == "none":
        return geo
    if mode == "minmax":
        g_min, g_max = geo.min(), geo.max()
        if g_max > g_min:
            return (geo - g_min) / (g_max - g_min)
        return np.ones_like(geo)
    # mode == "max"
    denom = max_possible if max_possible is not None else max(geo.max(), 1.0)
    return np.clip(geo / denom, 0.0, 1.0)


def fuse_scores(
    semantic_scores: np.ndarray,
    geometric_scores: np.ndarray,
    alpha: float,
    beta: float,
    normalize_geometric_mode: Literal["minmax", "max", "none"] = "minmax",
    geometric_max_possible: Optional[float] = None,
    use_semantic: bool = True,
    use_geometric: bool = True,
) -> np.ndarray:
    """
    加权融合：总分 = (use_semantic ? α·语义 : 0) + (use_geometric ? β·几何_norm : 0)。

    语义分一般已是余弦相似度，不做归一化；几何分按 normalize_geometric_mode 归一化。
    α、β 不强制 α+β=1；关闭某一路时该路不参与求和，权重仍保留在配置中供另一路使用。

    Args:
        semantic_scores: 语义相似度（如 CLIP cosine），与 geometric_scores 同长度
        geometric_scores: 几何得分（如匹配点数量）
        alpha: 语义权重
        beta: 几何权重
        normalize_geometric_mode: 几何归一化方式
        geometric_max_possible: "max" 模式下的分母
        use_semantic: False 时不把语义项计入融合分（等价于该项为 0）
        use_geometric: False 时不把几何项计入融合分（且调用方可跳过几何计算）

    Returns:
        融合后的总分数组，与输入同长度
    """
    sem = np.asarray(semantic_scores, dtype=np.float64)
    geo_raw = np.asarray(geometric_scores, dtype=np.float64)
    if use_geometric:
        geo_norm = normalize_geometric(
            geo_raw, mode=normalize_geometric_mode, max_possible=geometric_max_possible
        )
    else:
        geo_norm = np.zeros_like(sem, dtype=np.float64)
    term_sem = alpha * sem if use_semantic else np.zeros_like(sem, dtype=np.float64)
    term_geo = beta * geo_norm if use_geometric else np.zeros_like(sem, dtype=np.float64)
    return term_sem + term_geo


def fuse_and_select(
    semantic_scores: List[float],
    geometric_scores: List[float],
    alpha: float,
    beta: float,
    normalize_geometric_mode: Literal["minmax", "max", "none"] = "minmax",
    geometric_max_possible: Optional[float] = None,
    use_semantic: bool = True,
    use_geometric: bool = True,
) -> Tuple[int, np.ndarray]:
    """
    对多候选做语义-几何融合并返回最佳索引及融合分数数组。

    Args:
        semantic_scores: 每个候选的语义相似度
        geometric_scores: 每个候选的几何得分
        alpha, beta: 融合权重
        normalize_geometric_mode: 几何归一化方式
        geometric_max_possible: "max" 模式时的分母
        use_semantic / use_geometric: 是否把该路得分计入融合（见 fuse_scores）

    Returns:
        best_index: 融合分数最高的候选索引
        fused_scores: 融合分数数组（numpy）
    """
    sem = np.array(semantic_scores, dtype=np.float64)
    geo = np.array(geometric_scores, dtype=np.float64)
    fused = fuse_scores(
        sem,
        geo,
        alpha,
        beta,
        normalize_geometric_mode=normalize_geometric_mode,
        geometric_max_possible=geometric_max_possible,
        use_semantic=use_semantic,
        use_geometric=use_geometric,
    )
    best_i = int(np.argmax(fused))
    return best_i, fused


def adaptive_weights(
    semantic_scores: np.ndarray,
    geometric_scores: np.ndarray,
    base_alpha: float = 0.3,
    base_beta: float = 0.7,
    texture_richness_estimate: Optional[float] = None,
) -> Tuple[float, float]:
    """
    简单规则的自适应权重：几何得分整体较高时略提高 β，否则略提高 α。
    texture_richness_estimate in [0,1] 可外部传入（目标图纹理丰富度估计）。

    Returns:
        (alpha, beta) 归一化后满足 alpha + beta = 1
    """
    alpha, beta = base_alpha, base_beta
    if texture_richness_estimate is not None:
        if texture_richness_estimate > 0.6:
            beta = min(1.0, base_beta * 1.2)
        else:
            alpha = min(1.0, base_alpha * 1.2)
    geo = np.asarray(geometric_scores)
    if geo.size > 0 and geo.max() > 0:
        mean_geo = float(geo.mean())
        if mean_geo > geo.max() * 0.5:
            beta = min(1.0, beta * 1.1)
    total = alpha + beta
    return alpha / total, beta / total
