"""
创新点5：开放词汇评估基准

原评估：在 HM3D 的标准 IIN 测试集上评估（6 个 COCO 室内类别）
本文创新：构建零样本测试集——从 HM3D 中选取 COCO 类别外的物体，用于评估开放词汇能力。

- 标准 IIN 6 类：chair, couch, plant (potted plant), bed, toilet, tv
- 零样本测试集：仅保留 object_category 不在上述 6 类中的 episode，或不在 COCO 80 类中（可选）
- 指标：与标准 IIN 相同——Success、SPL（Anderson et al.），便于公平对比
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from typing import Any, Dict, List, Optional, Set

# 标准 IIN（Krantz et al.）使用的 6 个室内类别，与 COCO 子集一致
STANDARD_IIN_6_CATEGORIES: Set[str] = {
    "chair",
    "couch",
    "sofa",  # 常见别名
    "plant",
    "potted plant",
    "bed",
    "toilet",
    "tv",
    "television",
}
# 归一化：统一小写
STANDARD_IIN_6_CATEGORIES = {s.strip().lower() for s in STANDARD_IIN_6_CATEGORIES}

# COCO 80 类（常见名称，用于“排除全部 COCO”选项时）
COCO_80_CATEGORIES: Set[str] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}
COCO_80_CATEGORIES = {s.strip().lower() for s in COCO_80_CATEGORIES}


def _normalize_category(cat: Optional[str]) -> str:
    if cat is None or not isinstance(cat, str):
        return ""
    return cat.strip().lower()


def filter_episodes_for_open_vocab(
    episodes: List[Any],
    exclude_standard_iin: bool = True,
    exclude_coco80: bool = False,
    exclude_categories: Optional[Set[str]] = None,
) -> List[Any]:
    """
    从 episode 列表中筛出“开放词汇”子集：仅保留目标类别不在排除集内的 episode。

    Args:
        episodes: 元素需有 .object_category 属性（如 InstanceImageGoalNavEpisode）
        exclude_standard_iin: 若 True，排除标准 IIN 的 6 类
        exclude_coco80: 若 True，排除 COCO 80 类（与 exclude_standard_iin 叠加）
        exclude_categories: 额外排除的类别名集合（小写）；若为 None 则仅用上述两项

    Returns:
        满足条件的 episode 列表（原对象引用，未拷贝）
    """
    exclude: Set[str] = set()
    if exclude_standard_iin:
        exclude |= STANDARD_IIN_6_CATEGORIES
    if exclude_coco80:
        exclude |= COCO_80_CATEGORIES
    if exclude_categories:
        exclude |= {_normalize_category(c) for c in exclude_categories}

    out = []
    for ep in episodes:
        cat = _normalize_category(getattr(ep, "object_category", None))
        if cat and cat not in exclude:
            out.append(ep)
    return out


def filter_episode_dicts_for_open_vocab(
    episodes: List[Dict[str, Any]],
    exclude_standard_iin: bool = True,
    exclude_coco80: bool = False,
    exclude_categories: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    与 filter_episodes_for_open_vocab 等价，但针对“字典”形式的 episode（如从 JSON 读入）。
    """
    exclude: Set[str] = set()
    if exclude_standard_iin:
        exclude |= STANDARD_IIN_6_CATEGORIES
    if exclude_coco80:
        exclude |= COCO_80_CATEGORIES
    if exclude_categories:
        exclude |= {_normalize_category(c) for c in exclude_categories}

    out = []
    for ep in episodes:
        cat = _normalize_category(ep.get("object_category"))
        if cat and cat not in exclude:
            out.append(ep)
    return out


def build_open_vocab_dataset_file(
    source_json_gz_path: str,
    output_json_gz_path: str,
    exclude_standard_iin: bool = True,
    exclude_coco80: bool = False,
    exclude_categories: Optional[Set[str]] = None,
) -> int:
    """
    从现有 InstanceImageNav 格式的 .json.gz 中筛选出开放词汇 episode，并写出新的 .json.gz。
    会同时过滤 goals，只保留被保留 episode 引用的 goal_key。

    Args:
        source_json_gz_path: 源数据集路径（如 val.json.gz 或 content 目录下某 scene.json.gz）
        output_json_gz_path: 输出路径
        exclude_standard_iin: 排除标准 IIN 6 类
        exclude_coco80: 排除 COCO 80 类
        exclude_categories: 额外排除的类别

    Returns:
        保留的 episode 数量
    """
    with gzip.open(source_json_gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    goals = data.get("goals", {})

    kept = filter_episode_dicts_for_open_vocab(
        episodes,
        exclude_standard_iin=exclude_standard_iin,
        exclude_coco80=exclude_coco80,
        exclude_categories=exclude_categories,
    )
    kept_goal_keys = {ep.get("goal_key") for ep in kept if ep.get("goal_key")}
    goals_out = {k: v for k, v in goals.items() if k in kept_goal_keys}

    out = {
        "episodes": kept,
        "goals": goals_out,
    }
    if "content_scenes_path" in data:
        out["content_scenes_path"] = data["content_scenes_path"]

    os.makedirs(os.path.dirname(os.path.abspath(output_json_gz_path)) or ".", exist_ok=True)
    with gzip.open(output_json_gz_path, "wt", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return len(kept)


def build_open_vocab_from_content_dir(
    content_dir: str,
    output_dir: str,
    exclude_standard_iin: bool = True,
    exclude_coco80: bool = False,
) -> int:
    """
    遍历 content 目录下所有 .json.gz，逐个筛选并写出到 output_dir，文件名保持不变。
    用于从 HM3D content/<scene>.json.gz 生成 open_vocab_content/<scene>.json.gz。

    Returns:
        保留的 episode 总数
    """
    total = 0
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(content_dir)):
        if not name.endswith(".json.gz"):
            continue
        src = os.path.join(content_dir, name)
        dst = os.path.join(output_dir, name)
        n = build_open_vocab_dataset_file(
            src, dst,
            exclude_standard_iin=exclude_standard_iin,
            exclude_coco80=exclude_coco80,
        )
        total += n
        if n > 0:
            print(f"  {name}: {n} open-vocab episodes -> {dst}")
    return total


def get_metrics_description() -> Dict[str, str]:
    """
    开放词汇基准使用的指标与标准 IIN 一致，便于公平对比。
    """
    return {
        "Success": "是否在调用 stop 时位于目标 1.0m 内",
        "SPL": "Success × (optimal_path_length / max(taken_path_length, optimal_path_length))",
        "说明": "与 HM3D 标准 IIN 评估指标相同（Anderson et al.），仅 episode 集合改为 COCO 类别外物体。",
    }


def apply_open_vocab_filter_to_dataset(dataset: Any) -> int:
    """
    对已加载的 InstanceImageNav 数据集做原地过滤，只保留开放词汇 episode，并同步过滤 goals。
    用于在运行时通过配置 use_open_vocab_benchmark 启用零样本评估。

    Args:
        dataset: 需有 .episodes 与 .goals 属性（如 InstanceImageNavDatasetV1）

    Returns:
        过滤后保留的 episode 数量
    """
    episodes = getattr(dataset, "episodes", None)
    goals = getattr(dataset, "goals", None)
    if episodes is None or goals is None:
        return 0
    kept = filter_episodes_for_open_vocab(episodes, exclude_standard_iin=True, exclude_coco80=False)
    kept_goal_keys = {ep.goal_key for ep in kept}
    dataset.episodes = kept
    dataset.goals = {k: v for k, v in goals.items() if k in kept_goal_keys}
    return len(kept)


def main():
    parser = argparse.ArgumentParser(
        description="创新点5：开放词汇 IIN 基准——筛选零样本 episode 并生成新数据集文件"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    # build: 从单个 json.gz 或 content 目录生成开放词汇 json.gz
    build = sub.add_parser("build", help="从源数据集文件/目录生成开放词汇 .json.gz")
    build.add_argument(
        "--source",
        required=True,
        help="源 val.json.gz 或 content 目录路径",
    )
    build.add_argument(
        "--out",
        required=True,
        help="输出 .json.gz 文件路径，或输出目录（当 --source 为 content 目录时）",
    )
    build.add_argument(
        "--exclude_coco80",
        action="store_true",
        help="排除全部 COCO 80 类（默认仅排除标准 IIN 6 类）",
    )
    build.add_argument(
        "--content_dir",
        action="store_true",
        help="若设，--source 视为 content 目录，遍历其下所有 .json.gz 并输出到 --out 目录",
    )
    args = parser.parse_args()

    if args.command == "build":
        if args.content_dir:
            total = build_open_vocab_from_content_dir(
                args.source,
                args.out,
                exclude_standard_iin=True,
                exclude_coco80=args.exclude_coco80,
            )
            print(f"Total open-vocab episodes: {total}")
        else:
            n = build_open_vocab_dataset_file(
                args.source,
                args.out,
                exclude_standard_iin=True,
                exclude_coco80=args.exclude_coco80,
            )
            print(f"Wrote {n} open-vocab episodes to {args.out}")


if __name__ == "__main__":
    main()
