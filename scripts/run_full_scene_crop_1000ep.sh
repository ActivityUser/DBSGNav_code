#!/usr/bin/env bash
# 复现 full_scene_crop 的 1000 episode 验证集评测。
# 对应历史日志：
#   GaussianNavigation/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_probe/1000(0.8-0.6_inst_v3_vlm)/2/full_scene_crop_1000ep.nohup
# 该次运行使用配置 full_scene_crop.yaml，入口为 run_val.py，--max_episodes 1000。
# 数据、地图和 VLM 与当时一致时，日志末尾指标为：
#   succ / spl / dist = 0.80600 / 0.74618 / 1.53373
#
# 用法（在仓库根目录）：
#   bash scripts/run_full_scene_crop_1000ep.sh [GPU编号]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-0}"
exec bash "$ROOT/scripts/reproduce_archived_navigation.sh" full_scene_crop "$GPU" 1000
