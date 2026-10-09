# 标准 IIN 消融实验设计（审阅用）

## 固定协议

- 全表统一：`use_open_vocab_benchmark: false`（标准 IIN，与文献对比口径一致）。
- 其余 G6 量与 `full.yaml` 对齐（train split、`allow_gt_fallback: false` 等见各 yaml）。

## Preset 与含义（共 7 个）

| preset | 论文行含义（摘要） | 与 `full` 的差异要点 |
|--------|-------------------|----------------------|
| `full` | GaussNav++ 锚点 | SG+OV+动态语义；融合 α=0.8, β=0.6 |
| `wo_dyn` | w.o. 动态语义 | `enable_dynamic_semantic_update: false` |
| `wo_bench` | w.o. open-vocab benchmark | `use_open_vocab_benchmark: false`（与 full 在 std 下近似等价，仅作开关一致性） |
| `wo_ov` | w.o. OV 检索，SG only | `use_open_vocab_retrieval: false` |
| `wo_sg_semgeom` | w.o. SG；OV **纯语义**（导师：α=1, β=0） | `use_semantic_gaussian_goal: false`；`open_vocab_semantic_weight: 1.0`；`fusion_geometric_weight: 0.0` |
| `wo_sg_fusion_g6` | w.o. SG；OV **G6 双流融合**（α=0.8, β=0.6） | 同上关 SG；融合权重与 `full` 中 OV 分支一致 |
| `wo_sg_geom` | **主表 Baseline**（导师）：w.o. SG；OV **纯几何**（α=0, β=1） | `open_vocab_semantic_weight: 0.0`；`fusion_geometric_weight: 1.0` |

**已移除**：原 preset `baseline`（无 SG、无 OV，非 GT 下界）。本套件**不再**跑、不配该项 yaml。

**重要（相对你旧 log）**：原先 `wo_sg_semgeom.yaml` 曾为 α=0.8、β=0.6；现已改为导师口径 **α=1、β=0**。旧表「w.o. SG (OV sem.-geom.)」数字**不可**与新 `wo_sg_semgeom` 直接对比；若要与历史比，应看 **`wo_sg_fusion_g6`**（承接原融合权重）。

## 运行顺序（串行 `all`）

`run_ablation_fullsys_std_Agroup_g6_train.sh` 串行顺序为：

1. `wo_sg_geom` → 2. `wo_sg_semgeom` → 3. `wo_sg_fusion_g6` → 4. `wo_ov` → 5. `wo_bench` → 6. `wo_dyn` → 7. `full`

**并行**：7 个 preset 各占 1 卡，不分先后。

## 日志与汇总

- 每 preset：`logs_1000ep_rerun/ablation_fullsys_std_Agroup_g6_train/<preset>_1000ep.log`
- 汇总（表顺序为 **full → wo_dyn → … → wo_sg_geom (Baseline)**，与串行执行顺序不同）：

```bash
cd DBSGNavigation
python experiments/tune_goal_params/20260417_021210/tools/summarize_ablation_fullsys_std_Agroup_g6_train.py \
  --log-dir experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_fullsys_std_Agroup_g6_train \
  --ep 1000
```

未跑完时：`--allow-partial`。

## 代码层面其它已有关改动（见对话 / mapper）

- `mapper.compute_local_semantic_correlation` 已按 **余弦相似度** 计算（此前误用点积），重跑 `full` / `wo_dyn` 时动态语义规划行为会变化。

请确认本设计无误后再执行 `run_ablation_fullsys_std_Agroup_g6_train.sh`。
