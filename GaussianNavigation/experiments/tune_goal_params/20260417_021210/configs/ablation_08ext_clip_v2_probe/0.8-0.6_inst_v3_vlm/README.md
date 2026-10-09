# 0.8-0.6_inst_v3_vlm

Isolated ablation pack: **inst_v3** backbone + **SG-stage** Qwen3-VL-8B + **merit** SG/OV arbitration.

## vs `0.8-0.6_inst_v3`

| Item | inst_v3 | inst_v3_vlm |
|------|---------|-------------|
| SG goal | CLIP top cluster (navigable) | Same + **VLM pick among top-k clusters** when `mean_sim` margin ≤ 0.03 |
| OV goal | G6 fusion (α=0.8, β=0.6) | Same |
| OV-VLM | off (separate `full_api_rerank_*` rows) | **explicitly off** |
| SG/OV fusion | OV-first assist | **`goal_arbitration_mode: merit`** |

## vs `mainline_vlm_gated`

| Item | mainline_vlm_gated | inst_v3_vlm |
|------|-------------------|-------------|
| VLM stage | OV fused top-k rerank | **SG cluster** disambiguation |
| Map | dense_v2 nosg | habitat_val_inst_aligned_v3 |

## Rows (8)

1. `full.yaml` — Full + SG-VLM + merit
2. `wo_dyn.yaml`
3. `wo_bench.yaml`
4. `wo_ov.yaml` — SG only (OV off)
5. `wo_sg_semgeom.yaml`
6. `wo_sg_fusion_g6.yaml`
7. `wo_sg_geom.yaml`
8. `wo_vlm.yaml` — merit + no SG-VLM (`semantic_gaussian_api_rerank_enable: false`)

## Logs

`experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_probe/1000(0.8-0.6_inst_v3_vlm)/`

## Run

```bash
cd experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_clip_v2_probe/0.8-0.6_inst_v3_vlm
bash run_ablation_inst_v3_vlm.sh parallel8
```

Requires Qwen3-VL API at `http://127.0.0.1:8002/v1/chat/completions`.
