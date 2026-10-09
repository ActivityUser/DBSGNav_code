# 0.8-0.6_dense_v2_nosg_mainline_vlm_gated

This folder contains a 9-row ablation pack for the no-SG-mainline + gated VLM narrative.

- `mainline_nosg.yaml` (Full G6, OV + VLM + dynamic semantic map, std IIN, no SG)
- `wo_ov_nosg.yaml`
- `wo_vlm_nosg.yaml`
- `wo_dyn_nosg.yaml`
- `wo_bench_nosg.yaml`
- `wo_geom_nosg.yaml` (semantic-only, alpha=0.8, beta=0)
- `wo_semantic_nosg.yaml` (geom-only, alpha=0, beta=0.6)
- `w_sg.yaml`
- `replace_ov_with_sg.yaml` (replace OV with SG)

All rows are isolated from `0.8-0.6_dense_v2_nosg_mainline` to avoid contamination.

Gated VLM settings: `open_vocab_api_rerank_accept_mode: score_margin`, `open_vocab_api_rerank_margin_threshold: 0.05`. VLM selections only override rank-1 when fused rank-1/rank-2 scores are close enough.
