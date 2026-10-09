# 仅作文档/备忘：不要 python checkpoint_resume_snippet.py
#
# 断线续跑时，在 habitat_splatam_val.py 的 config dict 里改这三项（或复制下面括号内键值）：
#
#   load_checkpoint=True
#   checkpoint_time_idx=<N>   # 见下方如何选 N
#   save_checkpoints=True     # 续跑期间建议保持 True，便于再次断线
#
# 如何选 N：
#   cd DBSGConstruction
#   ls experiments/<group_name>/<scene>_seed4_clip/params*.npz
#   ls experiments/<group_name>/<scene>_seed4_clip/keyframe_time_indices*.npy
# 取「两个 glob 里都有同名 N」的最大 N，且 N <= 你中断时的帧号（不确定时取偏小更稳）。
#
# 示例（XB4GS9ShBRE + habitat_val_final，断在 64 帧附近，已有 params50）：
#
#   load_checkpoint=True,
#   checkpoint_time_idx=50,
#
# 方式 A — 批跑自动续（config 可一直保持 load_checkpoint=False）：
#
#   python scripts/run_build_val_semantic_gaussians.py --data_dir ... --group_name habitat_val_final \
#     --config configs/habitat/habitat_splatam_val.py --auto_resume
#
# 方式 B — 单场景 + 命令行指定 N（不必改 habitat_splatam_val.py）：
#
#   python scripts/habitat_splatam.py configs/habitat/habitat_splatam_val.py \
#     --sequence XB4GS9ShBRE --group_name habitat_val_final --checkpoint_time_idx 50
#
# 强制从头跑（忽略 config 里的 load_checkpoint）：加 --fresh
