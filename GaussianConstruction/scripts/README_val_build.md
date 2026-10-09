# 为 val 场景批量建 Semantic Gaussian（方案三）

为逼近论文 Table I 的 non-GT 指标，需要为 HM3D v3 val 的更多场景建带 CLIP 的 Semantic Gaussian（`params.npz` 含 `means3D` + `clip_vectors`）。

## 1. 建图数据要求

- 建图输入在 **configs/data/habitat/** 下，每个场景一个子目录，目录名 = scene_id（如 `CrMo8WxCyVb`）。
- 每个场景目录需包含：
  - `transforms.json`（NeRFStudio 格式：相机内参 + 每帧 `file_path`、`depth_file_path`、`transform_matrix`）
  - `rgb/` 下对应图片
  - 深度与 seg 路径与 `transforms.json` 中一致（如 `depth/`、`seg/` 等）。

若 val 场景的建图数据尚未采集，需要先在 Habitat 中按该格式采集（或从现有数据集中整理）。

## 2. 单场景建图（带 CLIP）

```bash
cd /path/to/GaussianConstruction
python scripts/habitat_splatam.py configs/habitat/habitat_splatam.py --sequence CrMo8WxCyVb --group_name habitat_val_final
```

输出：`experiments/habitat_val_final/CrMo8WxCyVb_seed4_clip/params.npz`（确保 config 中 `use_clip=True`）。

## 3. 批量建图

```bash
cd /path/to/GaussianConstruction

# 方式 A：自动发现 configs/data/habitat 下所有含 transforms.json 的子目录
python scripts/run_build_val_semantic_gaussians.py --data_dir configs/data/habitat_val --group_name habitat_val_final

# 方式 B：从文件读场景列表（每行一个 scene_id）
python scripts/run_build_val_semantic_gaussians.py --scenes_file val_scenes.txt --group_name habitat_val_final

# 跳过已存在 params.npz 的场景（默认）
# 强制全部重跑：加 --no_skip_existing
# 仅打印命令不执行：--dry_run
```

输出目录：`experiments/habitat_val_final/<scene_id>_seed4_clip/params.npz`。

## 4. 在 GaussNav（gaussnav_code_01）中评测

- 将 **gaussian_map_root** 指向上述建图根目录（例如 `experiments/habitat_val_final` 的**绝对路径**）。
- 在 `instance_imagenav.yaml` 中：
  - `use_gt_goal_localization: False`
  - `use_semantic_gaussian_goal: True`
  - `gaussian_map_root: "/path/to/GaussianConstruction/experiments/habitat_val_final"`
- `content_scenes` 留空或注释以跑全 val；若只测已建图场景，可设为如 `["CrMo8WxCyVb", ...]`。
- 运行 `python run.py` 进行 non-GT 评测，对比论文 Table I（succ≈0.725, spl≈0.578）。

## 5. 说明

- 建图必须在 **gaussnav_code** 的 GaussianConstruction 中跑（带 `use_clip`）；gaussnav_code_01 的 Construction 无 CLIP 实现。
- SemanticGaussianGoal 会优先匹配目录名含 `_clip` 的子目录，并读取该目录下 `params.npz` 的 `means3D` 与 `clip_vectors`。
