# DBSGNav

DBSGNav 的公开代码。仓库只包含源码、评测配置和环境安装脚本。场景数据、高斯地图、模型权重、运行日志和实验结果需要在本机自行准备并重新运行，不会随仓库提供。

代码在 [GaussNav](https://github.com/xiaohanlei/GaussNav) 与 [SplaTAM](https://github.com/spla-tam/SplaTAM) 的基础上实现。导航评测使用 Habitat-Sim 0.2.3 / Habitat-Lab 0.2.3，任务为 HM3D Instance ImageNav 验证集。

## 1. 环境

需要 Linux、NVIDIA 驱动，以及支持 CUDA 12.1 的 GPU。安装 Miniconda 或 Mambaforge 后，在仓库根目录执行：

```bash
bash scripts/restore_navigation_server.sh
```

脚本会在 `.runtime/gausshabitat` 创建独立环境，主要版本为：

- Python 3.8.20
- Habitat-Sim 0.2.3（headless）
- NumPy 1.24.4
- PyTorch 2.1.1+cu121、torchvision 0.16.1+cu121、torchaudio 2.1.1+cu121
- 本仓库内的 Habitat-Lab
- OpenCLIP、DISK、LightGlue（提交 `746fac2c042e05d1865315b1413419f1c1e7ba55`）

国内网络默认使用清华 conda-forge、清华 PyPI 和上海交大 PyTorch wheel 镜像，失败时会回退到官方源。可用这些变量改源：`GAUSSNAV_CONDA_FORGE_URL`、`GAUSSNAV_AIHABITAT_URL`、`GAUSSNAV_PIP_INDEX_URL`、`GAUSSNAV_TORCH_FIND_LINKS`、`GAUSSNAV_LIGHTGLUE_SOURCE`。

安装结束后脚本会检查依赖导入和一次 CUDA 运算。通过后激活环境：

```bash
conda activate .runtime/gausshabitat
```

`environment/gausshabitat_2026-4-26_environment.yml` 和同目录的 pip freeze 是 2026-04-26 的包版本记录，供对照。其中 `habitat-sim`、`lightglue`、`diff-gaussian-rasterization` 在原机器上是本地或 Conda 安装，不能直接用这份 YAML 一键还原。导航评测请使用上面的安装脚本。

若要编译建图用的高斯光栅化扩展：

```bash
cd GaussianConstruction/diff-gaussian-rasterization-w-depth
pip install --no-build-isolation .
```

特征点权重：

```bash
bash scripts/prefetch_navigation_feature_weights.sh
```

CLIP ViT-B-32 会在首次运行时按 `ViT-B-32` / `openai` 下载。脚本默认把 Hugging Face 端点设为 `https://hf-mirror.com`，可用 `HF_ENDPOINT` 覆盖。

## 2. 需要自行准备的数据

以下内容不在仓库中。

| 用途 | 放到 |
| --- | --- |
| HM3D v0.2 场景 | `GaussianNavigation/data/scene_datasets/hm3d_v0.2/` |
| Instance ImageNav HM3D v3 val（1000 episodes，36 个场景） | `GaussianNavigation/data/datasets/instance_imagenav/hm3d/v3/val/` |
| 语义高斯地图 | `GaussianConstruction/experiments/habitat_val_inst_aligned_v3/<scene>_seed4_clip/params.npz` |
| 实例检索与 v2 CLIP 向量 | `GaussianConstruction/configs/data/habitat_val/<scene>/` |

HM3D 和 Instance ImageNav 的下载遵循 [Habitat-Lab 0.2.3 数据集说明](https://github.com/facebookresearch/habitat-lab/blob/v0.2.3/DATASETS.md)。episode 配置读取 `data/datasets/instance_imagenav/hm3d/v3/{split}/{split}.json.gz`，场景目录为 `data/scene_datasets/`，路径都相对 `GaussianNavigation`。

每个场景的检索目录需要包含 `instance_retrieval.json`、`instance_clip_vectors_v2.json`、`transforms.json` 和 `rgb/`。v2 向量用实例掩码裁剪后再编码：

```bash
cd GaussianNavigation
python gaussian_utils/build_instance_clip_vectors_v2.py \
  --scenes_root ../GaussianConstruction/configs/data/habitat_val
```

语义高斯地图由建图结果对齐得到。先准备 `GaussianConstruction/configs/data/habitat_val/<scene>/` 下的 RGB、深度和 `transforms.json`，再在 `GaussianConstruction` 中建图并对齐：

```bash
cd GaussianConstruction
bash scripts/preflight_build_val.sh configs/data/habitat_val habitat_val_dense_clip_v2
python scripts/run_build_val_semantic_gaussians.py \
  --data_dir configs/data/habitat_val \
  --group_name habitat_val_dense_clip_v2 \
  --max_frames -1 \
  --stride 1
bash scripts/run_build_inst_aligned_parallel.sh \
  "0" \
  experiments/habitat_val_dense_clip_v2 \
  experiments/habitat_val_inst_aligned_v3
```

评测配置里的地图根目录是 `../GaussianConstruction/experiments/habitat_val_inst_aligned_v3`，检索根目录是 `../GaussianConstruction/configs/data/habitat_val`。

## 3. 启动 Qwen3-VL

`full_scene_crop` 会在语义高斯候选歧义时调用本机的 Qwen3-VL-8B-Instruct。另开一个终端，用独立环境启动 vLLM。配置中的模型 ID 是 `/dev/shm/changxiaosa_modelcache/Qwen3-VL-8B-Instruct`，服务的 `--served-model-name` 必须与客户端发送的 ID 一致。权重可以放在任意路径：

```bash
vllm serve /path/to/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 8002 \
  --served-model-name /dev/shm/changxiaosa_modelcache/Qwen3-VL-8B-Instruct \
  --limit-mm-per-prompt '{"image":4}' \
  --api-key EMPTY
```

若服务返回的模型 ID 不是上面这个字符串，评测前设置 `GAUSSNAV_VLM_MODEL` 为 `/v1/models` 里的实际 ID。接口和密钥也可以改：

```bash
export GAUSSNAV_VLM_URL=http://127.0.0.1:8002/v1/chat/completions
export GAUSSNAV_VLM_MODEL=/dev/shm/changxiaosa_modelcache/Qwen3-VL-8B-Instruct
export GAUSSNAV_VLM_API_KEY=EMPTY
```

## 4. 复现 full_scene_crop 的 1000 episode 结果

历史日志 `full_scene_crop_1000ep.nohup` 由下面这条评测产生：配置文件为 `full_scene_crop.yaml`，入口为 `GaussianNavigation/run_val.py`，`--max_episodes 1000`。日志里打印的指标是：

```text
succ / spl / dist / nonGT_cov / GT_fb_rate : 0.80600/0.74618/1.53373/1.00000/0.00000(1000)
```

在仓库根目录、Qwen3-VL 已经可访问之后执行。第二个参数是 GPU 编号，默认 0：

```bash
conda activate .runtime/gausshabitat
bash scripts/run_full_scene_crop_1000ep.sh 0
```

这条命令等价于：

```bash
bash scripts/reproduce_archived_navigation.sh full_scene_crop 0 1000
```

它会读取

`GaussianNavigation/experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_clip_v2_probe/0.8-0.6_inst_v3_vlm/full_scene_crop.yaml`

并把检查点目录、可视化输出、地图根目录和检索根目录改写到当前仓库，然后执行：

```bash
cd GaussianNavigation
CUDA_VISIBLE_DEVICES=0 python -u run_val.py \
  --config <改写后的 full_scene_crop.yaml> \
  --max_episodes 1000 \
  --log_file <新的输出目录>/full_scene_crop_1000ep.log
```

输出写到 `GaussianNavigation/experiments/server_restore/` 下新建的目录，里面有本次使用的配置、命令和日志。不会覆盖仓库里的源码。1000 局是完整评测；把最后的 `1000` 改成 `2` 可以先做连通性检查，两局结果不能用来对上面的指标。

也可以直接调用与当时 nohup 相同的启动脚本。此时 YAML 里的模型 ID 必须和服务端一致，并且不要依赖上面的路径改写：

```bash
cd GaussianNavigation
bash experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_clip_v2_probe/0.8-0.6_inst_v3_vlm/run_ablation_inst_v3_vlm.sh \
  full_scene_crop 0 1000
```

同一目录下还有 `wo_dyn`、`wo_bench`、`wo_ov`、`wo_sg_semgeom`、`wo_sg_fusion_g6`、`wo_sg_geom`、`wo_vlm` 等配置，把预设名换成对应文件名即可。

## 5. 目录

- `GaussianNavigation/`：Instance ImageNav 评测、开放词汇检索和语义高斯目标选择。
- `GaussianConstruction/`：语义高斯建图，以及带深度的 diff-gaussian-rasterization。
- `scripts/restore_navigation_server.sh`：安装导航评测环境。
- `scripts/run_full_scene_crop_1000ep.sh`：复现上述 1000 episode 评测。
