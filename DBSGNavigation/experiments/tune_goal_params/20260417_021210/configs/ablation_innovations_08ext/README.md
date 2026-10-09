# 08_ext 参数体系下的创新点消融

这套配置以 `08_ext_gm40_k8.yaml` 为底座，仅按 `docs/ABLATION.md` 打开/关闭创新点 2~6。

## 配置列表

- `baseline.yaml`：创新点 2~6 全关
- `innov2.yaml`：仅创新点2（语义高斯）
- `innov3.yaml`：仅创新点3（跨模态检索，几何优先）
- `innov4.yaml`：创新点3+4（检索+语义几何融合）
- `innov5.yaml`：创新点5（开放词汇评估基准）
- `innov6.yaml`：创新点6（动态语义更新）
- `full.yaml`：创新点2~6全开

> 创新点1（开放词汇分割）属于预处理阶段，不在导航 yaml 里直接开关。

## 运行

```bash
cd DBSGNavigation
bash experiments/tune_goal_params/20260417_021210/configs/ablation_innovations_08ext/run_ablation_08ext.sh all 0 100
```

参数分别为：`preset gpu_id max_episodes`。
