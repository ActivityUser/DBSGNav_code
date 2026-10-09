创新点5（use_open_vocab_benchmark）价值实验
======================================

目标
----
仅验证创新点5的影响：同一模型配置下，只切换 use_open_vocab_benchmark。

配置
----
- bench_off.yaml: use_open_vocab_benchmark=false
- bench_on.yaml:  use_open_vocab_benchmark=true

其余开关保持一致（SG/OV/动态语义等不变），便于单因素比较。

运行
----
cd DBSGNavigation

# 双卡并行（推荐）
bash experiments/tune_goal_params/20260417_021210/configs/ablation_innov5_benchmark_value/run_ablation_innov5_benchmark_value.sh parallel 1000 0 1

# 单卡串行
bash experiments/tune_goal_params/20260417_021210/configs/ablation_innov5_benchmark_value/run_ablation_innov5_benchmark_value.sh all 0 1000

汇总
----
python experiments/tune_goal_params/20260417_021210/tools/summarize_innov5_benchmark_value.py --ep 1000

输出文件：
experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_innov5_benchmark_value/summary_benchmark_value_1000ep.md
