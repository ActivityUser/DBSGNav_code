"""
验证加入 CLIP 是否有效果

步骤一：跑两次建图（同一序列、同一 seed，仅 use_clip 不同）
  1) 无 CLIP：config 里 use_clip=False，run_name 会自动带 _no_clip
     python scripts/habitat_splatam.py configs/habitat/habitat_splatam.py
     跑完后得到 experiments/habitat_test/CrMo8WxCyVb_seed4_no_clip/

  2) 有 CLIP：先安装 open_clip：pip install open-clip-torch
     config 里 use_clip=True，run_name 会自动带 _clip
     python scripts/habitat_splatam.py configs/habitat/habitat_splatam.py
     跑完后得到 experiments/habitat_test/CrMo8WxCyVb_seed4_clip/

步骤二：对比重建质量（加 CLIP 后应几乎不变）
  python scripts/eval_clip_effect.py compare \\
    ./experiments/habitat_test/CrMo8WxCyVb_seed4_no_clip/eval \\
    ./experiments/habitat_test/CrMo8WxCyVb_seed4_clip/eval

步骤三：检查 CLIP 是否写入（仅对有 CLIP 的那次 run）
  python scripts/eval_clip_effect.py check ./experiments/habitat_test/CrMo8WxCyVb_seed4_clip

有效果的判定：
  - 重建：compare 里 PSNR/SSIM/LPIPS 的 diff 很小 → CLIP 未破坏重建
  - 语义：check 显示 clip_vectors 非零比例 >0 → CLIP 已写入，可用于检索
"""

import argparse
import os
import numpy as np


def load_metrics(eval_dir):
    """从 eval 目录读取 psnr, l1, ssim, lpips（若存在）。"""
    out = {}
    for name, key in [("psnr.txt", "psnr"), ("l1.txt", "l1"), ("ssim.txt", "ssim"), ("lpips.txt", "lpips")]:
        p = os.path.join(eval_dir, name)
        if os.path.isfile(p):
            arr = np.loadtxt(p)
            out[key] = np.atleast_1d(arr)
            out[f"{key}_mean"] = float(np.mean(out[key]))
    return out


def compare_dirs(eval_dir_a, eval_dir_b):
    """对比两个 eval 目录的指标，用于验证加 CLIP 是否影响重建。"""
    ma = load_metrics(eval_dir_a)
    mb = load_metrics(eval_dir_b)
    print("===== 重建质量对比（加 CLIP 后应与之几乎一致）=====")
    print(f"  dir A: {eval_dir_a}")
    print(f"  dir B: {eval_dir_b}")
    print()
    for key in ["psnr", "l1", "ssim", "lpips"]:
        mean_a = ma.get(f"{key}_mean")
        mean_b = mb.get(f"{key}_mean")
        if mean_a is not None and mean_b is not None:
            diff = mean_b - mean_a
            print(f"  {key.upper():6s}  A={mean_a:.4f}  B={mean_b:.4f}   diff(B-A)={diff:+.4f}")
        else:
            print(f"  {key.upper():6s}  (missing in one or both dirs)")
    print()
    print("结论: 若 diff 很小，说明 CLIP 未损害重建；若 diff 很大需排查。")


def check_clip_and_metrics(run_dir):
    """检查该 run 的 params.npz 中 clip_vectors 及 eval 指标。"""
    params_path = os.path.join(run_dir, "params.npz")
    eval_dir = os.path.join(run_dir, "eval")
    print("===== CLIP 与重建指标检查 =====")
    print(f"  run_dir: {run_dir}")
    print()

    if not os.path.isfile(params_path):
        print("  params.npz 不存在，请先跑完建图并保存。")
        return
    params = np.load(params_path, allow_pickle=True)
    keys = list(params.keys())

    if "clip_vectors" not in keys:
        print("  clip_vectors: 未找到（可能为旧版 ckpt）")
    else:
        cv = params["clip_vectors"]
        if hasattr(cv, "shape"):
            shape = cv.shape
        else:
            shape = np.asarray(cv).shape
        cv = np.asarray(cv)
        nz = np.count_nonzero(np.abs(cv) > 1e-6)
        total = cv.size
        print(f"  clip_vectors shape: {shape}")
        print(f"  非零比例: {nz}/{total} = {nz/max(total,1):.4f}")
        if nz == 0:
            print("  -> 未使用 CLIP 或 CLIP 未成功写入，当前为占位零向量。")
        else:
            print("  -> 已写入 CLIP 语义向量，可用于下游检索。")
    print()

    if os.path.isdir(eval_dir):
        m = load_metrics(eval_dir)
        if m:
            print("  重建指标 (eval):")
            for k in ["psnr", "l1", "ssim", "lpips"]:
                if f"{k}_mean" in m:
                    print(f"    {k.upper():6s} = {m[f'{k}_mean']:.4f}")
        else:
            print("  eval 目录下无 psnr/l1/ssim/lpips 文件。")
    else:
        print("  eval 目录不存在，请跑完整流程以生成评估结果。")


def main():
    parser = argparse.ArgumentParser(description="Verify CLIP effect: compare metrics or check clip_vectors.")
    parser.add_argument("mode", choices=["compare", "check"], help="compare: 对比两个 eval 目录; check: 检查单次 run 的 clip 与指标")
    parser.add_argument("paths", nargs="+", help="compare 时: eval_dir_a eval_dir_b; check 时: run_dir")
    args = parser.parse_args()

    if args.mode == "compare":
        if len(args.paths) < 2:
            print("compare 模式需要两个路径: eval_dir_a eval_dir_b")
            return
        compare_dirs(args.paths[0], args.paths[1])
    else:
        if len(args.paths) < 1:
            print("check 模式需要至少一个 run_dir")
            return
        for run_dir in args.paths:
            check_clip_and_metrics(run_dir.rstrip("/"))


if __name__ == "__main__":
    main()
