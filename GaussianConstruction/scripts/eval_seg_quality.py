"""分割质量对比：DINO+SAM (seg_semantic) vs Mask-RCNN (seg_maskrcnn) 相对 GT (seg_gt)。

指标（COCO 6 类，按帧聚合）：
  - IoU（micro：交集和/并集和）
  - 像素 Recall（汇总表）；分类别表含 P/R
  - 实例 Recall（GT 连通域被 IoU>0.5 命中比例）
  - 边界 F1（mask 边界 1px 容差）
仅统计三方都存在的帧。

用法：
  python eval_seg_quality.py --scenes 4ok3usBNeis [--out ../../docs/seg_quality.md]
"""
import argparse
import glob
import os

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_ROOT = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "../configs/data/habitat_val")
)
COCO6 = {56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 61: "toilet", 62: "tv"}


def load_dino_sam(path, shape):
    """seg_semantic .npy: dict{coco_id:[bool masks]} -> {coco_id: bool 2D}."""
    d = np.load(path, allow_pickle=True).item()
    out = {}
    for cid in COCO6:
        masks = d.get(cid)
        acc = np.zeros(shape, dtype=bool)
        if masks:
            for m in masks:
                m = np.asarray(m)
                if m.shape != shape:
                    m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                acc |= m
        out[cid] = acc
    return out


def boundary(mask):
    m = mask.astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    return (m - er).astype(bool)


def boundary_f1(pred, gt, tol=1):
    bp, bg = boundary(pred), boundary(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    bg_d = cv2.dilate(bg.astype(np.uint8), k) > 0
    bp_d = cv2.dilate(bp.astype(np.uint8), k) > 0
    prec = (bp & bg_d).sum() / max(bp.sum(), 1)
    rec = (bg & bp_d).sum() / max(bg.sum(), 1)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def inst_recall(pred, gt):
    """GT 连通域被 pred (同类) IoU>0.5 命中的比例。返回 (hit, total)。"""
    num, lab = cv2.connectedComponents(gt.astype(np.uint8))
    hit = tot = 0
    for k in range(1, num):
        g = lab == k
        if g.sum() < 50:
            continue
        tot += 1
        inter = (g & pred).sum()
        union = (g | pred).sum()
        if union > 0 and inter / union > 0.5:
            hit += 1
    return hit, tot


def eval_method(get_pred_for_cid, scenes, frame_ids_per_scene=None):
    agg = {c: {"i": 0, "u": 0, "tp": 0, "fp": 0, "fn": 0, "bf": [], "ih": 0, "it": 0}
           for c in COCO6}
    n_frames = 0
    for sc in scenes:
        sd = os.path.join(FRAMES_ROOT, sc)
        if frame_ids_per_scene is not None:
            fids = frame_ids_per_scene.get(sc, [])
        else:
            fids = [os.path.basename(gf) for gf in sorted(
                glob.glob(os.path.join(sd, "seg_gt", "frame_*.npy"))
            )]
        for fid in fids:
            gf = os.path.join(sd, "seg_gt", fid)
            if not os.path.isfile(gf):
                continue
            gt = np.load(gf)
            shape = gt.shape
            preds = get_pred_for_cid(sd, fid, shape)
            if preds is None:
                continue
            n_frames += 1
            for c in COCO6:
                g = gt == c
                p = preds[c]
                inter = int((g & p).sum())
                union = int((g | p).sum())
                a = agg[c]
                a["i"] += inter
                a["u"] += union
                a["tp"] += inter
                a["fp"] += int((p & ~g).sum())
                a["fn"] += int((g & ~p).sum())
                if g.any() or p.any():
                    a["bf"].append(boundary_f1(p, g))
                h, t = inst_recall(p, g)
                a["ih"] += h
                a["it"] += t
    return agg, n_frames


def common_frame_ids(scenes, loaders):
    """仅保留 GT 与所有预测方法均存在的帧（公平对比）。"""
    per_scene = {}
    for sc in scenes:
        sd = os.path.join(FRAMES_ROOT, sc)
        gt_fids = {
            os.path.basename(gf)
            for gf in glob.glob(os.path.join(sd, "seg_gt", "frame_*.npy"))
        }
        ok = set(gt_fids)
        for load_fn in loaders:
            scene_ok = set()
            for fid in ok:
                gf = os.path.join(sd, "seg_gt", fid)
                if not os.path.isfile(gf):
                    continue
                shape = np.load(gf).shape
                if load_fn(sd, fid, shape) is not None:
                    scene_ok.add(fid)
            ok &= scene_ok
        per_scene[sc] = sorted(ok)
    return per_scene


def fmt(agg):
    rows = []
    ious, prs, rcs, bfs, irs = [], [], [], [], []
    for c, name in COCO6.items():
        a = agg[c]
        if a["u"] == 0 and a["it"] == 0:
            continue
        iou = a["i"] / a["u"] if a["u"] else float("nan")
        pr = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else float("nan")
        rc = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else float("nan")
        bf = float(np.mean(a["bf"])) if a["bf"] else float("nan")
        ir = a["ih"] / a["it"] if a["it"] else float("nan")
        rows.append((name, iou, pr, rc, bf, ir, a["it"]))
        for L, v in [(ious, iou), (prs, pr), (rcs, rc), (bfs, bf), (irs, ir)]:
            if v == v:
                L.append(v)
  return rows, (
        np.mean(ious) if ious else 0,
        np.mean(rcs) if rcs else 0,
        np.mean(bfs) if bfs else 0,
        np.mean(irs) if irs else 0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="../docs/seg_quality.md")
    ap.add_argument(
        "--dino_subdir",
        default="seg_semantic",
        help="DINO+SAM 掩码目录名，如 seg_semantic_tight（修过检后重生成）",
    )
    ap.add_argument(
        "--dino_label",
        default=None,
        help="结果表中的方法名，默认随 dino_subdir 自动命名",
    )
    ap.add_argument(
        "--protocol",
        choices=("auto", "fifty"),
        default="auto",
        help="fifty=约定 5 场景×10 帧/场景（stride 10），覆盖约 50 帧时不报警",
    )
    args = ap.parse_args()
    dino_tag = args.dino_label or (
        "DINO+SAM (tight)" if "tight" in args.dino_subdir else "DINO+SAM (ours)"
    )

    def dino(sd, fid, shape):
        p = os.path.join(sd, args.dino_subdir, fid + ".npy")
        if not os.path.isfile(p):
            p = os.path.join(sd, args.dino_subdir, fid)
        if not os.path.isfile(p):
            return None
        return load_dino_sam(p, shape)

    def mrcnn(sd, fid, shape):
        p = os.path.join(sd, "seg_maskrcnn", fid)
        if not os.path.isfile(p):
            return None
        m = np.load(p)
        if m.shape != shape:
            m = cv2.resize(m.astype(np.int16), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return {c: (m == c) for c in COCO6}

    shared = common_frame_ids(args.scenes, [mrcnn, dino])
    n_shared = sum(len(v) for v in shared.values())
    n_gt = sum(
        len(glob.glob(os.path.join(FRAMES_ROOT, sc, "seg_gt", "frame_*.npy")))
        for sc in args.scenes
    )

    res = {}
    for tag, fn in [("Mask-RCNN", mrcnn), (dino_tag, dino)]:
        agg, _nf = eval_method(fn, args.scenes, shared)
        rows, means = fmt(agg)
        res[tag] = (rows, means)

    cov_pct = 100.0 * n_shared / max(n_gt, 1)
    lines = ["# 分割质量对比（创新1）\n",
             f"场景: {args.scenes}；DINO 目录: `{args.dino_subdir}`；指标按 COCO 6 类，相对 Habitat GT。\n",
             f"评测帧: **{n_shared}** / GT **{n_gt}**（{cov_pct:.1f}% 覆盖；Mask-RCNN 与 DINO 仅在共有帧上统计）。\n",
             ]
    fifty_ok = args.protocol == "fifty" and 45 <= n_shared <= 55
    if cov_pct < 95.0 and not fifty_ok:
        lines.append(
            f"> ⚠️ 覆盖不足：请对 `{args.dino_subdir}` 用 `regen_seg_semantic_tight.py --stride 1 --no_resume` 补全后再评。\n"
        )
    elif fifty_ok:
        lines.append(
            "> 协议：**50 帧**（每场景 stride-10 与 `seg_gt` 对齐）；Mask-RCNN 与 DINO+SAM tight 同帧公平对比。\n"
        )
    lines.extend([
             "\n## 汇总（类别平均）\n",
             "| Method | mIoU↑ | Pixel-R↑ | Boundary-F1↑ | Inst-Recall↑ |",
             "|--------|------:|---------:|-------------:|-------------:|",
    ])
    for tag in ["Mask-RCNN", dino_tag]:
        m = res[tag][1]
        lines.append(f"| {tag} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} |")
    for tag in ["Mask-RCNN", dino_tag]:
        lines.append(f"\n## {tag} 分类别\n")
        lines.append("| Cat | IoU | P | R | B-F1 | InstRec | #GT inst |")
        lines.append("|-----|----:|--:|--:|-----:|--------:|---------:|")
        for name, iou, pr, rc, bf, ir, it in res[tag][0]:
            lines.append(f"| {name} | {iou:.3f} | {pr:.3f} | {rc:.3f} | {bf:.3f} | {ir:.3f} | {it} |")
    txt = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write(txt)
    print(txt)
    print("written to", args.out)


if __name__ == "__main__":
    main()
