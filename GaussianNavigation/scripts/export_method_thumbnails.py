"""Export thumbnail images for method-flow diagram boxes (paper preview).

Reads existing habitat_val / examples / qualitative figs — no navigation rerun.
Output: docs/figs/method_thumbs/<id>.png (lossless, high resolution)
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENE = os.path.join(ROOT, "GaussianConstruction/configs/data/habitat_val/4ok3usBNeis")
CASE_OK = os.path.join(ROOT, "GaussianNavigation/examples/viz_0.8-0.6/0/3")
CASE_FAIL = os.path.join(ROOT, "GaussianNavigation/examples/0/8")
GS_PLOT = os.path.join(
    ROOT,
    "GaussianConstruction/experiments/habitat_val_final/yr17PDCnDDW_seed4_clip/eval/plots/0499.png",
)
OUT = os.path.join(ROOT, "docs/figs/method_thumbs")
# Max canvas for standard nodes (sharp when zoomed in paper / HTML)
THUMB_W, THUMB_H = 640, 480
# Wider panel for side-by-side match visualizations
THUMB_WIDE_W, THUMB_WIDE_H = 1280, 520
MAX_EDGE = 960  # never shrink source below readable detail before padding
CATEGORY_TO_COCO = {
    "chair": "56",
    "sofa": "57",
    "plant": "58",
    "bed": "59",
    "toilet": "61",
    "tv_monitor": "62",
    "tv": "62",
}


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    # remove legacy JPEG previews
    for fn in os.listdir(OUT):
        if fn.lower().endswith(".jpg"):
            os.remove(os.path.join(OUT, fn))


def _fit_image(img_bgr, max_w, max_h, max_edge=MAX_EDGE):
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, max_edge / max(w, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if nw == w and nh == h:
        return img_bgr.copy()
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(img_bgr, (nw, nh), interpolation=interp)


def save_thumb(img_bgr, name: str, *, wide: bool = False):
    if img_bgr is None or img_bgr.size == 0:
        print("skip empty", name)
        return
    cw, ch = (THUMB_WIDE_W, THUMB_WIDE_H) if wide else (THUMB_W, THUMB_H)
    fitted = _fit_image(img_bgr, cw, ch)
    nh, nw = fitted.shape[:2]
    canvas = np.full((ch, cw, 3), 245, dtype=np.uint8)
    y0 = (ch - nh) // 2
    x0 = (cw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = fitted
    path = os.path.join(OUT, f"{name}.png")
    cv2.imwrite(path, canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    print("wrote", path, f"({cw}x{ch})")


def read_rgb(path):
    im = cv2.imread(path)
    return im


def read_goal_info(case_dir):
    info = {}
    gi = os.path.join(case_dir, "goal_info.txt")
    if not os.path.isfile(gi):
        return info
    for line in open(gi):
        line = line.strip()
        if "=" not in line or line.startswith("---"):
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
    return info


def parse_nav_xz(entry):
    """Return navigable (x, z) from instance_retrieval entry."""
    if len(entry) <= 2 or not entry[2]:
        bb = entry[0][0]
        return float((bb[0] + bb[1]) * 0.5), float((bb[4] + bb[5]) * 0.5)
    raw = entry[2]
    cur = raw
    while isinstance(cur, (list, tuple)) and len(cur) == 1:
        cur = cur[0]
    if isinstance(cur, (list, tuple)) and len(cur) >= 2:
        a, b = cur[0], cur[1]
        if isinstance(a, (list, tuple)):
            return parse_nav_xz([entry[0], entry[1], a])
        return float(a), float(b)
    return None, None


def find_instance_entry_for_episode(scene_path, goal_info):
    """
    Match the instance frame used at navigation time via pred_goal_pos (x, z).
    Same logic as open-vocab candidate position in instance_imagenav_env.
    """
    pred_raw = goal_info.get("pred_goal_pos", "")
    if not pred_raw:
        return None
    pred = json.loads(pred_raw.replace(" ", ""))
    if len(pred) < 3:
        return None
    px_t, pz_t = float(pred[0]), float(pred[2])

    ir_path = os.path.join(scene_path, "instance_retrieval.json")
    if not os.path.isfile(ir_path):
        return None
    with open(ir_path) as f:
        ir = json.load(f)

    cat = goal_info.get("object_category", "").lower()
    keys = [CATEGORY_TO_COCO[cat]] if cat in CATEGORY_TO_COCO else list(CATEGORY_TO_COCO.values())

    best = None
    best_d = 1e18
    for key in keys:
        for entry in ir.get(key, []):
            px, pz = parse_nav_xz(entry)
            if px is None:
                continue
            d = (px - px_t) ** 2 + (pz - pz_t) ** 2
            if d < best_d:
                best_d = d
                best = (key, entry)
    if best is None or best_d > 0.05:
        return None
    return best


def load_instance_frame_bgr(scene_path, entry):
    frame_id = entry[1]
    path = os.path.join(scene_path, f"rgb/frame_{frame_id}.png")
    return read_rgb(path)


def resize_to_height(img_bgr, h=400):
    if img_bgr is None:
        return None
    ih, iw = img_bgr.shape[:2]
    nw = max(1, int(iw * h / ih))
    return cv2.resize(img_bgr, (nw, h), interpolation=cv2.INTER_AREA)


def crop_vis_panels(frame_bgr):
    """Split mapper Vis.mp4 frame: [egocentric | occupancy | goal]."""
    h, w = frame_bgr.shape[:2]
    y1, y2 = 12, h - 12
    # three panels separated by ~20px gutters (see mapper.visualize)
    usable = w - 80
    pw = usable // 3
    x0 = 20
    rgb = frame_bgr[y1:y2, x0 : x0 + pw]
    occ = frame_bgr[y1:y2, x0 + pw + 20 : x0 + 2 * pw + 20]
    goal = frame_bgr[y1:y2, x0 + 2 * pw + 40 : x0 + 3 * pw + 40]
    return rgb, occ, goal


def grab_mp4_frame(case_dir, idx=-1):
    import glob

    mp4s = glob.glob(os.path.join(case_dir, "*-Vis.mp4"))
    if not mp4s:
        return None
    cap = cv2.VideoCapture(mp4s[0])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    i = n - 1 if idx < 0 else min(idx, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def crop_eval_plot(path, row, col):
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    # 2x3 grid with titles — approximate equal cells
    ch, cw = h // 2, w // 3
    r0, c0 = row * ch + 28, col * cw + 8
    cell = im[r0 : r0 + ch - 36, c0 : c0 + cw - 12]
    return cell


def instance_montage(scene_dir, n=4):
    ir_path = os.path.join(scene_dir, "instance_retrieval.json")
    if not os.path.isfile(ir_path):
        return None
    with open(ir_path) as f:
        data = json.load(f)
    rgb_dir = os.path.join(scene_dir, "rgb")
    panels = []
    for inst_id in sorted(data.keys(), key=lambda x: int(x))[:12]:
        entries = data[inst_id]
        if not entries:
            continue
        frame_id = entries[0][1]
        fp = os.path.join(rgb_dir, f"frame_{int(frame_id):05d}.png")
        if not os.path.isfile(fp):
            continue
        im = cv2.imread(fp)
        if im is None:
            continue
        panels.append(cv2.resize(im, (120, 90)))
        if len(panels) >= n:
            break
    if not panels:
        return None
    row = cv2.hconcat(panels)
    return row


def orb_match_thumb(goal_bgr, inst_bgr, max_matches=35):
    """Draw matches between goal image and the retrieved exploration frame."""
    g = resize_to_height(goal_bgr, 220)
    i = resize_to_height(inst_bgr, 220)
    if g is None or i is None:
        return None
    g0 = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1500)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(d0) < 8 or len(d1) < 8:
        return np.hstack([g, i])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:max_matches]
    vis = cv2.drawMatches(
        g, k0, i, k1, matches, None,
        matchColor=(0, 200, 0), singlePointColor=(80, 80, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis


def fusion_bar_thumb():
    scores = np.array([0.42, 0.38, 0.31, 0.27])
    fig, ax = plt.subplots(figsize=(4.0, 2.5), dpi=200)
    ax.barh(range(len(scores)), scores, color=["#2c5282", "#4a7ab8", "#7aa3d4", "#a8c4e8"])
    ax.set_xlim(0, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("fused score", fontsize=7)
    fig.tight_layout(pad=0.2)
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape((h, w, 3))
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def goal_source_thumb():
    src = os.path.join(ROOT, "docs/figs/fig_a3_goal_source.png")
    if os.path.isfile(src):
        return cv2.imread(src)
    return None


def main():
    ensure_out()

    # --- offline ---
    # align with viz_scene_exploration / viz_open_vocab_segmentation (frame_00330)
    save_thumb(read_rgb(os.path.join(SCENE, "rgb/frame_00330.png")), "scene_exploration")
    save_thumb(read_rgb(os.path.join(SCENE, "seg_semantic/frame_00330.png")), "ov_segmentation")
    save_thumb(crop_eval_plot(GS_PLOT, 0, 0), "semantic_gaussian_map")
    save_thumb(instance_montage(SCENE), "instance_database")

    # --- online from success case viz ---
    gq = read_rgb(os.path.join(CASE_OK, "goal_query.png"))
    save_thumb(gq, "goal_image")

    fr_mid = grab_mp4_frame(CASE_OK, idx=19)
    fr_end = grab_mp4_frame(CASE_OK, idx=-1)
    if fr_mid is not None:
        rgb, occ, goal_panel = crop_vis_panels(fr_mid)
        save_thumb(rgb, "agent_execution")
        save_thumb(occ, "occupancy_update")
        save_thumb(goal_panel, "multi_candidate")
    if fr_end is not None:
        rgb_e, occ_e, _ = crop_vis_panels(fr_end)
        save_thumb(occ_e, "path_planning")
        save_thumb(occ_e, "goal_position")
        save_thumb(rgb_e, "semantic_geometric_grounding")

    b1 = os.path.join(ROOT, "docs/figs/fig_b1_success_case.png")
    if os.path.isfile(b1):
        save_thumb(cv2.imread(b1), "multi_candidate_alt")

    save_thumb(goal_source_thumb(), "goal_decision")

    # --- Fig.2 SG branch ---
    save_thumb(gq, "clip_encoding")
    save_thumb(crop_eval_plot(GS_PLOT, 1, 0), "similarity_gaussian_map")
    fail_fr = grab_mp4_frame(CASE_FAIL, idx=10)
    if fail_fr is not None:
        _, occ_f, _ = crop_vis_panels(fail_fr)
        save_thumb(occ_f, "spatial_clustering")
        save_thumb(occ_f, "snap_navigable")
        save_thumb(occ_f, "sg_candidate")
    save_thumb(read_rgb(os.path.join(CASE_FAIL, "goal_query.png")), "vlm_rerank")
    if fr_end is not None:
        _, occ_e, _ = crop_vis_panels(fr_end)
        save_thumb(occ_e, "select_cluster")

    # --- Fig.3 OV branch (use success episode matched instance, not random frame) ---
    ok_info = read_goal_info(CASE_OK)
    ok_scene = os.path.join(
        ROOT,
        "GaussianConstruction/configs/data/habitat_val",
        ok_info.get("scene", "4ok3usBNeis"),
    )
    matched = find_instance_entry_for_episode(ok_scene, ok_info)
    inst_row = instance_montage(ok_scene, n=3)
    save_thumb(inst_row, "instance_retrieval")
    if gq is not None and matched is not None:
        _, entry = matched
        inst_frame = load_instance_frame_bgr(ok_scene, entry)
        print(
            "[geometric_matching] scene=%s frame=%s category=%s"
            % (ok_info.get("scene"), entry[1], ok_info.get("object_category"))
        )
        save_thumb(orb_match_thumb(gq, inst_frame), "geometric_matching", wide=True)
        save_thumb(inst_frame, "best_candidate")
    elif gq is not None:
        print("[geometric_matching] WARN: no instance match for success case; skip")
    if fr_end is not None:
        _, occ_ov, _ = crop_vis_panels(fr_end)
        save_thumb(occ_ov, "ov_candidate")
    save_thumb(fusion_bar_thumb(), "score_fusion")
    # semantic similarity: retrieval bar from clip (proxy)
    fig, ax = plt.subplots(figsize=(4.0, 2.5), dpi=200)
    ax.barh([0, 1, 2], [0.81, 0.76, 0.69], color="#c05621")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("CLIP cosine", fontsize=7)
    fig.tight_layout(pad=0.2)
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape((h, w, 3))
    plt.close(fig)
    save_thumb(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR), "semantic_similarity")

    print("\nDone. Thumbnails in:", OUT)


if __name__ == "__main__":
    main()
