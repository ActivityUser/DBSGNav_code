"""导出 GT 语义类别图（COCO 6 类），与已存 rgb/ 帧逐像素对齐（habitat-lab Env 渲染）。

位姿约定（已校准）：sensor 世界位置 = transform_matrix[:3,3] + [0,0.88,0]，
旋转 = quaternion.from_rotation_matrix(transform_matrix[:3,:3])，640x480，hfov=90。
HM3D 细类 -> COCO 6 类（chair/couch/potted plant/bed/toilet/tv）。

输出：<scene>/seg_gt/frame_XXXXX.npy —— int16 类别图，0=背景，否则 COCO id(56/57/58/59/61/62)。

用法（在 GaussianNavigation 目录）：
  HSIM_GPU=0 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  <python> ../GaussianConstruction/scripts/export_gt_semantic.py --scene 4ok3usBNeis [--stride 5]
"""
import argparse
import json
import os
import re

import numpy as np

CFG = "configs/instance_imagenav.yaml"
FRAMES_ROOT = "../GaussianConstruction/configs/data/habitat_val"
W, H, HFOV, CAMH = 640, 480, 90, 0.88
TARGET_COCO = {56, 57, 58, 59, 61, 62}


def hm3d_name_to_coco(name: str):
    """HM3D 细粒度类别名 -> COCO 6 类 id；不匹配返回 0。"""
    n = (name or "").strip().lower()
    if not n:
        return 0
    # 排除"家具/配件"假阳性（tv stand, toilet paper 等）
    EXCL = ["stand", "table", "unit", "cabinet", "console", "mount", "remote",
            "paper", "cleaner", "dispenser", "brush", "roll", "holder", "frame"]
    # tv / monitor（排除 tv stand/table/cabinet 等）
    if any(k in n for k in ["tv", "television", "monitor", "flat panel"]):
        if not any(e in n for e in EXCL):
            return 62
    if "toilet" in n:
        if not any(e in n for e in EXCL):
            return 61
    if n == "bed" or n.endswith(" bed") or "bunk" in n or "crib" in n:
        return 59
    if (n == "plant" or "potted" in n
            or any(k in n for k in ["houseplant", "flowerpot", "flower pot", "plant pot"])):
        return 58
    if any(k in n for k in ["sofa", "couch", "loveseat", "settee"]):
        return 57
    if any(k in n for k in ["chair", "armchair", "recliner"]):
        if not any(e in n for e in EXCL):
            return 56
    return 0


def build_env(scene, gpu):
    import habitat
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
    from habitat import make_dataset

    cfg = get_config(CFG)
    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.split = "val"
        cfg.habitat.dataset.content_scenes = [scene]
        ss = cfg.habitat.simulator.agents.main_agent.sim_sensors
        for name in ["rgb_sensor", "depth_sensor"]:
            if name in ss:
                ss[name].width = W
                ss[name].height = H
                ss[name].hfov = HFOV
                ss[name].position = [0.0, CAMH, 0.0]
        ss["semantic_sensor"] = HabitatSimSemanticSensorConfig(
            width=W, height=H, hfov=HFOV, position=[0.0, CAMH, 0.0]
        )
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu
    ds = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    if len(ds.episodes) == 0:
        raise RuntimeError(f"no val episodes for scene {scene}")
    return habitat.Env(config=cfg, dataset=ds)


def main():
    import quaternion  # noqa

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    gpu = int(os.environ.get("HSIM_GPU", "0"))

    scene_dir = os.path.join(FRAMES_ROOT, args.scene)
    meta = json.load(open(os.path.join(scene_dir, "transforms.json")))
    env = build_env(args.scene, gpu)
    env.reset()
    sim = env.sim

    # 渲染 semantic_id -> COCO id LUT
    id2coco = {}
    mapped_names = {}
    for o in sim.semantic_scene.objects:
        if o is None:
            continue
        try:
            sid = int(o.semantic_id)
            nm = o.category.name()
        except Exception:
            continue
        cid = hm3d_name_to_coco(nm)
        if cid:
            id2coco[sid] = cid
            mapped_names.setdefault(cid, set()).add(nm.strip().lower())
    print(f"[{args.scene}] mapped semantic ids -> 6类: {len(id2coco)}")
    for cid in sorted(mapped_names):
        print(f"   COCO {cid}: {sorted(mapped_names[cid])}")
    maxid = (max(id2coco) + 1) if id2coco else 1
    lut = np.zeros((maxid,), dtype=np.int16)
    for sid, cid in id2coco.items():
        lut[sid] = cid

    out_dir = os.path.join(scene_dir, "seg_gt")
    os.makedirs(out_dir, exist_ok=True)
    agent = sim.get_agent(0)
    frames = meta["frames"]
    if args.stride > 1:
        frames = frames[:: args.stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    n = 0
    for fr in frames:
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        pos = c2w[:3, 3]
        q = quaternion.from_rotation_matrix(c2w[:3, :3])
        st = agent.get_state()
        st.position = pos
        st.rotation = q
        for s in st.sensor_states.values():
            s.position = pos + np.array([0.0, CAMH, 0.0])
            s.rotation = q
        agent.set_state(st, infer_sensor_states=False)
        sem = np.asarray(sim.get_sensor_observations()["semantic"]).astype(np.int64)
        sem_clip = np.clip(sem, 0, maxid - 1)
        cat = lut[sem_clip].astype(np.int16)
        cat[sem >= maxid] = 0
        fid = re.sub(r".*frame_(\d+).*", r"\1", os.path.basename(fr["file_path"]))
        np.save(os.path.join(out_dir, f"frame_{fid}.npy"), cat)
        n += 1
    print(f"[{args.scene}] saved {n} GT semantic maps -> {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
