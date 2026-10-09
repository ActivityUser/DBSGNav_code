"""可行性验证：用 Habitat 语义传感器在已存帧位姿渲染 GT 语义掩码，并与已存 RGB 对齐校验。

仅用于验证 GT 渲染可行性（创新1 分割质量对比的前置）。
"""
import json
import os
import sys

import numpy as np

SCENE_ID = "data/scene_datasets/hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
SCENE_DATASET = "data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
FRAMES_DIR = "../DBSGConstruction/configs/data/habitat_val/4ok3usBNeis"
OUT = "/tmp/gt_verify"


def make_cfg(w, h, hfov):
    import habitat_sim
    bk = habitat_sim.SimulatorConfiguration()
    bk.scene_id = SCENE_ID
    bk.scene_dataset_config_file = SCENE_DATASET
    bk.enable_physics = False
    bk.gpu_device_id = int(os.environ.get("HSIM_GPU", "7"))

    def sensor(uuid, stype):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [h, w]
        s.position = [0.0, 0.0, 0.0]
        s.hfov = hfov
        return s

    rgb = sensor("rgb", habitat_sim.SensorType.COLOR)
    sem = sensor("semantic", habitat_sim.SensorType.SEMANTIC)
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = [rgb, sem]
    return habitat_sim.Configuration(bk, [ac])


def main():
    import habitat_sim
    import quaternion  # noqa
    import cv2
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(FRAMES_DIR, "transforms.json")))
    w, h = meta["w"], meta["h"]
    fl = meta["fl_x"]
    hfov = float(np.degrees(2 * np.arctan((w / 2.0) / fl)))
    print(f"w={w} h={h} fl={fl} hfov={hfov:.2f}")
    sim = habitat_sim.Simulator(make_cfg(w, h, hfov))

    f0 = meta["frames"][0]
    c2w = np.array(f0["transform_matrix"], dtype=np.float64)
    pos = c2w[:3, 3]
    R = c2w[:3, :3]
    agent = sim.get_agent(0)
    saved = cv2.imread(os.path.join(FRAMES_DIR, "rgb/frame_00000.png"))
    orb = cv2.ORB_create(2000)
    kb, db = orb.detectAndCompute(cv2.cvtColor(saved, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 旋转用 identity（场景已对）；测相机高度偏移（agent 基座 -> 相机）。
    q = quaternion.from_rotation_matrix(R)
    best = (-1, None, None, None)
    for H in [0.0, 0.88, 1.31, 1.5]:
        st = agent.get_state()
        st.position = pos
        st.rotation = q
        for s in st.sensor_states.values():
            s.position = pos + np.array([0.0, H, 0.0])
            s.rotation = q
        agent.set_state(st, infer_sensor_states=False)
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3][:, :, ::-1].copy()
        ka, da = orb.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), None)
        ng = 0
        if da is not None and db is not None:
            ng = len([m for m in bf.match(da, db) if m.distance < 48])
        print(f"  camera_height={H:.2f} ORB_good={ng}")
        if ng > best[0]:
            best = (ng, "H=%.2f" % H, rgb, obs["semantic"].copy())
    ng, name, rgb, sem = best
    print(f"BEST conv={name} ORB_good={ng}  semantic_instances={len(np.unique(sem))}")
    cv2.imwrite(os.path.join(OUT, "render_rgb.png"), rgb)
    cv2.imwrite(os.path.join(OUT, "saved_frame0.png"), saved)
    np.save(os.path.join(OUT, "render_sem.npy"), sem)
    print("saved outputs to", OUT)
    sim.close()


if __name__ == "__main__":
    main()
