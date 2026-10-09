#!/usr/bin/env python3
"""
补跑创新点1（Grounding DINO + SAM）未完成的场景。

用法：
  cd DBSGNavigation
  python scripts/run_seg_missing_scenes.py

会只对 HY1NcmCgn3n 和 LT9Jq6dN3Ea 两个场景运行 seg_train_imgs。
"""

import os
import subprocess
import sys

_NAV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_NAV_ROOT)
sys.path.insert(0, _NAV_ROOT)

SCENES_ROOT = os.path.normpath(os.path.join(_NAV_ROOT, "../DBSGConstruction/configs/data/habitat"))
CONFIG = os.path.join(_NAV_ROOT, "configs", "seg_open_vocab.yaml")
MISSING_SCENES = "HY1NcmCgn3n,LT9Jq6dN3Ea"


def main():
    if not os.path.isdir(SCENES_ROOT):
        print(f"Error: 场景根目录不存在: {SCENES_ROOT}")
        return 1
    if not os.path.isfile(CONFIG):
        print(f"Error: 配置文件不存在: {CONFIG}")
        return 1
    cmd = [
        sys.executable,
        "gaussian_utils/seg_train_imgs.py",
        "--config", CONFIG,
        "--scenes_root", SCENES_ROOT,
        "--scenes", MISSING_SCENES,
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=_NAV_ROOT)


if __name__ == "__main__":
    sys.exit(main())
