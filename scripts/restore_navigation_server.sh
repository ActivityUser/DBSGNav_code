#!/usr/bin/env bash
# Restore the dependencies used by the seven archived navigation evaluations.
# Keep this environment separate from the server's existing environments.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${GAUSSNAV_ENV_PREFIX:-$ROOT/.runtime/gausshabitat}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/.runtime/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.runtime/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.runtime/cache}"
export PYTHONNOUSERSITE=1
# Per-project mirrors; do not change the server's global Conda/pip configuration.
CONDA_FORGE_URL="${GAUSSNAV_CONDA_FORGE_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}"
AIHABITAT_URL="${GAUSSNAV_AIHABITAT_URL:-https://conda.anaconda.org/aihabitat}"
PIP_MIRROR="${GAUSSNAV_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_LINKS="${GAUSSNAV_TORCH_FIND_LINKS:-https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html}"
LIGHTGLUE_COMMIT=746fac2c042e05d1865315b1413419f1c1e7ba55
LIGHTGLUE_SOURCE="${GAUSSNAV_LIGHTGLUE_SOURCE:-https://codeload.github.com/cvg/LightGlue/zip/$LIGHTGLUE_COMMIT}"
MODE="${1:-install}"
case "$MODE" in install|verify|plan) ;; *) echo 'Usage: bash scripts/restore_navigation_server.sh [install|verify|plan]' >&2; exit 2;; esac
command -v conda >/dev/null || { echo 'Conda is required.' >&2; exit 1; }
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
# conda-forge calls this package "quaternion"; "numpy-quaternion" is its PyPI name.
# Version 2023.0.4 has a Python 3.8 build compatible with NumPy 1.24.4.
CONDA_SPECS=(python=3.8.20 pip=24.2 numpy=1.24.4 habitat-sim=0.2.3 headless quaternion=2023.0.4)

create_conda() {
  if conda create "$@" --prefix "$ENV_PREFIX" --override-channels \
    -c "$AIHABITAT_URL" -c "$CONDA_FORGE_URL" "${CONDA_SPECS[@]}"; then
    return 0
  fi
  echo 'Conda mirror attempt failed; retrying the official channels.' >&2
  conda create "$@" --prefix "$ENV_PREFIX" --override-channels \
    -c https://conda.anaconda.org/aihabitat \
    -c https://conda.anaconda.org/conda-forge "${CONDA_SPECS[@]}"
}

pip_packages() {
  if "$PY" -m pip install --index-url "$PIP_MIRROR" --timeout 45 --retries 2 "$@"; then
    return 0
  fi
  echo 'PyPI mirror attempt failed; retrying pypi.org.' >&2
  "$PY" -m pip install --index-url https://pypi.org/simple --timeout 45 --retries 2 "$@"
}

if [[ "$MODE" == plan ]]; then
  create_conda --dry-run
  exit 0
fi

if [[ "$MODE" == install ]]; then
  if [[ ! -f "$ENV_PREFIX/conda-meta/history" ]]; then
    create_conda -y
  fi
  PY="$ENV_PREFIX/bin/python"
  "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 8), sys.version'
  echo "PyPI mirror: $PIP_MIRROR"
  echo "PyTorch wheel mirror: $TORCH_LINKS"
  TORCH_SPECS=(torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121)
  if ! "$PY" -m pip install --index-url "$PIP_MIRROR" --find-links "$TORCH_LINKS" \
    --timeout 45 --retries 2 "${TORCH_SPECS[@]}"; then
    echo 'PyTorch mirror attempt failed; retrying the official CUDA 12.1 wheel index.' >&2
    "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
      --timeout 45 --retries 2 "${TORCH_SPECS[@]}"
  fi
  pip_packages -r "$ROOT/scripts/requirements-navigation-server.txt"
  "$PY" -m pip install --no-build-isolation --no-deps -e \
    "$ROOT/DBSGNavigation/3rdparty/habitat-lab-0.2.3/habitat-lab"
  # Fixed-commit archive avoids downloading the Git repository's history.
  "$PY" -m pip install --no-build-isolation --no-deps \
    --timeout 45 --retries 2 "$LIGHTGLUE_SOURCE"
fi

PY="$ENV_PREFIX/bin/python"
[[ -x "$PY" ]] || { echo "Environment not installed: $ENV_PREFIX" >&2; exit 1; }
"$PY" -m pip check
cd "$ROOT/DBSGNavigation"
"$PY" - <<'PY'
import sys, torch, habitat_sim, habitat, quaternion, Quaternion
import cv2, skfmm, open_clip
from lightglue import DISK, LightGlue
from runners.instance_imagenav_runner import Runner
print('Python:', sys.version)
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Habitat-Sim:', habitat_sim.__version__)
print('Habitat-Lab:', habitat.__file__)
assert torch.cuda.is_available(), 'Dependencies imported, but CUDA is unavailable; environment NOT ready for evaluation.'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones(4, device='cuda'); assert x.sum().item() == 4
print('Dependency imports and CUDA operation passed. Simulator/data/model smoke run still required.')
PY
echo "Activate: conda activate $ENV_PREFIX"
