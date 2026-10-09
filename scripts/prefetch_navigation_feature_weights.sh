#!/usr/bin/env bash
# Populate exactly the filenames used by Kornia DISK and LightGlue's torch.hub calls.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GAUSSNAV_PYTHON:-${GAUSSNAV_ENV_PREFIX:-$ROOT/.runtime/gausshabitat}/bin/python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.runtime/cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export PYTHONNOUSERSITE=1
CACHE="$TORCH_HOME/hub/checkpoints"
MODE="${1:-all}"
case "$MODE" in all|disk|lightglue) ;; *) echo 'Usage: bash scripts/prefetch_navigation_feature_weights.sh [all|disk|lightglue]' >&2; exit 2;; esac
[[ -x "$PY" ]] || { echo "Environment missing: $PY" >&2; exit 1; }
command -v curl >/dev/null
mkdir -p "$CACHE"

validate() {
  "$PY" - "$1" "$2" <<'PY'
import sys, torch
path, kind = sys.argv[1:]
state = torch.load(path, map_location='cpu', weights_only=True)
assert isinstance(state, dict) and state, 'Empty or invalid checkpoint'
if kind == 'disk':
    from kornia.feature import DISK
    DISK().load_state_dict(state['extractor'], strict=True)
else:
    assert all(isinstance(v, torch.Tensor) for v in state.values()), 'Expected a tensor state dict'
print('Checkpoint structure validated:', path)
PY
}

fetch() (
  name="$1"; kind="$2"; shift 2
  dest="$CACHE/$name"
  if [[ -s "$dest" ]]; then
    validate "$dest" "$kind"
    return 0
  fi
  part="$(mktemp "$CACHE/.${name}.XXXXXX")"
  trap 'rm -f -- "$part"' EXIT
  for url in "$@"; do
    echo "Downloading $name from $url"
    if curl --fail --location --connect-timeout 10 --max-time 180 \
      --retry 1 --output "$part" "$url" && validate "$part" "$kind"; then
      mv -- "$part" "$dest"
      echo "Cached: $dest"
      return 0
    fi
    echo "Download or checkpoint validation failed for $url" >&2
  done
  echo "Could not cache $name. An accessible URL can be supplied via GAUSSNAV_DISK_WEIGHTS_URL or GAUSSNAV_LIGHTGLUE_WEIGHTS_URL." >&2
  return 1
)

if [[ "$MODE" == all || "$MODE" == disk ]]; then
  fetch depth-save.pth disk \
    "${GAUSSNAV_DISK_WEIGHTS_URL:-https://cdn.jsdelivr.net/gh/cvlab-epfl/disk@master/depth-save.pth}" \
    https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth
fi
if [[ "$MODE" == all || "$MODE" == lightglue ]]; then
  fetch disk_lightglue_v0-1_arxiv.pth lightglue \
    "${GAUSSNAV_LIGHTGLUE_WEIGHTS_URL:-https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth}"
fi
echo 'Requested feature weights are cached. CLIP and Qwen3-VL are separate models.'
