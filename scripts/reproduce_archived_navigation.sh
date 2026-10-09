#!/usr/bin/env bash
# Reuse the migrated YAML files, with only server paths/API endpoint adapted.
# Existing experiment logs are never used as output destinations.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAV="$ROOT/GaussianNavigation"
PRESET="${1:-full_scene_crop}"
GPU="${2:-0}"
EP="${3:-2}"
MODE="${4:-run}"
case "$MODE" in run|prepare) ;; *) echo 'Fourth argument must be run or prepare.' >&2; exit 2;; esac
case "$PRESET" in full_scene_crop|wo_bench|wo_dyn|wo_ov|wo_sg_fusion_g6|wo_sg_geom|wo_sg_semgeom) ;; *) echo "Unknown preset: $PRESET" >&2; exit 2;; esac
[[ "$EP" =~ ^[1-9][0-9]*$ ]] || { echo 'Episode count must be positive.' >&2; exit 2; }
PY="${GAUSSNAV_PYTHON:-${GAUSSNAV_ENV_PREFIX:-$ROOT/.runtime/gausshabitat}/bin/python}"
[[ -x "$PY" ]] || { echo "Install the environment first: bash $ROOT/scripts/restore_navigation_server.sh" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.runtime/cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
# Set before Python imports so huggingface_hub sees the selected endpoint.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CFG_SOURCE="$NAV/experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_clip_v2_probe/0.8-0.6_inst_v3_vlm/$PRESET.yaml"
mkdir -p "$NAV/experiments/server_restore"
OUT="$(mktemp -d "$NAV/experiments/server_restore/${PRESET}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
"$PY" - "$CFG_SOURCE" "$OUT/config.yaml" "$NAV" <<'PY'
import os, sys, yaml
from pathlib import Path
src, dst, nav = map(Path, sys.argv[1:])
cfg = yaml.safe_load(src.read_text())
env = cfg['end2end_imagenav']
env['checkpoint_folder'] = str(nav / 'data/new_checkpoints')
env['dump_location'] = str(dst.parent / 'examples')
for key in ('gaussian_map_root', 'base_scene_path'):
    env[key] = str((nav / env[key]).resolve())
for var, key in [('GAUSSNAV_VLM_URL', 'semantic_gaussian_api_rerank_url'),
                 ('GAUSSNAV_VLM_MODEL', 'semantic_gaussian_api_rerank_model'),
                 ('GAUSSNAV_CLIP_WEIGHTS', 'clip_pretrained_path')]:
    if os.environ.get(var):
        env[key] = os.environ[var]
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY
{
  printf 'source_config=%s\nconfig=%s\ngpu=%s\nepisodes=%s\npython=%s\n' "$CFG_SOURCE" "$OUT/config.yaml" "$GPU" "$EP" "$PY"
  printf 'command: '; printf '%q ' "$PY" -u run_val.py --config "$OUT/config.yaml" --max_episodes "$EP" --log_file "$OUT/${PRESET}_${EP}ep.log"; printf '\n'
  sha256sum "$CFG_SOURCE" "$OUT/config.yaml"
} > "$OUT/command.txt"
cd "$NAV"
echo "Output: $OUT"
[[ "$MODE" == prepare ]] && exit 0
"$PY" - "$OUT/config.yaml" <<'PY'
import json, os, sys, urllib.request, yaml, torch
env = yaml.safe_load(open(sys.argv[1]))['end2end_imagenav']
assert torch.cuda.is_available(), 'CUDA unavailable; evaluation not started.'
if env.get('use_semantic_gaussian_goal') and env.get('semantic_gaussian_api_rerank_enable'):
    url = env['semantic_gaussian_api_rerank_url'].rsplit('/chat/completions', 1)[0] + '/models'
    headers = {}
    api_key = os.environ.get('GAUSSNAV_VLM_API_KEY', '').strip()
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        models = {m['id'] for m in json.load(response)['data']}
    assert env['semantic_gaussian_api_rerank_model'] in models, (
        'VLM model ID not served. Set GAUSSNAV_VLM_MODEL to the ID returned by /v1/models.'
    )
PY
exec "$PY" -u run_val.py --config "$OUT/config.yaml" --max_episodes "$EP" --log_file "$OUT/${PRESET}_${EP}ep.log"
