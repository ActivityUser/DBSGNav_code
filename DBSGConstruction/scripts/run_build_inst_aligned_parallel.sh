#!/usr/bin/env bash
# 多 GPU 并行跑 build_instance_aligned_gaussian_clip.py（每卡独立进程，写同一 out_root，子目录互不冲突）。
#
# 用法（默认 4 卡：0 5 6 7）：
#   cd /path/to/DBSGConstruction
#   bash scripts/run_build_inst_aligned_parallel.sh
#
# 自定义 GPU 列表与 in/out：
#   bash scripts/run_build_inst_aligned_parallel.sh \
#     "0 5 6 7" \
#     experiments/habitat_val_dense_clip_v2 \
#     experiments/habitat_val_inst_aligned_v3
#
set -euo pipefail

GC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${GC_ROOT}"

GPUS_STR="${1:-0 5 6 7}"
IN_ROOT="${2:-experiments/habitat_val_dense_clip_v2}"
OUT_ROOT="${3:-experiments/habitat_val_inst_aligned_v3}"
SEED_TAG="_seed4_clip"

read -r -a GPUS <<< "${GPUS_STR}"
N_PROCS="${#GPUS[@]}"
if [[ "${N_PROCS}" -lt 1 ]]; then
  echo "用法: $0 \"<gpu_ids_space_separated>\" [in_root] [out_root]" >&2
  exit 1
fi

LIST_ALL="$(mktemp)"
SPLIT_FILES=()
cleanup() {
  rm -f "${LIST_ALL}" 2>/dev/null || true
  for _f in "${SPLIT_FILES[@]}"; do
    rm -f "${_f}" 2>/dev/null || true
  done
}
trap cleanup EXIT

# 与 build 脚本 discover_scenes 一致：in_root 下 *${SEED_TAG}/params.npz
for d in "${IN_ROOT}"/*"${SEED_TAG}"; do
  [[ -d "${d}" && -f "${d}/params.npz" ]] || continue
  basename "${d}" "${SEED_TAG}"
done | sort > "${LIST_ALL}"

NSC="$(wc -l < "${LIST_ALL}" | tr -d ' ')"
if [[ "${NSC}" -eq 0 ]]; then
  echo "[ERR] 未在 ${IN_ROOT} 下找到任何 *${SEED_TAG}/params.npz" >&2
  exit 1
fi

echo "[INFO] scenes=${NSC}  GPUs=${GPUS[*]}  in=${IN_ROOT}  out=${OUT_ROOT}"

# 轮询分到各卡，平衡大小场景
SPLIT_FILES=()
for ((i = 0; i < N_PROCS; i++)); do
  SPLIT_FILES+=("$(mktemp -t inst_v3_split_XXXXXX)")
done
line_no=0
while IFS= read -r scene; do
  [[ -z "${scene}" ]] && continue
  idx=$((line_no % N_PROCS))
  echo "${scene}" >> "${SPLIT_FILES[idx]}"
  line_no=$((line_no + 1))
done < "${LIST_ALL}"

pids=()
for ((i = 0; i < N_PROCS; i++)); do
  gpu="${GPUS[i]}"
  sf="${SPLIT_FILES[i]}"
  n_lines="$(wc -l < "${sf}" | tr -d ' ')"
  [[ "${n_lines}" -eq 0 ]] && continue
  log="/tmp/inst_aligned_v3_build_gpu${gpu}.log"
  echo "[parallel] gpu=${gpu} scenes=${n_lines} log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec python scripts/build_instance_aligned_gaussian_clip.py \
      --in_root "${IN_ROOT}" \
      --out_root "${OUT_ROOT}" \
      --scenes_file "${sf}" \
      --verbose
  ) > "${log}" 2>&1 &
  pids+=($!)
done

ec=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    ec=1
  fi
done

if [[ "${ec}" -ne 0 ]]; then
  echo "[WARN] 有子进程非 0 退出，请检查 /tmp/inst_aligned_v3_build_gpu*.log" >&2
fi
exit "${ec}"
