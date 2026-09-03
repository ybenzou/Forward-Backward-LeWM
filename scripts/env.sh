#!/usr/bin/env bash
# Path lock: all FBLeWM artifacts stay under FBLEWM_ROOT.
# Usage:  source /home/yuanben/WorldModel/FBLeWM/scripts/env.sh

export FBLEWM_ROOT="${FBLEWM_ROOT:-/home/yuanben/WorldModel/FBLeWM}"
# Always lock artifacts under FBLeWM (do not inherit LeWM's STABLEWM_HOME/HF_HOME).
export STABLEWM_HOME="$FBLEWM_ROOT/.stable-wm"
export HF_HOME="$FBLEWM_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
# Shared dataset root for PushT / Cube / TwoRoom (do not copy tens of GB).
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/home/yuanben/WorldModel/LeWM/data}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
unset HF_ENDPOINT 2>/dev/null || true
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p \
  "$FBLEWM_ROOT/logs/runs" \
  "$FBLEWM_ROOT/outputs/hydra" \
  "$FBLEWM_ROOT/outputs/eval" \
  "$FBLEWM_ROOT/outputs/diag" \
  "$FBLEWM_ROOT/outputs/checkpoints" \
  "$FBLEWM_ROOT/data/incoming" \
  "$FBLEWM_ROOT/data/extracted" \
  "$STABLEWM_HOME/checkpoints" \
  "$STABLEWM_HOME/datasets" \
  "$HUGGINGFACE_HUB_CACHE"

# Soft-link dataset names into STABLEWM_HOME/datasets for eval HDF5 tooling.
if [[ -d "$LOCAL_DATASET_DIR" ]]; then
  for name in pusht_expert_train.h5 pusht_expert_train.lance; do
    src="$LOCAL_DATASET_DIR/$name"
    link="$STABLEWM_HOME/datasets/$name"
    if [[ -e "$src" && ! -e "$link" ]]; then
      ln -s "$(readlink -f "$src")" "$link" 2>/dev/null || true
    fi
  done
fi

echo "[env.sh] FBLEWM_ROOT=$FBLEWM_ROOT"
echo "[env.sh] STABLEWM_HOME=$STABLEWM_HOME"
echo "[env.sh] LOCAL_DATASET_DIR=$LOCAL_DATASET_DIR"
echo "[env.sh] HF_HOME=$HF_HOME"
