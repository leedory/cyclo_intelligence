#!/usr/bin/env bash
# Reproducible Task_000458 sim+real ACT workflow. Run from any directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$REPO_ROOT/docker/workspace"
SIM_ROOT="$WORKSPACE/rosbag2/Task_000458_Pick_Peanut_Mix_WhiteShelf_SimtoReal_SIM_MCAP"
REAL_ROOT="$WORKSPACE/rosbag2/Task_000458_Pick_Peanut_Mix_WhiteShelf_SimtoReal_REAL_MCAP"
REMOTE_REAL="1050:/home/robotis/cyclo_intelligence/workspace/rosbag2/Task_000458_Pick_Peanut_Mix_WhiteShelf_SimtoReal_MCAP/"
DATASET_ROOT="$WORKSPACE/dataset/robotis/task_000458_peanut_mix_sim_real_act_native_aspect_v30"
DATASET_REPO_ID="robotis/task_000458_peanut_mix_sim_real_act_native_aspect_v30"
# Editable ACT training parameters (shared defaults across all three scripts).
ACT_DEVICE="${ACT_DEVICE:-cuda}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-16}"
ACT_STEPS="${ACT_STEPS:-50000}"
ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-45}"
ACT_N_ACTION_STEPS="${ACT_N_ACTION_STEPS:-45}"
ACT_EVAL_SPLIT="${ACT_EVAL_SPLIT:-0.0}"
ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-10000}"
ACT_LOG_FREQ="${ACT_LOG_FREQ:-100}"
ACT_WANDB_ENABLE="${ACT_WANDB_ENABLE:-false}"
ACT_PUSH_TO_HUB="${ACT_PUSH_TO_HUB:-false}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-4}"
ACT_SEED="${ACT_SEED:-1000}"
ACT_ENV_EVAL_FREQ="${ACT_ENV_EVAL_FREQ:-0}"
ACT_EVAL_STEPS="${ACT_EVAL_STEPS:-0}"
ACT_SAVE_CHECKPOINT="${ACT_SAVE_CHECKPOINT:-true}"
ACT_IMAGE_TRANSFORMS_ENABLE="${ACT_IMAGE_TRANSFORMS_ENABLE:-true}"
ACT_IMAGE_TRANSFORMS_MAX_NUM="${ACT_IMAGE_TRANSFORMS_MAX_NUM:-2}"
ACT_IMAGE_TRANSFORMS_TFS='{"brightness":{"weight":1,"type":"ColorJitter","kwargs":{"brightness":[0.95,1.05]}},"affine":{"weight":1,"type":"RandomAffine","kwargs":{"degrees":[-2,2],"translate":[0.01,0.01]}}}'
ACT_JOB_NAME="${ACT_JOB_NAME:-task_000458_peanut_mix_sim_real_act_native_aspect}"
ACT_OUTPUT_DIR="${ACT_OUTPUT_DIR:-/workspace/model/lerobot/${ACT_JOB_NAME}}"
ACT_LOG_FILE="${ACT_LOG_FILE:-$WORKSPACE/logs/lerobot/${ACT_JOB_NAME}.log}"
MAIN_CONTAINER="${CYCLO_MAIN_CONTAINER_NAME:-cyclo_intelligence_s2r}"
LEROBOT_CONTAINER="${LEROBOT_CONTAINER_NAME:-lerobot_server}"
CONTAINER_REPO="/root/ros2_ws/src/cyclo_intelligence"
CONTAINER_WORKSPACE="/workspace"
CONTAINER_SIM_ROOT="$CONTAINER_WORKSPACE/rosbag2/Task_000458_Pick_Peanut_Mix_WhiteShelf_SimtoReal_SIM_MCAP"
CONTAINER_REAL_ROOT="$CONTAINER_WORKSPACE/rosbag2/Task_000458_Pick_Peanut_Mix_WhiteShelf_SimtoReal_REAL_MCAP"
CONTAINER_DATASET_ROOT="$CONTAINER_WORKSPACE/dataset/robotis/task_000458_peanut_mix_sim_real_act_native_aspect_v30"

usage() {
  cat <<'EOF'
Usage: tools/act_sim_real/run.sh <sync-real|validate|build-dataset|train|all>

sync-real     Read the real MCAP/videos from SSH host 1050 into this checkout.
validate      Check sim/real episodes and the three common camera files.
build-dataset Convert all episodes at 10 Hz, preserving head/wrist aspect ratios.
train         Run fresh ACT training in lerobot_server (start it first with docker/container.sh start-lerobot).
all           sync-real, validate, then build-dataset. Does not start GPU training.
EOF
}

require_container() {
  if ! docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null | grep -qx true; then
    echo "Container is not running: $1" >&2
    exit 1
  fi
}

sync_real() {
  mkdir -p "$REAL_ROOT"
  rsync -a --partial --info=progress2 "$REMOTE_REAL" "$REAL_ROOT/"
}

validate() {
  python3 - "$SIM_ROOT" "$REAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

for domain, root_raw in zip(("sim", "real"), sys.argv[1:], strict=True):
    root = Path(root_raw)
    episodes = sorted(path.parent for path in root.rglob("metadata.yaml") if any(path.parent.glob("*.mcap")) and "segments" not in path.parent.relative_to(root).parts)
    if not episodes:
        raise SystemExit(f"{domain}: no MCAP episodes in {root}")
    for episode in episodes:
        missing = [camera for camera in ("cam_left_head", "cam_left_wrist", "cam_right_wrist")
                   if not any((episode / "videos").glob(f"*/*{camera}.mp4"))]
        if missing:
            raise SystemExit(f"{domain}: {episode} missing {missing}")
        info = json.loads((episode / "episode_info.json").read_text())
        print(f"{domain}: episode={info.get('episode_index', episode.name)} task={info.get('task', '')}")
    print(f"{domain}: {len(episodes)} episodes OK")
PY
}

build_dataset() {
  require_container "$MAIN_CONTAINER"
  if [[ -d "$DATASET_ROOT" ]] && [[ -n "$(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite existing dataset: $DATASET_ROOT" >&2
    exit 1
  fi
  docker exec "$MAIN_CONTAINER" env CYCLO_PREPARED_EPISODE_CACHE_DISABLE=1 CYCLO_V30_SOURCE_AGGREGATE_CACHE_DISABLE=1 CYCLO_V30_DATA_AGGREGATE_CACHE_DISABLE=1 python3 "$CONTAINER_REPO/tools/act_sim_real/convert_sim_real.py" \
    --sim-root "$CONTAINER_SIM_ROOT" \
    --real-root "$CONTAINER_REAL_ROOT" \
    --output "$CONTAINER_DATASET_ROOT" \
    --repo-id "$DATASET_REPO_ID" \
    --robot-config "$CONTAINER_REPO/tools/act_sim_real/config/ffw_sg2_rev1_act_common_cameras.yaml" \
    --fps 10
}

train() {
  require_container "$LEROBOT_CONTAINER"
  if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
    echo "Dataset is missing. Run build-dataset first: $DATASET_ROOT" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$ACT_LOG_FILE")"
  docker exec "$LEROBOT_CONTAINER" lerobot-train --policy.type=act --policy.chunk_size="$ACT_CHUNK_SIZE" --policy.n_action_steps="$ACT_N_ACTION_STEPS" --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$CONTAINER_DATASET_ROOT" --dataset.eval_split="$ACT_EVAL_SPLIT" --dataset.image_transforms.enable="$ACT_IMAGE_TRANSFORMS_ENABLE" --dataset.image_transforms.max_num_transforms="$ACT_IMAGE_TRANSFORMS_MAX_NUM" --dataset.image_transforms.tfs="$ACT_IMAGE_TRANSFORMS_TFS" --policy.push_to_hub="$ACT_PUSH_TO_HUB" --policy.device="$ACT_DEVICE" --batch_size="$ACT_BATCH_SIZE" --num_workers="$ACT_NUM_WORKERS" --seed="$ACT_SEED" --steps="$ACT_STEPS" --save_checkpoint="$ACT_SAVE_CHECKPOINT" --save_freq="$ACT_SAVE_FREQ" --log_freq="$ACT_LOG_FREQ" --env_eval_freq="$ACT_ENV_EVAL_FREQ" --eval_steps="$ACT_EVAL_STEPS" --wandb.enable="$ACT_WANDB_ENABLE" --job_name="$ACT_JOB_NAME" --output_dir="$ACT_OUTPUT_DIR" 2>&1 | tee "$ACT_LOG_FILE"
}

case "${1:-}" in
  sync-real) sync_real ;;
  validate) validate ;;
  build-dataset) build_dataset ;;
  train) train ;;
  all) sync_real; validate; build_dataset ;;
  *) usage; exit 2 ;;
esac
