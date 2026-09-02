#!/usr/bin/env bash
# Train Task_000525 ACT with the Task_000458 recipe plus 22D state-only noise.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$REPO_ROOT/docker/workspace"

SOURCE_DATASET_ROOT="${TASK000525_DATASET_ROOT:-/home/robotis-ai/cyclo_lab/datasets/lerobot/task_000525_all_subtasks_original_plus_visual_aug_450_v30}"
DATASET_NAME="task_000525_all_subtasks_original_plus_visual_aug_450_v30"
DATASET_REPO_ID="robotis/${DATASET_NAME}"
STAGED_DATASET_ROOT="$WORKSPACE/dataset/$DATASET_NAME"
CONTAINER_DATASET_ROOT="/workspace/dataset/$DATASET_NAME"

# Task_000458 ACT defaults. The only added augmentation is Gaussian state noise.
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

# Noise is applied to normalized observation.state values immediately before the ACT update.
# It never changes action, images, evaluation data, or the source dataset on disk.
ACT_STATE_DIM="${ACT_STATE_DIM:-22}"
ACT_STATE_NOISE_STD="${ACT_STATE_NOISE_STD:-0.01}"
NOISE_TAG="${ACT_STATE_NOISE_STD//./p}"
ACT_JOB_NAME="${ACT_JOB_NAME:-task_000525_all_subtasks_original_plus_visual_aug_450_v30_act_state_noise_std${NOISE_TAG}}"
ACT_OUTPUT_DIR="${ACT_OUTPUT_DIR:-/workspace/model/lerobot/${ACT_JOB_NAME}}"
ACT_LOG_FILE="${ACT_LOG_FILE:-$WORKSPACE/logs/lerobot/${ACT_JOB_NAME}.log}"

LEROBOT_CONTAINER="${LEROBOT_CONTAINER_NAME:-lerobot_server}"
HOST_TRAINER="$SCRIPT_DIR/train_with_state_noise.py"
STAGED_TRAINER="$WORKSPACE/act_sim_real/train_with_state_noise.py"
CONTAINER_TRAINER="/workspace/act_sim_real/train_with_state_noise.py"

usage() {
  cat <<'EOF'
Usage: tools/act_sim_real/run_task000525_mobile_22d_state_noise.sh <validate|prepare|train>

validate  Verify the supplied LeRobot v3 dataset is a 22D mobile-policy dataset.
prepare   Hard-link the dataset into docker/workspace and stage the noise trainer.
train     Prepare as needed, then run fresh ACT training in lerobot_server.

Start lerobot_server first with docker/container.sh start-lerobot.
Override the normalized Gaussian noise with ACT_STATE_NOISE_STD (default: 0.01).
EOF
}

require_container() {
  if ! docker inspect --format '{{.State.Running}}' "$LEROBOT_CONTAINER" 2>/dev/null | grep -qx true; then
    echo "Container is not running: $LEROBOT_CONTAINER" >&2
    echo "Start it with: $REPO_ROOT/docker/container.sh start-lerobot" >&2
    exit 1
  fi
}

validate_dataset() {
  python3 - "$SOURCE_DATASET_ROOT" "$ACT_STATE_DIM" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_dim = int(sys.argv[2])
info_path = root / "meta" / "info.json"
if not info_path.is_file():
    raise SystemExit(f"Missing LeRobot metadata: {info_path}")

info = json.loads(info_path.read_text(encoding="utf-8"))
if info.get("codebase_version") != "v3.0":
    raise SystemExit(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")

features = info.get("features", {})
for key in ("observation.state", "action"):
    feature = features.get(key, {})
    if feature.get("shape") != [expected_dim]:
        raise SystemExit(f"Expected {key} shape [{expected_dim}], got {feature.get('shape')!r}")
    names = feature.get("names")
    if not isinstance(names, list) or len(names) != expected_dim:
        raise SystemExit(f"Expected {expected_dim} names for {key}, got {names!r}")

required_mobile_names = {"linear_x", "linear_y", "angular_z"}
state_names = set(features["observation.state"]["names"])
action_names = set(features["action"]["names"])
if not required_mobile_names <= state_names or not required_mobile_names <= action_names:
    raise SystemExit("22D policy schema is missing linear_x, linear_y, or angular_z")

required_cameras = {
    "observation.images.rgb.cam_left_head",
    "observation.images.rgb.cam_left_wrist",
    "observation.images.rgb.cam_right_wrist",
}
missing_cameras = sorted(required_cameras - features.keys())
if missing_cameras:
    raise SystemExit(f"Dataset is missing policy cameras: {missing_cameras}")

if not (root / "data").is_dir() or not (root / "videos").is_dir():
    raise SystemExit(f"Dataset data/videos directories are incomplete: {root}")

print(
    f"Dataset OK: episodes={info.get('total_episodes')} frames={info.get('total_frames')} "
    f"fps={info.get('fps')} state={expected_dim}D action={expected_dim}D"
)
PY
}

stage_dataset() {
  validate_dataset
  mkdir -p "$(dirname "$STAGED_DATASET_ROOT")" "$(dirname "$STAGED_TRAINER")"

  if [[ -e "$STAGED_DATASET_ROOT" ]]; then
    if [[ ! -f "$STAGED_DATASET_ROOT/meta/info.json" ]] || ! cmp -s \
      "$SOURCE_DATASET_ROOT/meta/info.json" "$STAGED_DATASET_ROOT/meta/info.json"; then
      echo "Refusing incompatible existing staged dataset: $STAGED_DATASET_ROOT" >&2
      exit 1
    fi
    echo "Using existing staged dataset: $STAGED_DATASET_ROOT"
  else
    if [[ "$(stat -c '%d' "$SOURCE_DATASET_ROOT")" != "$(stat -c '%d' "$WORKSPACE")" ]]; then
      echo "Source and workspace are on different filesystems; hard-link staging is unavailable." >&2
      exit 1
    fi
    cp -al -- "$SOURCE_DATASET_ROOT" "$STAGED_DATASET_ROOT"
    echo "Hard-linked dataset for container access: $STAGED_DATASET_ROOT"
  fi

  install -m 0755 "$HOST_TRAINER" "$STAGED_TRAINER"
  echo "Staged state-noise trainer: $STAGED_TRAINER"
}

train() {
  stage_dataset
  require_container
  mkdir -p "$(dirname "$ACT_LOG_FILE")"

  echo "Training Task_000525 ACT: dataset=$DATASET_REPO_ID state=${ACT_STATE_DIM}D normalized_state_noise_std=$ACT_STATE_NOISE_STD"
  docker exec \
    -e ACT_STATE_DIM="$ACT_STATE_DIM" \
    -e ACT_STATE_NOISE_STD="$ACT_STATE_NOISE_STD" \
    "$LEROBOT_CONTAINER" \
    python3 "$CONTAINER_TRAINER" \
    --policy.type=act \
    --policy.chunk_size="$ACT_CHUNK_SIZE" \
    --policy.n_action_steps="$ACT_N_ACTION_STEPS" \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$CONTAINER_DATASET_ROOT" \
    --dataset.eval_split="$ACT_EVAL_SPLIT" \
    --dataset.image_transforms.enable="$ACT_IMAGE_TRANSFORMS_ENABLE" \
    --dataset.image_transforms.max_num_transforms="$ACT_IMAGE_TRANSFORMS_MAX_NUM" \
    --dataset.image_transforms.tfs="$ACT_IMAGE_TRANSFORMS_TFS" \
    --policy.push_to_hub="$ACT_PUSH_TO_HUB" \
    --policy.device="$ACT_DEVICE" \
    --batch_size="$ACT_BATCH_SIZE" \
    --num_workers="$ACT_NUM_WORKERS" \
    --seed="$ACT_SEED" \
    --steps="$ACT_STEPS" \
    --save_checkpoint="$ACT_SAVE_CHECKPOINT" \
    --save_freq="$ACT_SAVE_FREQ" \
    --log_freq="$ACT_LOG_FREQ" \
    --env_eval_freq="$ACT_ENV_EVAL_FREQ" \
    --eval_steps="$ACT_EVAL_STEPS" \
    --wandb.enable="$ACT_WANDB_ENABLE" \
    --job_name="$ACT_JOB_NAME" \
    --output_dir="$ACT_OUTPUT_DIR" \
    2>&1 | tee "$ACT_LOG_FILE"
}

case "${1:-}" in
  validate) validate_dataset ;;
  prepare) stage_dataset ;;
  train) train ;;
  *) usage; exit 2 ;;
esac
