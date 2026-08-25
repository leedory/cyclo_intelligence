#!/usr/bin/env bash
# Reproducible Task_000459 generated-simulation ACT smoke workflow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$REPO_ROOT/docker/workspace"
NATIVE_SOURCE_ROOT="${NATIVE_SOURCE_ROOT:-/home/robotis-ai/cyclo_lab/data/generated/Task_000459_Pick_Peanut_Mix_WhiteShelf_GENERATED_SIM}"
STAGED_SOURCE_ROOT="$WORKSPACE/native_sources/Task_000459_Pick_Peanut_Mix_WhiteShelf_GENERATED_SIM"
DATASET_ROOT="$WORKSPACE/dataset/robotis/task_000459_peanut_mix_generated_smoke_act_v30"
DATASET_REPO_ID="robotis/task_000459_peanut_mix_generated_smoke_act_v30"
CONVERSION_IMAGE_HEIGHT="${CONVERSION_IMAGE_HEIGHT:-480}"
CONVERSION_IMAGE_WIDTH="${CONVERSION_IMAGE_WIDTH:-640}"
# Identical ACT defaults to run.sh, run_sim_only.sh, and run_real_only.sh.
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
ACT_JOB_NAME="${ACT_JOB_NAME:-task_000459_peanut_mix_generated_smoke_act}"
ACT_OUTPUT_DIR="${ACT_OUTPUT_DIR:-/workspace/model/lerobot/${ACT_JOB_NAME}}"
ACT_LOG_FILE="${ACT_LOG_FILE:-$WORKSPACE/model/lerobot/${ACT_JOB_NAME}/train.log}"
LEROBOT_CONTAINER="${LEROBOT_CONTAINER_NAME:-lerobot_server}"
CONTAINER_WORKSPACE="/workspace"
CONTAINER_SOURCE_ROOT="$CONTAINER_WORKSPACE/native_sources/Task_000459_Pick_Peanut_Mix_WhiteShelf_GENERATED_SIM"
CONTAINER_DATASET_ROOT="$CONTAINER_WORKSPACE/dataset/robotis/task_000459_peanut_mix_generated_smoke_act_v30"
CONTAINER_TOOL="$CONTAINER_WORKSPACE/act_sim_real/convert_native_peanut_smoke.py"

usage() {
  cat <<'EOF'
Usage: tools/act_sim_real/run_generated_smoke.sh <stage-source|validate|build-dataset|train|all>

stage-source  Snapshot the verified Task_000459 archive into docker/workspace.
validate      Verify all source episodes are successful and conversion-ready.
build-dataset Convert the staged archive to a 10 Hz, 480x640 LeRobot v3 dataset.
train         Run fresh ACT training in lerobot_server (start it first with docker/container.sh start-lerobot).
all           stage-source, validate, then build-dataset. Does not start GPU training.

This is a smoke/overfit dataset. The 12 demonstrations are fixed successful
simulation replays and are not evidence of deployment readiness.
EOF
}

require_container() {
  if ! docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null | grep -qx true; then
    echo "Container is not running: $1" >&2
    exit 1
  fi
}

stage_source() {
  [[ -f "$NATIVE_SOURCE_ROOT/manifest.json" ]] || { echo "Native source is missing: $NATIVE_SOURCE_ROOT" >&2; exit 1; }
  mkdir -p "$STAGED_SOURCE_ROOT" "$WORKSPACE/act_sim_real"
  rsync -a --partial --info=progress2 "$NATIVE_SOURCE_ROOT/" "$STAGED_SOURCE_ROOT/"
  install -m 0755 "$SCRIPT_DIR/convert_native_peanut_smoke.py" "$WORKSPACE/act_sim_real/convert_native_peanut_smoke.py"
  echo "Staged native source: $STAGED_SOURCE_ROOT"
}

validate() {
  require_container "$LEROBOT_CONTAINER"
  [[ -f "$WORKSPACE/act_sim_real/convert_native_peanut_smoke.py" ]] || { echo "Run stage-source first." >&2; exit 1; }
  docker exec "$LEROBOT_CONTAINER" python3 "$CONTAINER_TOOL" --source "$CONTAINER_SOURCE_ROOT" --output "$CONTAINER_DATASET_ROOT" --repo-id "$DATASET_REPO_ID" --fps 10 --image-height "$CONVERSION_IMAGE_HEIGHT" --image-width "$CONVERSION_IMAGE_WIDTH" --validate-only
}

build_dataset() {
  require_container "$LEROBOT_CONTAINER"
  [[ -f "$WORKSPACE/act_sim_real/convert_native_peanut_smoke.py" ]] || { echo "Run stage-source first." >&2; exit 1; }
  if [[ -d "$DATASET_ROOT" ]] && [[ -n "$(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite existing dataset: $DATASET_ROOT" >&2
    exit 1
  fi
  docker exec "$LEROBOT_CONTAINER" python3 "$CONTAINER_TOOL" --source "$CONTAINER_SOURCE_ROOT" --output "$CONTAINER_DATASET_ROOT" --repo-id "$DATASET_REPO_ID" --fps 10 --image-height "$CONVERSION_IMAGE_HEIGHT" --image-width "$CONVERSION_IMAGE_WIDTH"
}

train() {
  require_container "$LEROBOT_CONTAINER"
  [[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "Dataset is missing. Run build-dataset first: $DATASET_ROOT" >&2; exit 1; }
  mkdir -p "$(dirname "$ACT_LOG_FILE")"
  docker exec "$LEROBOT_CONTAINER" lerobot-train --policy.type=act --policy.chunk_size="$ACT_CHUNK_SIZE" --policy.n_action_steps="$ACT_N_ACTION_STEPS" --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$CONTAINER_DATASET_ROOT" --dataset.eval_split="$ACT_EVAL_SPLIT" --dataset.image_transforms.enable="$ACT_IMAGE_TRANSFORMS_ENABLE" --dataset.image_transforms.max_num_transforms="$ACT_IMAGE_TRANSFORMS_MAX_NUM" --dataset.image_transforms.tfs="$ACT_IMAGE_TRANSFORMS_TFS" --policy.push_to_hub="$ACT_PUSH_TO_HUB" --policy.device="$ACT_DEVICE" --batch_size="$ACT_BATCH_SIZE" --num_workers="$ACT_NUM_WORKERS" --seed="$ACT_SEED" --steps="$ACT_STEPS" --save_checkpoint="$ACT_SAVE_CHECKPOINT" --save_freq="$ACT_SAVE_FREQ" --log_freq="$ACT_LOG_FREQ" --env_eval_freq="$ACT_ENV_EVAL_FREQ" --eval_steps="$ACT_EVAL_STEPS" --wandb.enable="$ACT_WANDB_ENABLE" --job_name="$ACT_JOB_NAME" --output_dir="$ACT_OUTPUT_DIR" 2>&1 | tee "$ACT_LOG_FILE"
}

case "${1:-}" in
  stage-source) stage_source ;;
  validate) validate ;;
  build-dataset) build_dataset ;;
  train) train ;;
  all) stage_source; validate; build_dataset ;;
  *) usage; exit 2 ;;
esac
