#!/usr/bin/env bash
# Train the three Task_000458 ACT variants serially. Stop on first failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_sim_only.sh" train &&
"$SCRIPT_DIR/run.sh" train &&
"$SCRIPT_DIR/run_real_only.sh" train
