# SG2 ACT training recipes

The task recipes are the single place to choose the dataset, exact robot I/O,
camera layout, ACT architecture, optimizer, scheduler, state noise, and run
parameters.

```bash
python3 tools/training/act_recipe.py validate training/task_000525/act.yaml
python3 tools/training/act_recipe.py plan training/task_000525/act.yaml
python3 tools/training/act_recipe.py run training/task_000525/act.yaml
```

Dataset and output paths are container paths below `/workspace`; on the host
they resolve below `docker/workspace`. Use `--dataset-root` or `--output-dir`
for a one-run override without editing a recipe.

The supplied recipes describe the new 15 Hz camera contract. They deliberately
reject older all-landscape datasets: head is upright 672x376 and wrists are
upright portrait 480x640. Shape validation cannot prove visual uprightness, so
the recorder/converter still needs a visual orientation check.

`policy_io.cameras` is the camera catalog and should normally stay unchanged.
Select the cameras used by a new policy with the single `camera_inputs` line;
the checked-in default uses all three:

```yaml
# all cameras (default)
camera_inputs: [head, left_wrist, right_wrist]

# examples for separate experiments
camera_inputs: [head]
camera_inputs: [head, left_wrist]
camera_inputs: [left_wrist, right_wrist]
```

The selected order is written into the ACT `policy.input_features` contract and
only those cameras are written to `cyclo_policy.yaml`. A canonical dataset may
still contain any of the other cameras declared in the catalog; they are not
model inputs. At inference, the checkpoint manifest makes RobotClient subscribe
to and wait for only the selected sources. There is intentionally no separate
camera toggle in the inference UI, because that could disagree with the trained
model. Use a distinct `training.job_name` and `training.output_dir` for every
camera combination, and do not change `camera_inputs` when resuming a run.

LeRobot currently still decodes unselected camera videos from an all-camera
dataset during training. This costs I/O but does not expose those tensors to ACT.
If decode throughput becomes limiting, create a derived dataset containing the
same samples and only the selected video features rather than changing the
policy contract.

`policy_io.state_components` and `policy_io.action_components` are independent.
For example, a Task 525 policy can observe the full robot while commanding only
the right arm:

```yaml
policy_io:
  # Keep the existing components map, including mobile.
  state_components: [arm_left, arm_right, head, lift, mobile]
  action_components: [arm_right]
  inactive_actions:
    arm_left: hold_current
    head: hold_current
    lift: hold_current
    mobile: hold_current
```

For a stationary Task 525 policy that commands both arms, head, and lift but not
the base, use `action_components: [arm_left, arm_right, head, lift]` and
`inactive_actions: {mobile: hold_current}`. An omitted action component must
always appear in `inactive_actions` with `hold_current`; the validator checks
that the two sets match exactly. Omitted state components need no inactive entry.
Task 458 has no mobile component at all, because the fixed-base environment does
not provide it, so mobile is absent rather than marked inactive there. Component
order determines the exact state/action vector order written into the deployment
manifest; keep each component's feature names in canonical robot order.

Task 458 keeps state noise off by default. Task 525 reproduces the latest useful
run with normalized-space Gaussian standard deviation 0.01 on all 22 state
columns. `state_noise.std` applies one standard deviation to every selected
state feature; `std_by_feature` overrides named columns (use `0` to exclude one).
In `space: normalized`, values are post-normalization units and injection occurs
after the saved preprocessor. In `space: raw`, values use each feature's dataset
units and injection occurs before preprocessing. Evaluation, actions, images,
and files on disk remain clean.

The default `optimizer.mode: policy_preset` uses ACT's AdamW parameter grouping
and no scheduler; only fields under `optimizer.policy_preset` are active. Set
`optimizer.mode: custom` to activate `optimizer.custom` (including clipping,
betas, and epsilon) and choose a supported scheduler. This mirrors LeRobot's
constraint that an explicit optimizer also requires a scheduler.

`training.mixed_precision` sets Accelerate's supported `no`, `fp16`, or `bf16`
mode through `ACCELERATE_MIXED_PRECISION`; it is not an invented LeRobot CLI
field. `policy.inference_amp` is the separate AMP setting stored for inference.

During a run, the launcher writes a concise deployment `cyclo_policy.yaml` to
the run root and each complete `checkpoints/<step>/pretrained_model` directory.
The full training provenance is kept separately as `resolved_recipe.yaml` at
the run root. A checkpoint is finalized only after its model, policy config,
train config, and positive training-step state are all present.
