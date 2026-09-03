# Task 000525 contract smoke runs

These recipes are short end-to-end wiring checks, not policies intended to solve the task.
Both use two canonical Isaac-rendered episodes derived from `demo_0` in the existing Task 000525 HDF5.

| Recipe | Dataset I/O | Cameras | Steps |
| --- | --- | --- | ---: |
| `task_000525_full22_all_cameras_act.yaml` | full 22D state/action | head + both wrists | 1,000 |
| `task_000525_right_head_lift11_head_only_act.yaml` | right arm/gripper + head + lift, 11D state/action | head only | 250 |

The second dataset is a name-based projection of the first dataset's canonical replay staging.
It deliberately declares `arm_left` and `mobile` as `hold_current`, so an inference runtime never
has to guess what to do with components that the policy does not output.

Run a smoke recipe with the same interface as a full training recipe:

```bash
python3 tools/training/act_recipe.py validate training/smoke/task_000525_full22_all_cameras_act.yaml
python3 tools/training/act_recipe.py plan training/smoke/task_000525_full22_all_cameras_act.yaml
python3 tools/training/act_recipe.py run training/smoke/task_000525_full22_all_cameras_act.yaml
```

Generated datasets and checkpoints live below `docker/workspace/` and are intentionally not tracked
by Git. A successful run writes `resolved_recipe.yaml` and `cyclo_policy.yaml` at the run root, plus
`cyclo_policy.yaml` inside every complete `pretrained_model` checkpoint.

Experiment source:

- HDF5: `/home/robotis-ai/cyclo_lab/datasets/task_000525_trajectory_ccw_rootstable_success_50_v2.hdf5`
- HDF5 SHA-256: `fe5794dd726520ca78e6ec74efcee91bb846303896ca872cb3e77f5f7876dfe0`
- Replay: `demo_0`, two visual seeds, 2,028 frames total at 15 Hz
- Camera tensors: head `[3,376,672]`; wrists `[3,640,480]`
