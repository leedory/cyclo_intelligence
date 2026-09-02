# Archived SG2 ACT experiment launchers

These files preserve the exact commands used by earlier experiments. They are
reference material, not the supported training interface. New SG2 ACT runs use
`training/task_000458/act.yaml`, `training/task_000525/act.yaml`, and
`tools/training/act_recipe.py`.

## Task 000458 10 Hz experiments

`task_000458_10hz/` contains the old sim-only, real-only, and combined MCAP
launchers. They explicitly converted state, action, and video onto a 10 Hz
LeRobot grid. Current SG2 recording, training, and inference use 15 Hz, so do
not use these launchers for a new dataset.

The converters and camera configuration they called remain under
`tools/act_sim_real/` until the data-conversion workflow has a supported
replacement. Existing datasets and models have not been moved.

## Task 000525 22D all-landscape experiment

`task_000525_22d_all_landscape/` contains exact copies restored from backup
commit `581e894`. That run trained the full 22D mobile state/action layout and
added Gaussian noise with standard deviation 0.01 after state normalization.
It expected all three policy images to have shape `[3, 480, 640]`, including
the wrist cameras.

That historical image layout is superseded. New recipes require an upright
672x376 landscape head image and upright 480x640 portrait wrist images, all at
15 Hz. The archived noise wrapper also relied on a process-local monkey patch
and did not write its noise setting into the standard checkpoint configuration;
the recipe launcher replaces it with declared raw/normalized noise spaces,
scalar or named standard deviations, and checkpoint-local `cyclo_policy.yaml`
files.
