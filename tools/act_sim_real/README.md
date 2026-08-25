# Task_000458 sim-to-real ACT

This workflow converts the Task_000458 simulated and physical MCAP recordings into one local LeRobot v3 dataset, then trains one ACT policy from the shared features.

The source domains have the same SG2 state/action topics, but only these cameras are common to both: `cam_left_head`, `cam_left_wrist`, and `cam_right_wrist`. `cam_external` is deliberately excluded. Recorder-side rotation is already baked into the MP4s. Conversion keeps the head landscape at `(H=480, W=640)` and both wrist cameras portrait at `(H=640, W=480)`; it does not stretch portrait images into landscape.
The real recordings command only the right arm, lift, and base, so the common ACT state and action each have 12 dimensions.

Run from the repository root:

```bash
tools/act_sim_real/run.sh all
```

This downloads the 1050 real recordings to `docker/workspace/rosbag2/Task_000458_Pick_Peanut_Mix_WhiteShelf_SimtoReal_REAL_MCAP`, validates the episodes, and writes the combined dataset to:

```text
docker/workspace/dataset/robotis/task_000458_peanut_mix_sim_real_act_native_aspect_v30
```

The output contains `meta/cyclo_source_manifest.json`, which records each converted episode's sim/real source. It also computes one set of normalization statistics across both domains.

To train, start the policy container once and then run:

```bash
docker/container.sh start-lerobot
tools/act_sim_real/run.sh train
```

The ACT recipe uses 10 Hz, three cameras with equal pixel counts but per-camera orientation, and the parameters in `run.sh`. With only about 12 sim and 12 real episodes, use the result as a baseline; adding successful real demonstrations or fine-tuning on real-only data is normally needed for reliable on-robot performance.

The build script refuses to overwrite an existing dataset output. Rename or archive a previous output before rebuilding it.
