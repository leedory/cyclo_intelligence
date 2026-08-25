#!/usr/bin/env python3
"""Convert verified Cyclo Lab Task_000459 native episodes to LeRobot v3."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image


TASK = "Pick up the Peanut Mix with right gripper."
# Source-camera name -> common ACT camera name, clockwise rotation in degrees.
SOURCE_CAMERAS = {
    "cam_head": ("cam_left_head", 0),
    "cam_wrist_left": ("cam_left_wrist", 270),
    "cam_wrist_right": ("cam_right_wrist", 270),
}
# Same action groups and ordering as ffw_sg2_rev1_act_common_cameras.yaml.
ACTION_NAMES = (
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
    "lift_joint", "linear_x", "linear_y", "angular_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def find_episodes(source: Path) -> list[Path]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cyclo_lab_native_simulation_dataset/v1":
        raise ValueError(f"Unexpected dataset format: {manifest.get('format')!r}")
    if manifest.get("task_id") != "000459":
        raise ValueError(f"Expected Task_000459, found {manifest.get('task_id')!r}")
    episodes = sorted((source / "episodes").glob("episode_*"))
    if len(episodes) != int(manifest.get("episode_count", 0)) or not episodes:
        raise RuntimeError(f"Source episode count mismatch below {source / 'episodes'}")
    return episodes


def load_episode(episode: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    metrics_path, trajectory_path = episode / "metrics.json", episode / "trajectory.npz"
    if not metrics_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(f"{episode}: missing metrics.json or trajectory.npz")
    if json.loads(metrics_path.read_text(encoding="utf-8")).get("success") is not True:
        raise RuntimeError(f"{episode}: refusing unsuccessful source episode")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        state = np.asarray(archive["observation_state"], dtype=np.float32)
        action = np.asarray(archive["action"], dtype=np.float32)
        timestamps = np.asarray(archive["timestamps_s"], dtype=np.float64)
        state_names = [str(value) for value in archive["observation_state_names"].tolist()]
        action_names = [str(value) for value in archive["action_names"].tolist()]
    if state.ndim != 2 or action.ndim != 2 or len(state) != len(action) or len(action) != len(timestamps):
        raise ValueError(f"{episode}: inconsistent trajectory arrays")
    if state.shape[1] != len(state_names) or action.shape[1] != len(action_names):
        raise ValueError(f"{episode}: trajectory dimensions do not match names")
    missing = set(ACTION_NAMES) - set(action_names)
    if missing:
        raise ValueError(f"{episode}: action lacks {sorted(missing)}")
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"{episode}: timestamps must be strictly increasing")
    return state, action, timestamps, state_names, action_names


def resample_indices(timestamps: np.ndarray, fps: int) -> np.ndarray:
    duration_s = float(timestamps[-1] - timestamps[0])
    target_times = timestamps[0] + np.arange(int(round(duration_s * fps)) + 1) / fps
    indices = np.abs(timestamps[:, None] - target_times[None, :]).argmin(axis=0)
    if np.any(np.diff(indices) <= 0):
        raise RuntimeError("Target FPS is too high for strictly ordered source samples")
    return indices


def source_frame(episode: Path, camera: str, index: int) -> Path:
    path = episode / "frames" / camera / f"frame_{index:06d}.jpg"
    if not path.is_file():
        raise FileNotFoundError(f"{episode}: missing {camera} frame {index}")
    return path


def load_rgb(path: Path, rotation_deg: int, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if rotation_deg:
            image = image.rotate(rotation_deg, expand=True)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def validate_source(args: argparse.Namespace, episodes: list[Path]):
    loaded = []
    reference_state_names: list[str] | None = None
    for episode in episodes:
        state, action, timestamps, state_names, action_names = load_episode(episode)
        if reference_state_names is None:
            reference_state_names = state_names
        elif state_names != reference_state_names:
            raise ValueError(f"{episode}: state name ordering differs from first episode")
        indices = resample_indices(timestamps, args.fps)
        for camera in SOURCE_CAMERAS:
            for index in indices:
                source_frame(episode, camera, int(index))
        logging.info("Validated %s: %d source frames -> %d policy frames", episode.name, len(action), len(indices))
        loaded.append((episode, state, action, state_names, action_names, indices))
    return loaded


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if min(args.fps, args.image_height, args.image_width) <= 0:
        raise ValueError("fps and image dimensions must be positive")
    if not args.validate_only and args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output}")
    loaded = validate_source(args, find_episodes(args.source))
    if args.validate_only:
        logging.info("Validated %d successful native episodes; no output written", len(loaded))
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    state_names = loaded[0][3]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": list(ACTION_NAMES)},
        **{
            f"observation.images.{target}": {
                "dtype": "video", "shape": (3, args.image_height, args.image_width),
                "names": ["channels", "height", "width"],
            }
            for target, _rotation in SOURCE_CAMERAS.values()
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, root=args.output, robot_type="ffw_sg2_rev1",
        features=features, use_videos=True, batch_encoding_size=1,
    )
    try:
        for episode, state, action, _state_names, action_names, indices in loaded:
            action_columns = [action_names.index(name) for name in ACTION_NAMES]
            for index in indices:
                index = int(index)
                frame = {
                    "observation.state": state[index].astype(np.float32, copy=False),
                    "action": action[index, action_columns].astype(np.float32, copy=False),
                    "task": TASK,
                }
                for source, (target, rotation_deg) in SOURCE_CAMERAS.items():
                    frame[f"observation.images.{target}"] = load_rgb(
                        source_frame(episode, source, index), rotation_deg, args.image_height, args.image_width
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
            logging.info("Converted %s", episode.name)
    finally:
        dataset.finalize()

    manifest = {
        "task": TASK,
        "source_kind": "cyclo_lab_native_simulation_dataset/v1 (not MCAP)",
        "source_root": str(args.source),
        "fps": args.fps,
        "image_size_hw": [args.image_height, args.image_width],
        "policy_schema": {
            "observation_state_dim": len(state_names), "action_dim": len(ACTION_NAMES),
            "action_names": list(ACTION_NAMES),
            "camera_mapping": {source: target for source, (target, _rotation) in SOURCE_CAMERAS.items()},
        },
        "episodes": [
            {"dataset_episode_index": item, "source": str(episode), "source_frames": len(action), "policy_frames": len(indices)}
            for item, (episode, _state, action, _state_names, _action_names, indices) in enumerate(loaded)
        ],
    }
    manifest_path = args.output / "meta" / "cyclo_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
