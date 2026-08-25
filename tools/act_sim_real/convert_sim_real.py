#!/usr/bin/env python3
"""Create one LeRobot v3 ACT dataset from the Task_000458 sim and real MCAPs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Direct execution in the Cyclo main container needs the editable package root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyclo_data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))


CAMERAS = ("cam_left_head", "cam_left_wrist", "cam_right_wrist")
DEFAULT_IMAGE_RESIZES = {
    "cam_left_head": (480, 640),
    "cam_left_wrist": (640, 480),
    "cam_right_wrist": (640, 480),
}


def camera_size(value: str) -> tuple[str, tuple[int, int]]:
    """Parse ``CAMERA=HEIGHTxWIDTH`` for argparse."""
    try:
        camera, size = value.split("=", 1)
        height_raw, width_raw = size.lower().split("x", 1)
        height, width = int(height_raw), int(width_raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected CAMERA=HEIGHTxWIDTH, e.g. cam_left_wrist=640x480"
        ) from exc
    if camera not in CAMERAS:
        raise argparse.ArgumentTypeError(
            f"unknown camera {camera!r}; expected one of {', '.join(CAMERAS)}"
        )
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("image dimensions must be positive")
    return camera, (height, width)


def episode_sort_key(path: Path) -> tuple[int, str]:
    try:
        info = json.loads((path / "episode_info.json").read_text(encoding="utf-8"))
        return (int(info.get("episode_index", path.name)), str(path))
    except (OSError, ValueError, json.JSONDecodeError):
        try:
            return (int(path.name), str(path))
        except ValueError:
            return (sys.maxsize, str(path))


def video_exists(episode: Path, camera: str) -> bool:
    return any((episode / "videos").glob(f"*/*{camera}.mp4"))


def find_episodes(root: Path, domain: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"{domain} root does not exist: {root}")

    episodes = sorted(
        {
            metadata.parent
            for metadata in root.rglob("metadata.yaml")
            if any(metadata.parent.glob("*.mcap")) and "segments" not in metadata.parent.relative_to(root).parts
        },
        key=episode_sort_key,
    )
    if not episodes:
        raise RuntimeError(f"No MCAP episodes found below {root}")

    missing = [
        f"{episode}: {camera}"
        for episode in episodes
        for camera in CAMERAS
        if not video_exists(episode, camera)
    ]
    if missing:
        raise RuntimeError(
            f"{domain} has episodes missing required common cameras:\n  "
            + "\n  ".join(missing)
        )
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--robot-config", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--camera-size",
        action="append",
        type=camera_size,
        default=[],
        metavar="CAMERA=HEIGHTxWIDTH",
        help=(
            "Per-camera output size. May be repeated. Defaults preserve the "
            "640x480 landscape head and 480x640 portrait wrists."
        ),
    )
    parser.add_argument(
        "--image-height",
        type=int,
        help="Legacy global output height; must be paired with --image-width.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        help="Legacy global output width; must be paired with --image-height.",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Convert only real episodes; --sim-root is ignored.",
    )
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="Convert only sim episodes; --real-root is ignored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_only and args.sim_only:
        raise ValueError("--real-only and --sim-only cannot be used together")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Output directory already exists and is not empty: {args.output}")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if (args.image_height is None) != (args.image_width is None):
        raise ValueError("--image-height and --image-width must be used together")
    if args.camera_size and args.image_height is not None:
        raise ValueError(
            "--camera-size cannot be combined with the legacy global image size"
        )
    if (
        args.image_height is not None
        and min(args.image_height, args.image_width) <= 0
    ):
        raise ValueError("image dimensions must be positive")

    global_resize = None
    per_camera_resizes = dict(DEFAULT_IMAGE_RESIZES)
    if args.image_height is not None:
        global_resize = (args.image_height, args.image_width)
        per_camera_resizes = {}
    else:
        per_camera_resizes.update(dict(args.camera_size))

    sim = [] if args.real_only else find_episodes(args.sim_root, "sim")
    real = [] if args.sim_only else find_episodes(args.real_root, "real")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("sim_real_act")
    logger.info("Episodes: sim=%d, real=%d; common cameras=%s", len(sim), len(real), CAMERAS)

    # The repository's v3 writer keeps one combined set of normalization
    # statistics, which is essential for a single ACT policy.
    from cyclo_data.converter.to_lerobot_v30 import (
        RosbagToLerobotV30Converter,
        V30ConversionConfig,
    )

    config = V30ConversionConfig(
        repo_id=args.repo_id,
        output_dir=args.output,
        fps=args.fps,
        robot_type="ffw_sg2_rev1",
        robot_config_path=args.robot_config,
        image_resize=global_resize,
        image_resize_by_camera=per_camera_resizes,
        selected_cameras=list(CAMERAS),
    )
    success = RosbagToLerobotV30Converter(
        config, logger
    ).convert_multiple_rosbags(sim + real)
    if not success:
        raise RuntimeError("LeRobot conversion failed")

    source_manifest = {
        "task": "Pick up the Peanut Mix with right gripper.",
        "fps": args.fps,
        "image_resize": list(global_resize) if global_resize else None,
        "image_resize_by_camera": {
            camera: list(size) for camera, size in per_camera_resizes.items()
        },
        "common_cameras": list(CAMERAS),
        "episodes": [
            {"dataset_episode_index": index, "domain": domain, "source": str(path)}
            for index, (domain, path) in enumerate(
                [("sim", path) for path in sim] + [("real", path) for path in real]
            )
        ],
    }
    manifest_path = args.output / "meta" / "cyclo_source_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
