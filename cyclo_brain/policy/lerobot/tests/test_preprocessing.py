#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest

import numpy as np
import torch


ENGINE_DIR = Path(
    os.environ.get(
        "LEROBOT_ENGINE_DIR",
        Path(__file__).resolve().parents[1] / "lerobot_engine",
    )
)
package = types.ModuleType("lerobot_engine")
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault("lerobot_engine", package)

for module_name in ("constants", "image_preprocessing", "policy_contract", "io_mapping"):
    spec = importlib.util.spec_from_file_location(
        f"lerobot_engine.{module_name}", ENGINE_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if module_name == "io_mapping":
        robot_client_stub = types.ModuleType("robot_client")
        robot_client_stub.RobotClient = object
        sys.modules.setdefault("robot_client", robot_client_stub)
    spec.loader.exec_module(module)

spec = importlib.util.spec_from_file_location(
    "lerobot_engine.preprocessing", ENGINE_DIR / "preprocessing.py"
)
preprocessing = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = preprocessing
spec.loader.exec_module(preprocessing)

CameraContract = sys.modules["lerobot_engine.policy_contract"].CameraContract
StateSource = sys.modules["lerobot_engine.io_mapping"].StateSource
STATE_KEY = sys.modules["lerobot_engine.constants"].STATE_KEY


class FakeRobot:
    def __init__(self, positions, image=None, odom=None):
        self._positions = positions
        self._image = image
        self._odom = odom
        self._config = {"cameras": {"cam": {"rotation_deg": 0}}}

    def get_images(self, format="rgb"):
        del format
        if self._image is None:
            return {"unused": np.zeros((2, 2, 3), dtype=np.uint8)}
        return {"cam": self._image}

    def get_joint_positions(self):
        return self._positions

    def get_odom(self):
        return self._odom


class Preprocessor(preprocessing.PreprocessingMixin):
    def __init__(self, robot, sources, cameras=()):
        self._robot = robot
        self._state_sources = sources
        self._policy_contract = SimpleNamespace(
            state_features=tuple(source.feature for source in sources)
        )
        self._cameras = {camera.source: camera.key for camera in cameras}
        self._camera_contracts = {camera.source: camera for camera in cameras}
        self._device = torch.device("cpu")

    def _fail(self, message):
        return {"success": False, "message": message}


class PreprocessingTest(unittest.TestCase):
    def test_builds_state_in_exact_contract_order_including_odom(self):
        sources = [
            StateSource("joint_b", "joint", "follower_arm", 1),
            StateSource("joint_a", "joint", "follower_arm", 0),
            StateSource("linear_y", "odom", None, 1),
        ]
        robot = FakeRobot(
            {"follower_arm": np.array([1.0, 2.0], dtype=np.float32)},
            odom={"linear_velocity": [3.0, 4.0, 0.0], "angular_velocity": [0.0, 0.0, 5.0]},
        )

        batch = Preprocessor(robot, sources)._build_observation("task")

        torch.testing.assert_close(batch[STATE_KEY], torch.tensor([[2.0, 1.0, 4.0]]))

    def test_missing_named_state_source_fails_instead_of_padding(self):
        sources = [StateSource("joint_b", "joint", "follower_arm", 1)]
        robot = FakeRobot({"follower_arm": np.array([1.0], dtype=np.float32)})

        result = Preprocessor(robot, sources)._build_observation("task")

        self.assertFalse(result["success"])
        self.assertIn("Missing exact joint source", result["message"])

    def test_camera_is_delivered_at_contract_shape_and_key(self):
        camera = CameraContract("cam", "observation.images.rgb.cam", 6, 4)
        robot = FakeRobot(
            {"follower_arm": np.array([1.0], dtype=np.float32)},
            image=np.zeros((4, 6, 3), dtype=np.uint8),
        )
        sources = [StateSource("joint_a", "joint", "follower_arm", 0)]

        batch = Preprocessor(robot, sources, (camera,))._build_observation("task")

        self.assertEqual(tuple(batch[camera.key].shape), (1, 3, 4, 6))

    def test_wrong_camera_resolution_fails_instead_of_resizing(self):
        camera = CameraContract("cam", "observation.images.rgb.cam", 6, 4)
        robot = FakeRobot(
            {"follower_arm": np.array([1.0], dtype=np.float32)},
            image=np.zeros((2, 3, 3), dtype=np.uint8),
        )
        sources = [StateSource("joint_a", "joint", "follower_arm", 0)]

        result = Preprocessor(robot, sources, (camera,))._build_observation("task")

        self.assertFalse(result["success"])
        self.assertIn("produced shape", result["message"])


if __name__ == "__main__":
    unittest.main()
