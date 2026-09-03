#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest

import numpy as np
import torch


ENGINE_DIR = Path(__file__).resolve().parents[1] / "lerobot_engine"
package = types.ModuleType("lerobot_engine")
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault("lerobot_engine", package)

constants_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.constants",
    ENGINE_DIR / "constants.py",
)
constants = importlib.util.module_from_spec(constants_spec)
sys.modules[constants_spec.name] = constants
constants_spec.loader.exec_module(constants)

image_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.image_preprocessing",
    ENGINE_DIR / "image_preprocessing.py",
)
image_preprocessing = importlib.util.module_from_spec(image_spec)
sys.modules[image_spec.name] = image_preprocessing
image_spec.loader.exec_module(image_preprocessing)

contract_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.policy_contract",
    ENGINE_DIR / "policy_contract.py",
)
policy_contract = importlib.util.module_from_spec(contract_spec)
sys.modules[contract_spec.name] = policy_contract
contract_spec.loader.exec_module(policy_contract)

preprocessing_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.preprocessing",
    ENGINE_DIR / "preprocessing.py",
)
preprocessing = importlib.util.module_from_spec(preprocessing_spec)
sys.modules[preprocessing_spec.name] = preprocessing
preprocessing_spec.loader.exec_module(preprocessing)

PreprocessingMixin = preprocessing.PreprocessingMixin
STATE_KEY = constants.STATE_KEY
CameraContract = policy_contract.CameraContract


class FakeRobot:
    _config = {"cameras": {}}

    def __init__(self, positions):
        self._positions = positions

    def get_images(self, format="rgb"):
        return {"unused": np.zeros((2, 2, 3), dtype=np.uint8)}

    def get_joint_positions(self):
        return {"follower_arm": self._positions}


class Preprocessor(PreprocessingMixin):
    def __init__(self, positions, expected):
        self._robot = FakeRobot(positions)
        self._cameras = {}
        self._state_modalities = ["arm"]
        self._image_resize = {}
        self._device = torch.device("cpu")
        feature = SimpleNamespace(shape=(expected,))
        config = SimpleNamespace(input_features={STATE_KEY: feature})
        self._policy = SimpleNamespace(config=config)

    def _fail(self, message):
        return {"error": message}


class PreprocessingTest(unittest.TestCase):
    def test_pads_short_state_to_policy_shape(self):
        preprocessor = Preprocessor([1.0, 2.0], expected=4)

        batch = preprocessor._build_observation("task")

        np.testing.assert_allclose(
            batch[STATE_KEY].numpy(),
            np.asarray([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32),
        )

    def test_truncates_long_state_to_policy_shape(self):
        preprocessor = Preprocessor([1.0, 2.0, 3.0, 4.0], expected=2)

        batch = preprocessor._build_observation("task")

        np.testing.assert_allclose(
            batch[STATE_KEY].numpy(),
            np.asarray([[1.0, 2.0]], dtype=np.float32),
        )


class ContractRobot:
    def __init__(self, positions, image=None, odom=None, rotation_deg=0):
        self._positions = positions
        self._image = image
        self._odom = odom
        self._config = {"cameras": {"cam": {"rotation_deg": rotation_deg}}}

    def get_images(self, format="rgb"):
        del format
        if self._image is None:
            return {"unused": np.zeros((2, 2, 3), dtype=np.uint8)}
        return {"cam": self._image}

    def get_joint_positions(self):
        return self._positions

    def get_odom(self):
        return self._odom


class ContractPreprocessor(PreprocessingMixin):
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


class ContractPreprocessingTest(unittest.TestCase):
    @staticmethod
    def source(feature, kind, group, index):
        return SimpleNamespace(
            feature=feature,
            kind=kind,
            group=group,
            index=index,
        )

    def test_builds_state_in_exact_contract_order_including_odom(self):
        sources = [
            self.source("joint_b", "joint", "follower_arm", 1),
            self.source("joint_a", "joint", "follower_arm", 0),
            self.source("linear_y", "odom", None, 1),
        ]
        robot = ContractRobot(
            {"follower_arm": np.asarray([1.0, 2.0], dtype=np.float32)},
            odom={
                "linear_velocity": [3.0, 4.0, 0.0],
                "angular_velocity": [0.0, 0.0, 5.0],
            },
        )

        batch = ContractPreprocessor(robot, sources)._build_observation("task")

        torch.testing.assert_close(
            batch[STATE_KEY], torch.tensor([[2.0, 1.0, 4.0]])
        )

    def test_missing_named_state_fails_instead_of_padding(self):
        sources = [self.source("joint_b", "joint", "follower_arm", 1)]
        robot = ContractRobot(
            {"follower_arm": np.asarray([1.0], dtype=np.float32)}
        )

        result = ContractPreprocessor(robot, sources)._build_observation("task")

        self.assertFalse(result["success"])
        self.assertIn("Missing exact joint source", result["message"])

    def test_selected_camera_is_delivered_at_contract_shape_and_key(self):
        camera = CameraContract("cam", "observation.images.rgb.cam", 6, 4)
        sources = [self.source("joint_a", "joint", "follower_arm", 0)]
        robot = ContractRobot(
            {"follower_arm": np.asarray([1.0], dtype=np.float32)},
            image=np.zeros((4, 6, 3), dtype=np.uint8),
        )

        batch = ContractPreprocessor(robot, sources, (camera,))._build_observation(
            "task"
        )

        self.assertEqual(tuple(batch[camera.key].shape), (1, 3, 4, 6))

    def test_wrong_camera_shape_fails_instead_of_resizing(self):
        camera = CameraContract("cam", "observation.images.rgb.cam", 6, 4)
        sources = [self.source("joint_a", "joint", "follower_arm", 0)]
        robot = ContractRobot(
            {"follower_arm": np.asarray([1.0], dtype=np.float32)},
            image=np.zeros((2, 3, 3), dtype=np.uint8),
        )

        result = ContractPreprocessor(robot, sources, (camera,))._build_observation(
            "task"
        )

        self.assertFalse(result["success"])
        self.assertIn("produced shape", result["message"])


if __name__ == "__main__":
    unittest.main()
