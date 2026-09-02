#!/usr/bin/env python3

from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "orchestrator"
    / "internal"
    / "communication"
    / "inference_mode.py"
)

spec = importlib.util.spec_from_file_location("inference_mode", HELPER_PATH)
inference_mode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_mode)
publish_to_robot_from_task_info = inference_mode.publish_to_robot_from_task_info


class InferenceModeTests(unittest.TestCase):
    def test_defaults_to_simulation(self) -> None:
        self.assertFalse(publish_to_robot_from_task_info(SimpleNamespace()))

    def test_robot_mode_enables_robot_publish(self) -> None:
        task_info = SimpleNamespace(inference_mode="robot")

        self.assertTrue(publish_to_robot_from_task_info(task_info))

    def test_simulation_mode_blocks_robot_publish(self) -> None:
        task_info = SimpleNamespace(inference_mode="simulation")

        self.assertFalse(publish_to_robot_from_task_info(task_info))

    def test_isaac_mode_publishes_actions_to_simulation_topics(self) -> None:
        task_info = SimpleNamespace(inference_mode="isaac")

        self.assertTrue(publish_to_robot_from_task_info(task_info))

    def test_tags_support_backward_compatible_mode(self) -> None:
        task_info = SimpleNamespace(tags=["inference_mode:robot"])

        self.assertTrue(publish_to_robot_from_task_info(task_info))


if __name__ == "__main__":
    unittest.main()
