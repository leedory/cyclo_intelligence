#!/usr/bin/env python3

import sys
import types
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace


robot_client_stub = types.ModuleType("robot_client")
robot_client_stub.RobotClient = object
sys.modules.setdefault("robot_client", robot_client_stub)

ENGINE_DIR = Path(__file__).resolve().parents[1] / "lerobot_engine"
package = types.ModuleType("lerobot_engine")
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault("lerobot_engine", package)

spec = importlib.util.spec_from_file_location(
    "lerobot_engine.io_mapping",
    ENGINE_DIR / "io_mapping.py",
)
io_mapping = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = io_mapping
spec.loader.exec_module(io_mapping)
IoMappingMixin = io_mapping.IoMappingMixin


class IoMappingCameraAliasTest(unittest.TestCase):
    def test_maps_rgb_prefixed_cameras_to_policy_keys(self):
        robot_cameras = [
            "rgb.cam_left_head",
            "rgb.cam_right_head",
            "rgb.cam_left_wrist",
            "rgb.cam_right_wrist",
        ]
        policy_keys = {
            "observation.images.cam_left_head",
            "observation.images.cam_right_head",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        }

        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(robot_cameras, policy_keys),
            {
                "rgb.cam_left_head": "observation.images.cam_left_head",
                "rgb.cam_right_head": "observation.images.cam_right_head",
                "rgb.cam_left_wrist": "observation.images.cam_left_wrist",
                "rgb.cam_right_wrist": "observation.images.cam_right_wrist",
            },
        )

    def test_keeps_exact_camera_key_preferred(self):
        robot_cameras = ["rgb.cam_left_head"]
        policy_keys = {
            "observation.images.rgb.cam_left_head",
            "observation.images.cam_left_head",
        }

        with self.assertRaisesRegex(RuntimeError, "Missing camera mappings"):
            IoMappingMixin._resolve_camera_mappings(robot_cameras, policy_keys)

        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(
                robot_cameras,
                {"observation.images.rgb.cam_left_head"},
            ),
            {"rgb.cam_left_head": "observation.images.rgb.cam_left_head"},
        )

    def test_maps_legacy_single_head_policy_key_to_left_head_camera(self):
        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(
                [
                    "cam_left_head",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ],
                {
                    "observation.images.rgb.cam_head",
                    "observation.images.cam_wrist_left",
                    "observation.images.cam_wrist_right",
                },
            ),
            {
                "cam_left_head": "observation.images.rgb.cam_head",
                "cam_left_wrist": "observation.images.cam_wrist_left",
                "cam_right_wrist": "observation.images.cam_wrist_right",
            },
        )


class IoMappingPolicyLayoutTest(unittest.TestCase):
    def setUp(self):
        self.mapper = IoMappingMixin()
        self.mapper._robot = SimpleNamespace(
            _config={
                "joint_groups": {
                    "follower_arm_left": {
                        "joint_names": [f"arm_l_joint{i}" for i in range(1, 8)]
                        + ["gripper_l_joint1"],
                    },
                    "follower_arm_right": {
                        "joint_names": [f"arm_r_joint{i}" for i in range(1, 8)]
                        + ["gripper_r_joint1"],
                    },
                    "follower_head": {
                        "joint_names": ["head_joint1", "head_joint2"],
                    },
                    "follower_lift": {"joint_names": ["lift_joint"]},
                }
            },
            _action_groups={
                "mobile": {
                    "joint_names": ["linear_x", "linear_y", "angular_z"],
                }
            },
        )
        self.available = ["arm_left", "arm_right", "head", "lift", "mobile"]

    def test_uses_explicit_reduced_joint_layout(self):
        expected_names = [f"arm_r_joint{i}" for i in range(1, 8)] + [
            "gripper_r_joint1",
            "lift_joint",
            "linear_x",
            "linear_y",
            "angular_z",
        ]
        policy_io = {
            "observation_state_modalities": ["arm_right", "lift", "mobile"],
            "observation_state_joint_names": expected_names,
        }

        self.assertEqual(
            self.mapper._resolve_policy_modalities(
                self.available,
                12,
                policy_io,
                "observation_state_modalities",
                "observation_state_joint_names",
                label="observation.state",
            ),
            ["arm_right", "lift", "mobile"],
        )

    def test_rejects_ambiguous_dimension_without_sidecar(self):
        with self.assertRaisesRegex(RuntimeError, "cyclo_policy_io.json"):
            self.mapper._resolve_policy_modalities(
                self.available,
                12,
                {},
                "observation_state_modalities",
                "observation_state_joint_names",
                label="observation.state",
            )

    def test_scopes_wrist_rotation_to_a_policy_mapping(self):
        active = {
            "cam_left_head": "observation.images.rgb.cam_left_head",
            "cam_left_wrist": "observation.images.rgb.cam_left_wrist",
            "cam_right_wrist": "observation.images.rgb.cam_right_wrist",
        }

        self.assertEqual(
            self.mapper._resolve_camera_rotation_overrides(
                {
                    "camera_rotation_deg": {
                        "cam_left_wrist": 270,
                        "cam_right_wrist": 270,
                    }
                },
                active,
            ),
            {"cam_left_wrist": 270, "cam_right_wrist": 270},
        )
        self.assertEqual(
            self.mapper._resolve_camera_rotation_overrides({}, active),
            {},
        )

    def test_loads_458_model_mapping_from_engine_registry(self):
        mapping = self.mapper._load_policy_io_mapping(
            "/models/Task_000458_peanut_mix_sim_real_act_Intern/"
            "checkpoints/050000/pretrained_model"
        )

        self.assertEqual(
            mapping["action_modalities"],
            ["arm_right", "lift", "mobile"],
        )

    def test_loads_458_sim_act_model_mapping_from_engine_registry(self):
        mapping = self.mapper._load_policy_io_mapping(
            "/models/Task_000458_peanut_mix_sim_act_Intern/"
            "checkpoints/050000/pretrained_model"
        )

        self.assertEqual(
            mapping["action_modalities"],
            ["arm_right", "lift", "mobile"],
        )

    def test_loads_458_sim_only_model_mapping_from_engine_registry(self):
        mapping = self.mapper._load_policy_io_mapping(
            "/models/task_000458_peanut_mix_sim_only_act/"
            "checkpoints/050000/pretrained_model"
        )

        self.assertEqual(
            mapping["observation_state_modalities"],
            ["arm_right", "lift", "mobile"],
        )


if __name__ == "__main__":
    unittest.main()
