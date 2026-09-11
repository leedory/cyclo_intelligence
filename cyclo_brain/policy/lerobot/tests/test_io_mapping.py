#!/usr/bin/env python3

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest


robot_client_stub = types.ModuleType("robot_client")
robot_client_stub.RobotClient = object
sys.modules.setdefault("robot_client", robot_client_stub)

ENGINE_DIR = Path(
    os.environ.get(
        "LEROBOT_ENGINE_DIR",
        Path(__file__).resolve().parents[1] / "lerobot_engine",
    )
)
package = types.ModuleType("lerobot_engine")
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault("lerobot_engine", package)

contract_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.policy_contract", ENGINE_DIR / "policy_contract.py"
)
policy_contract = importlib.util.module_from_spec(contract_spec)
sys.modules[contract_spec.name] = policy_contract
contract_spec.loader.exec_module(policy_contract)

mapping_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.io_mapping", ENGINE_DIR / "io_mapping.py"
)
io_mapping = importlib.util.module_from_spec(mapping_spec)
sys.modules[mapping_spec.name] = io_mapping
mapping_spec.loader.exec_module(io_mapping)

CameraContract = policy_contract.CameraContract
PolicyContract = policy_contract.PolicyContract
SimulationContract = policy_contract.SimulationContract
IoMappingMixin = io_mapping.IoMappingMixin


class IoMappingTest(unittest.TestCase):
    def test_robot_client_subscribes_only_to_contract_cameras(self):
        class FakeRobotClient:
            requested = None

            def __init__(self, robot_type, camera_names=None):
                self.robot_type = robot_type
                self.camera_names = list(camera_names or ())
                FakeRobotClient.requested = tuple(camera_names or ())
                self._config = {
                    "cameras": {name: {} for name in self.camera_names},
                    "joint_groups": {
                        "follower_arm": {
                            "role": "follower",
                            "joint_names": ["joint_a"],
                        }
                    },
                    "sensors": {},
                }
                self._action_groups = {"arm": {"joint_names": ["joint_a"]}}

            def is_image_ready(self, name):
                return name == "cam_left_head"

            def is_joint_ready(self, name):
                return name == "follower_arm"

            def is_sensor_ready(self, name):
                return False

        contract = PolicyContract(
            task_id="task",
            robot_type="ffw_sg2_rev1",
            fps=15,
            state_features=("joint_a",),
            action_features=("joint_a",),
            inactive_actions={},
            cameras=(
                CameraContract(
                    "cam_left_head",
                    "observation.images.rgb.cam_left_head",
                    672,
                    376,
                ),
            ),
            simulation=SimulationContract("env", "random_env", "deterministic"),
        )
        engine = IoMappingMixin()
        engine._policy_contract = contract
        engine._robot = None
        original = io_mapping.RobotClient
        self.addCleanup(setattr, io_mapping, "RobotClient", original)
        io_mapping.RobotClient = FakeRobotClient

        engine._init_robot("ffw_sg2_rev1")

        self.assertEqual(FakeRobotClient.requested, ("cam_left_head",))
        self.assertEqual(engine._robot.camera_names, ["cam_left_head"])

    def test_contract_readiness_ignores_unselected_joint_groups_and_sensors(self):
        class FakeRobot:
            def is_image_ready(self, name):
                return name == "cam_left_head"

            def is_joint_ready(self, name):
                return name == "follower_arm_right"

            def is_sensor_ready(self, name):
                return False

        sources = [
            io_mapping.StateSource("arm_r_joint1", "joint", "follower_arm_right", 0),
            io_mapping.StateSource("head_joint1", "joint", "follower_head", 0),
        ]
        missing = io_mapping.missing_contract_inputs(
            FakeRobot(), ["cam_left_head"], sources
        )

        self.assertEqual(missing, ["joint:follower_head"])
        self.assertNotIn("joint:follower_arm_left", missing)
        self.assertNotIn("sensor:odom", missing)

        mobile_sources = [io_mapping.StateSource("linear_x", "odom", None, 0)]
        self.assertEqual(
            io_mapping.missing_contract_inputs(FakeRobot(), [], mobile_sources),
            ["sensor:odom"],
        )

    def test_camera_mapping_uses_exact_declared_source_and_key(self):
        cameras = (
            CameraContract(
                "cam_left_head", "observation.images.rgb.cam_left_head", 672, 376
            ),
            CameraContract(
                "cam_left_wrist", "observation.images.rgb.cam_left_wrist", 480, 640
            ),
        )
        self.assertEqual(
            IoMappingMixin._resolve_camera_mappings(
                ["cam_left_head", "cam_left_wrist", "unused"], cameras
            ),
            {
                "cam_left_head": "observation.images.rgb.cam_left_head",
                "cam_left_wrist": "observation.images.rgb.cam_left_wrist",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "missing=.*cam_left_head"):
            IoMappingMixin._resolve_camera_mappings(["rgb.cam_left_head"], cameras)

    def test_state_mapping_preserves_contract_order_and_prefers_named_child(self):
        groups = {
            "follower_upper_body": {
                "role": "follower",
                "joint_names": ["joint_a", "joint_b"],
            },
            "follower_arm_left": {
                "role": "follower",
                "parent": "follower_upper_body",
                "joint_names": ["joint_a", "joint_b"],
            },
        }
        sources = IoMappingMixin._resolve_state_sources(
            groups,
            ["joint_b", "joint_a", "linear_x", "angular_z"],
            has_odom=True,
        )
        self.assertEqual([source.feature for source in sources], ["joint_b", "joint_a", "linear_x", "angular_z"])
        self.assertEqual([source.group for source in sources[:2]], ["follower_arm_left"] * 2)
        self.assertEqual([source.index for source in sources], [1, 0, 0, 2])

    def test_state_mapping_fails_on_missing_or_ambiguous_exact_source(self):
        with self.assertRaisesRegex(RuntimeError, "0 exact robot sources"):
            IoMappingMixin._resolve_state_sources({}, ["joint_a"], has_odom=False)
        with self.assertRaisesRegex(RuntimeError, "requires observation.state mobile/odom"):
            IoMappingMixin._resolve_state_sources({}, ["linear_x"], has_odom=False)

    def test_action_mapping_requires_complete_exact_groups_in_contract_order(self):
        groups = {
            "arm_left": {"joint_names": ["l1", "l2"]},
            "head": {"joint_names": ["h1", "h2"]},
            "mobile": {"joint_names": ["linear_x", "linear_y", "angular_z"]},
        }
        self.assertEqual(
            IoMappingMixin._resolve_action_keys(
                groups,
                ["h1", "h2", "l1", "l2"],
                {"mobile": "hold_current"},
            ),
            ["head", "arm_left"],
        )
        with self.assertRaisesRegex(RuntimeError, "do not resolve"):
            IoMappingMixin._resolve_action_keys(groups, ["l1"], {})
        with self.assertRaisesRegex(RuntimeError, "both active and inactive"):
            IoMappingMixin._resolve_action_keys(groups, ["h1", "h2"], {"head": "hold_current"})


if __name__ == "__main__":
    unittest.main()
