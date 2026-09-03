#!/usr/bin/env python3

import sys
import types
import unittest
import importlib.util
from pathlib import Path


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
contract_io = sys.modules["lerobot_engine.contract_io"]
CameraContract = contract_io.CameraContract

contract_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.policy_contract",
    ENGINE_DIR / "policy_contract.py",
)
policy_contract = sys.modules.get(contract_spec.name)
if policy_contract is None:
    policy_contract = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_spec.name] = policy_contract
    contract_spec.loader.exec_module(policy_contract)
PolicyContract = policy_contract.PolicyContract
SimulationContract = policy_contract.SimulationContract


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

class IoMappingContractTest(unittest.TestCase):
    def test_robot_client_subscribes_only_to_declared_cameras(self):
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

            def wait_for_ready(self, timeout):
                return True

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
        mapper = IoMappingMixin()
        mapper._policy_contract = contract
        original = contract_io.RobotClient
        self.addCleanup(setattr, contract_io, "RobotClient", original)
        contract_io.RobotClient = FakeRobotClient

        mapper._init_robot("ffw_sg2_rev1")

        self.assertEqual(FakeRobotClient.requested, ("cam_left_head",))
        self.assertEqual(mapper._robot.camera_names, ["cam_left_head"])

    def test_contract_readiness_ignores_unselected_joint_groups_and_sensors(self):
        class FakeRobot:
            def is_image_ready(self, name):
                return name == "cam_left_head"

            def is_joint_ready(self, name):
                return name == "follower_arm_right"

            def is_sensor_ready(self, name):
                return False

        sources = [
            contract_io.StateSource("arm_r_joint1", "joint", "follower_arm_right", 0),
            contract_io.StateSource("head_joint1", "joint", "follower_head", 0),
        ]
        robot = FakeRobot()

        self.assertEqual(
            contract_io.missing_contract_inputs(robot, ["cam_left_head"], sources),
            ["joint:follower_head"],
        )
        self.assertNotIn(
            "joint:follower_arm_left",
            contract_io.missing_contract_inputs(robot, ["cam_left_head"], sources),
        )
        self.assertNotIn(
            "sensor:odom",
            contract_io.missing_contract_inputs(robot, ["cam_left_head"], sources),
        )

        mobile_sources = [contract_io.StateSource("linear_x", "odom", None, 0)]
        self.assertEqual(
            contract_io.missing_contract_inputs(robot, [], mobile_sources),
            ["sensor:odom"],
        )

    def test_camera_mapping_requires_exact_declared_source(self):
        cameras = (
            CameraContract(
                "cam_left_head",
                "observation.images.rgb.cam_left_head",
                672,
                376,
            ),
            CameraContract(
                "cam_left_wrist",
                "observation.images.rgb.cam_left_wrist",
                480,
                640,
            ),
        )
        self.assertEqual(
            contract_io.resolve_camera_mappings(
                ["cam_left_head", "cam_left_wrist"], cameras
            ),
            {
                "cam_left_head": "observation.images.rgb.cam_left_head",
                "cam_left_wrist": "observation.images.rgb.cam_left_wrist",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "missing=.*cam_left_head"):
            contract_io.resolve_camera_mappings(
                ["rgb.cam_left_head", "cam_left_wrist"], cameras
            )

    def test_state_mapping_preserves_contract_order_and_named_child(self):
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

        sources = contract_io.resolve_state_sources(
            groups,
            ["joint_b", "joint_a", "linear_x", "angular_z"],
            has_odom=True,
        )

        self.assertEqual(
            [source.feature for source in sources],
            ["joint_b", "joint_a", "linear_x", "angular_z"],
        )
        self.assertEqual(
            [source.group for source in sources[:2]],
            ["follower_arm_left", "follower_arm_left"],
        )
        self.assertEqual([source.index for source in sources], [1, 0, 0, 2])

    def test_contract_mapping_fails_for_missing_state_or_partial_action_group(self):
        with self.assertRaisesRegex(RuntimeError, "0 exact robot sources"):
            contract_io.resolve_state_sources(
                {}, ["joint_a"], has_odom=False
            )
        with self.assertRaisesRegex(RuntimeError, "requires observation.state mobile/odom"):
            contract_io.resolve_state_sources(
                {}, ["linear_x"], has_odom=False
            )

        action_groups = {
            "arm_left": {"joint_names": ["l1", "l2"]},
            "head": {"joint_names": ["h1", "h2"]},
            "mobile": {"joint_names": ["linear_x", "linear_y", "angular_z"]},
        }
        self.assertEqual(
            contract_io.resolve_action_keys(
                action_groups,
                ["h1", "h2", "l1", "l2"],
                {"mobile": "hold_current"},
            ),
            ["head", "arm_left"],
        )
        with self.assertRaisesRegex(RuntimeError, "do not resolve"):
            contract_io.resolve_action_keys(
                action_groups, ["l1"], {}
            )
        with self.assertRaisesRegex(RuntimeError, "both active and inactive"):
            contract_io.resolve_action_keys(
                action_groups,
                ["h1", "h2"],
                {"head": "hold_current"},
            )
if __name__ == "__main__":
    unittest.main()
