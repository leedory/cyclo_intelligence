from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "lerobot_engine" / "policy_contract.py"
SPEC = importlib.util.spec_from_file_location("policy_contract_test_module", MODULE_PATH)
policy_contract = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = policy_contract
SPEC.loader.exec_module(policy_contract)


def manifest():
    return {
        "task": "task_000458",
        "robot": "ffw_sg2_rev1",
        "policy_hz": 15,
        "state": {"names": ["joint_a", "joint_b"]},
        "action": {"names": ["joint_a", "joint_b"], "inactive": {}},
        "cameras": {
            "cam_left_head": {
                "key": "observation.images.rgb.cam_left_head",
                "width": 672,
                "height": 376,
            }
        },
        "simulation": {
            "environment": "Cyclo-Real-Showroom-Task000458-FFW-SG2-v0",
            "randomized_environment": "Cyclo-Real-Showroom-Task000458-Random-FFW-SG2-v0",
            "default_reset": "deterministic",
        },
    }


class PolicyContractTest(unittest.TestCase):
    def load(self, payload):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "cyclo_policy.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return policy_contract.load_policy_contract(path.parent)

    def test_loads_concise_manifest(self):
        contract = self.load(manifest())

        self.assertEqual(contract.task_id, "task_000458")
        self.assertEqual(contract.state_features, ("joint_a", "joint_b"))
        self.assertEqual(contract.cameras[0].source, "cam_left_head")
        self.assertEqual(contract.cameras[0].height, 376)

    def test_rejects_extra_training_or_version_fields(self):
        for field in ("training", "version", "hash"):
            with self.subTest(field=field):
                payload = manifest()
                payload[field] = {}
                with self.assertRaisesRegex(RuntimeError, "root fields mismatch"):
                    self.load(payload)

    def test_validates_exact_config_dimensions_keys_and_shapes(self):
        contract = self.load(manifest())
        config = SimpleNamespace(
            input_features={
                "observation.state": SimpleNamespace(shape=(2,)),
                "observation.images.rgb.cam_left_head": SimpleNamespace(shape=(3, 376, 672)),
            },
            output_features={"action": SimpleNamespace(shape=(2,))},
        )
        policy_contract.validate_policy_config(contract, config)

        config.input_features["observation.state"] = SimpleNamespace(shape=(3,))
        with self.assertRaisesRegex(RuntimeError, "policy state shape"):
            policy_contract.validate_policy_config(contract, config)

    def test_missing_contract_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "missing required policy contract"):
                policy_contract.load_policy_contract(temporary)


if __name__ == "__main__":
    unittest.main()
