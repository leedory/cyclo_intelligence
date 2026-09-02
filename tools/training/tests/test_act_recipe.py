from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import sys
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "act_recipe", REPO_ROOT / "tools" / "training" / "act_recipe.py"
)
act_recipe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = act_recipe
SPEC.loader.exec_module(act_recipe)


class ActRecipeTest(unittest.TestCase):
    def resolve(self, task: str):
        return act_recipe.resolve_recipe(
            REPO_ROOT / "training" / task / "act.yaml", validate_data=False
        )

    def test_task_contracts_are_exact_15_hz_19d_and_22d(self):
        fixed = self.resolve("task_000458")
        mobile = self.resolve("task_000525")

        self.assertEqual(len(fixed.state_features), 19)
        self.assertEqual(fixed.state_features, fixed.action_features)
        self.assertNotIn("linear_x", fixed.state_features)
        self.assertEqual(len(mobile.state_features), 22)
        self.assertEqual(mobile.state_features, mobile.action_features)
        self.assertEqual(mobile.state_features[-3:], ("linear_x", "linear_y", "angular_z"))
        self.assertEqual(fixed.payload["policy_io"]["fps"], 15)

    def test_task525_defaults_to_historical_normalized_noise(self):
        resolved = self.resolve("task_000525")

        self.assertEqual(resolved.noise_indices, tuple(range(22)))
        self.assertEqual(resolved.noise_std, (0.01,) * 22)
        command = act_recipe.build_docker_command(resolved, "lerobot_server")
        noise_env = next(value for value in command if value.startswith("CYCLO_STATE_NOISE="))
        payload = json.loads(noise_env.split("=", 1)[1])
        self.assertEqual(payload["space"], "normalized")
        self.assertEqual(payload["features"], list(resolved.state_features))

    def test_optimizer_mode_controls_generated_arguments(self):
        resolved = self.resolve("task_000458")
        preset_args = act_recipe.build_lerobot_args(resolved)
        self.assertIn("--use_policy_training_preset=true", preset_args)
        self.assertIn("--policy.optimizer_lr=1e-05", preset_args)
        self.assertFalse(any(arg.startswith("--optimizer.grad_clip_norm=") for arg in preset_args))

        payload = copy.deepcopy(resolved.payload)
        payload["optimizer"]["mode"] = "custom"
        payload["optimizer"]["custom"]["lr"] = 0.000321
        payload["optimizer"]["custom"]["grad_clip_norm"] = 7.0
        payload["scheduler"] = {"type": "constant_with_warmup", "num_warmup_steps": 123}
        custom = replace(resolved, payload=payload)
        custom_args = act_recipe.build_lerobot_args(custom)
        self.assertIn("--use_policy_training_preset=false", custom_args)
        self.assertIn("--optimizer.lr=0.000321", custom_args)
        self.assertIn("--optimizer.grad_clip_norm=7.0", custom_args)
        self.assertIn("--scheduler.num_warmup_steps=123", custom_args)
        self.assertFalse(any(arg.startswith("--policy.optimizer_lr=") for arg in custom_args))

    def test_manifest_is_concise_and_matches_camera_and_simulation_contract(self):
        manifest = act_recipe.policy_manifest(self.resolve("task_000458"))

        self.assertEqual(
            set(manifest),
            {"task", "robot", "policy_hz", "state", "action", "cameras", "simulation"},
        )
        self.assertEqual(manifest["task"], "task_000458")
        self.assertEqual(manifest["robot"], "ffw_sg2_rev1")
        self.assertEqual(manifest["policy_hz"], 15)
        self.assertEqual(manifest["action"]["inactive"], {})
        self.assertEqual(
            manifest["cameras"]["cam_left_head"],
            {
                "key": "observation.images.rgb.cam_left_head",
                "width": 672,
                "height": 376,
            },
        )
        self.assertEqual(manifest["cameras"]["cam_left_wrist"]["height"], 640)
        self.assertEqual(
            manifest["simulation"]["environment"],
            "Cyclo-Real-Showroom-Task000458-FFW-SG2-v0",
        )
        self.assertNotIn("training", manifest)
        self.assertNotIn("version", manifest)

    def test_finalizer_marks_complete_checkpoints_only(self):
        resolved = self.resolve("task_000458")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            complete = output / "checkpoints" / "010000" / "pretrained_model"
            incomplete = output / "checkpoints" / "020000" / "pretrained_model"
            complete.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            for name in ("config.json", "model.safetensors", "train_config.json"):
                (complete / name).write_text("{}", encoding="utf-8")
                (incomplete / name).write_text("{}", encoding="utf-8")
            state = complete.parent / "training_state"
            state.mkdir()
            (state / "training_step.json").write_text('{"step": 10000}', encoding="utf-8")

            written = act_recipe.finalize_contracts(replace(resolved, output_host_root=output))

            self.assertIn(complete / "cyclo_policy.yaml", written)
            self.assertTrue((output / "cyclo_policy.yaml").is_file())
            self.assertTrue((output / "resolved_recipe.yaml").is_file())
            self.assertFalse((incomplete / "cyclo_policy.yaml").exists())
            root_manifest = yaml.safe_load((output / "cyclo_policy.yaml").read_text())
            self.assertEqual(set(root_manifest), set(act_recipe.policy_manifest(resolved)))
            provenance = yaml.safe_load((output / "resolved_recipe.yaml").read_text())
            self.assertIn("training", provenance)


if __name__ == "__main__":
    unittest.main()
