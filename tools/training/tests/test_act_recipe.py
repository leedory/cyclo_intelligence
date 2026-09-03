from __future__ import annotations

import copy
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest import mock

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

    @staticmethod
    def select_cameras(resolved, *names: str):
        payload = copy.deepcopy(resolved.payload)
        payload["policy_io"]["camera_inputs"] = list(names)
        return replace(resolved, payload=payload, camera_inputs=tuple(names))

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
        self.assertEqual(fixed.camera_inputs, ("head", "left_wrist", "right_wrist"))

    def test_cli_defaults_to_s2r_lerobot_container(self):
        previous = os.environ.pop("LEROBOT_CONTAINER_NAME", None)
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("LEROBOT_CONTAINER_NAME", previous)
                if previous is not None
                else os.environ.pop("LEROBOT_CONTAINER_NAME", None)
            )
        )

        args = act_recipe.parse_args(["plan", str(REPO_ROOT / "training/task_000458/act.yaml")])

        self.assertEqual(args.container, "lerobot_server_s2r")

    def test_camera_inputs_control_act_features_and_deployment_manifest(self):
        resolved = self.resolve("task_000458")

        for selected, expected_sources in (
            (("head",), {"cam_left_head"}),
            (("head", "left_wrist"), {"cam_left_head", "cam_left_wrist"}),
            (("left_wrist", "right_wrist"), {"cam_left_wrist", "cam_right_wrist"}),
        ):
            with self.subTest(selected=selected):
                candidate = self.select_cameras(resolved, *selected)
                feature_arg = next(
                    arg
                    for arg in act_recipe.build_lerobot_args(candidate)
                    if arg.startswith("--policy.input_features=")
                )
                features = json.loads(feature_arg.split("=", 1)[1])
                self.assertIn("observation.state", features)
                image_keys = {
                    key for key in features if key.startswith("observation.images.")
                }
                self.assertEqual(
                    image_keys,
                    {
                        resolved.payload["policy_io"]["cameras"][name]["key"]
                        for name in selected
                    },
                )
                self.assertEqual(
                    set(act_recipe.policy_manifest(candidate)["cameras"]),
                    expected_sources,
                )

    def test_camera_inputs_reject_empty_duplicate_or_unknown_names(self):
        payload = copy.deepcopy(self.resolve("task_000458").payload)
        payload.pop("resolved_policy_io")
        payload.pop("source_recipe")
        for selected, message in (
            ([], "must not be empty"),
            (["head", "head"], "contains duplicates"),
            (["head", "elbow"], "unknown cameras"),
        ):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temporary:
                payload["policy_io"]["camera_inputs"] = selected
                recipe_path = Path(temporary) / "act.yaml"
                recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
                with self.assertRaisesRegex(act_recipe.RecipeError, message):
                    act_recipe.resolve_recipe(recipe_path, validate_data=False)

    def test_dataset_may_contain_declared_unselected_cameras(self):
        resolved = self.select_cameras(self.resolve("task_000458"), "head")
        cameras = resolved.payload["policy_io"]["cameras"]
        features = {
            "observation.state": {
                "shape": [len(resolved.state_features)],
                "names": list(resolved.state_features),
            },
            "action": {
                "shape": [len(resolved.action_features)],
                "names": list(resolved.action_features),
            },
            **{
                camera["key"]: {"shape": [3, camera["height"], camera["width"]]}
                for camera in cameras.values()
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            info_path = root / "meta" / "info.json"
            info = {
                "robot_type": resolved.payload["task"]["robot_type"],
                "fps": 15,
                "features": features,
            }
            info_path.write_text(json.dumps(info), encoding="utf-8")

            act_recipe.validate_dataset(
                resolved.payload,
                root,
                resolved.state_features,
                resolved.action_features,
            )

            features[cameras["left_wrist"]["key"]]["shape"] = [3, 1, 1]
            info_path.write_text(json.dumps(info), encoding="utf-8")
            act_recipe.validate_dataset(
                resolved.payload,
                root,
                resolved.state_features,
                resolved.action_features,
            )

            features[cameras["head"]["key"]]["shape"] = [3, 1, 1]
            info_path.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaisesRegex(act_recipe.RecipeError, "camera shape mismatch"):
                act_recipe.validate_dataset(
                    resolved.payload,
                    root,
                    resolved.state_features,
                    resolved.action_features,
                )

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

    def test_contract_write_falls_back_to_training_container_on_permission_error(self):
        path = act_recipe.HOST_WORKSPACE / "model" / "run" / "cyclo_policy.yaml"
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            mock.patch.object(
                act_recipe, "_write_if_changed", side_effect=PermissionError("root owned")
            ),
            mock.patch.object(act_recipe, "_container_running", return_value=True),
            mock.patch.object(act_recipe.subprocess, "run", return_value=completed) as run,
        ):
            act_recipe._write_contract(path, "task: smoke\n", "lerobot_server_s2r")

        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["docker", "exec", "-i", "lerobot_server_s2r"])
        self.assertEqual(command[-1], "/workspace/model/run/cyclo_policy.yaml")
        self.assertEqual(run.call_args.kwargs["input"], "task: smoke\n")


if __name__ == "__main__":
    unittest.main()
