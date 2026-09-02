from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


MODULE_PATH = Path(
    os.environ.get(
        "CYCLO_NOISE_TRAINER_PATH",
        Path(__file__).resolve().parents[1] / "lerobot_train_with_state_noise.py",
    )
)
SPEC = importlib.util.spec_from_file_location("state_noise_trainer", MODULE_PATH)
state_noise = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(state_noise)


class OneSampleDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {"observation.state": torch.tensor([1.0, 2.0, 3.0]), "clean": True}


class StateNoiseTest(unittest.TestCase):
    def test_raw_hook_wraps_train_dataset_only(self):
        train = OneSampleDataset()
        evaluation = object()
        original_update = object()
        module = SimpleNamespace(
            make_train_eval_datasets=lambda cfg: (train, evaluation),
            update_policy=original_update,
        )

        state_noise.install_noise_hooks(module, "raw", (1,), (0.5,))

        wrapped_train, returned_eval = module.make_train_eval_datasets(object())
        self.assertIsInstance(wrapped_train, state_noise.RawStateNoiseDataset)
        self.assertIs(returned_eval, evaluation)
        self.assertIs(module.update_policy, original_update)
        with patch.object(state_noise.torch, "randn_like", side_effect=torch.ones_like):
            sample = wrapped_train[0]
        torch.testing.assert_close(sample["observation.state"], torch.tensor([1.0, 2.5, 3.0]))

    def test_normalized_hook_changes_selected_preprocessed_columns_only(self):
        seen = {}

        def original_update(train_metrics, policy, batch, optimizer, *args, **kwargs):
            seen["batch"] = batch
            return "updated"

        original_make = lambda cfg: (OneSampleDataset(), object())
        module = SimpleNamespace(
            make_train_eval_datasets=original_make,
            update_policy=original_update,
        )
        state_noise.install_noise_hooks(module, "normalized", (0, 2), (0.1, 0.3))
        preprocessed = {"observation.state": torch.tensor([[10.0, 20.0, 30.0]])}

        with patch.object(state_noise.torch, "randn_like", side_effect=torch.ones_like):
            result = module.update_policy(None, None, preprocessed, None)

        self.assertEqual(result, "updated")
        torch.testing.assert_close(
            seen["batch"]["observation.state"], torch.tensor([[10.1, 20.0, 30.3]])
        )
        torch.testing.assert_close(
            preprocessed["observation.state"], torch.tensor([[10.0, 20.0, 30.0]])
        )
        self.assertIs(module.make_train_eval_datasets, original_make)


if __name__ == "__main__":
    unittest.main()
