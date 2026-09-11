#!/usr/bin/env python3
"""Run LeRobot training with Gaussian noise on named state columns.

This is staged into ``/workspace`` by ``act_recipe.py`` only when a recipe
enables state noise. Raw noise wraps only the training dataset before the saved
preprocessor; normalized noise wraps ``update_policy`` after that preprocessor.
Evaluation data, actions, images, and the dataset on disk are never modified.
"""

from __future__ import annotations

import json
import os
from functools import wraps

import draccus
import torch
from torch.utils.data import Dataset

from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train


class RawStateNoiseDataset(Dataset):
    def __init__(self, dataset, indices: tuple[int, ...], std: tuple[float, ...]):
        self.dataset = dataset
        self.indices = indices
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return add_state_noise(self.dataset[index], self.indices, self.std)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def add_state_noise(batch, indices: tuple[int, ...], std: tuple[float, ...]):
    """Return a shallow batch copy with independent noise on selected columns."""
    result = dict(batch)
    state = result["observation.state"].clone()
    selected = state[..., list(indices)]
    scales = torch.as_tensor(std, dtype=state.dtype, device=state.device)
    state[..., list(indices)] = selected + torch.randn_like(selected) * scales
    result["observation.state"] = state
    return result


def wrap_update_policy(original, indices: tuple[int, ...], std: tuple[float, ...]):
    """Inject noise into the already-preprocessed training batch only."""
    @wraps(original)
    def update_policy_with_noise(train_metrics, policy, batch, optimizer, *args, **kwargs):
        noisy_batch = add_state_noise(batch, indices, std)
        return original(train_metrics, policy, noisy_batch, optimizer, *args, **kwargs)

    return update_policy_with_noise


def install_noise_hooks(lerobot_train_module, space, indices, std) -> None:
    if space == "raw":
        original_make_datasets = lerobot_train_module.make_train_eval_datasets

        def make_train_eval_datasets_with_noise(cfg):
            train_dataset, eval_dataset = original_make_datasets(cfg)
            return RawStateNoiseDataset(train_dataset, indices, std), eval_dataset

        lerobot_train_module.make_train_eval_datasets = make_train_eval_datasets_with_noise
        return
    if space == "normalized":
        lerobot_train_module.update_policy = wrap_update_policy(
            lerobot_train_module.update_policy, indices, std
        )
        return
    raise ValueError(f"unsupported state noise space: {space!r}")


def main() -> None:
    raw = os.environ.get("CYCLO_STATE_NOISE")
    if not raw:
        raise RuntimeError("CYCLO_STATE_NOISE is required")
    payload = json.loads(raw)
    space = payload.get("space")
    indices = tuple(int(index) for index in payload["indices"])
    std = tuple(float(value) for value in payload["std"])
    if not indices or len(indices) != len(std) or any(value <= 0.0 for value in std):
        raise ValueError("state noise requires matching non-empty indices and positive std arrays")

    install_noise_hooks(lerobot_train, space, indices, std)
    cfg = draccus.parse(TrainPipelineConfig)
    lerobot_train.train(cfg)


if __name__ == "__main__":
    main()
