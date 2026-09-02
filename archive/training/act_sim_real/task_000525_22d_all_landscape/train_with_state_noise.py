#!/usr/bin/env python3
"""Run LeRobot training with Gaussian noise on training state samples only."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch


STATE_KEY = "observation.state"
STATE_NOISE_STD_ENV = "ACT_STATE_NOISE_STD"
STATE_DIM_ENV = "ACT_STATE_DIM"


def _read_positive_float(name: str) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def _read_positive_int(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def add_state_noise(batch: dict[str, Any], noise_std: float, state_dim: int) -> dict[str, Any]:
    """Copy a normalized training batch and perturb only its state tensor."""
    if STATE_KEY not in batch:
        raise KeyError(f"Training batch does not contain {STATE_KEY!r}")

    state = batch[STATE_KEY]
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{STATE_KEY!r} must be a torch.Tensor, got {type(state).__name__}")
    if not state.is_floating_point():
        raise TypeError(f"{STATE_KEY!r} must be floating point, got {state.dtype}")
    if state.ndim == 0 or state.shape[-1] != state_dim:
        raise ValueError(f"Expected {STATE_KEY!r} last dimension {state_dim}, got shape {tuple(state.shape)}")

    noisy_batch = dict(batch)
    noisy_batch[STATE_KEY] = state + torch.randn_like(state) * noise_std
    return noisy_batch


def _validate_dataset_schema(dataset: Any, state_dim: int) -> None:
    try:
        shape = tuple(dataset.meta.features[STATE_KEY]["shape"])
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"Dataset metadata does not define {STATE_KEY!r}") from exc
    if shape != (state_dim,):
        raise ValueError(f"Expected dataset {STATE_KEY!r} shape ({state_dim},), got {shape}")


def main() -> None:
    noise_std = _read_positive_float(STATE_NOISE_STD_ENV)
    state_dim = _read_positive_int(STATE_DIM_ENV)

    import lerobot.scripts.lerobot_train as lerobot_train

    original_make_datasets = lerobot_train.make_train_eval_datasets
    original_update_policy = lerobot_train.update_policy

    def make_train_eval_datasets_with_schema_check(cfg: Any) -> tuple[Any, Any]:
        train_dataset, eval_dataset = original_make_datasets(cfg)
        _validate_dataset_schema(train_dataset, state_dim)
        logging.info(
            "Applying Gaussian state augmentation after normalization to training batches only: "
            "key=%s dim=%d normalized_std=%g",
            STATE_KEY,
            state_dim,
            noise_std,
        )
        return train_dataset, eval_dataset

    def update_policy_with_state_noise(
        train_metrics: Any,
        policy: Any,
        batch: dict[str, Any],
        optimizer: Any,
        grad_clip_norm: float,
        accelerator: Any,
        lr_scheduler: Any = None,
        lock: Any = None,
        sample_weighter: Any = None,
    ) -> tuple[Any, Any]:
        return original_update_policy(
            train_metrics,
            policy,
            add_state_noise(batch, noise_std, state_dim),
            optimizer,
            grad_clip_norm,
            accelerator,
            lr_scheduler=lr_scheduler,
            lock=lock,
            sample_weighter=sample_weighter,
        )

    # Patch module-scope references before main(). Evaluation does not call update_policy,
    # so held-out data and environment rollouts remain noise-free.
    lerobot_train.make_train_eval_datasets = make_train_eval_datasets_with_schema_check
    lerobot_train.update_policy = update_policy_with_state_noise

    lerobot_train.main()


if __name__ == "__main__":
    main()
