#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot engine loading helpers (LoadingMixin).

Extracted from ``engine.py`` to keep the core ``LeRobotEngine`` class
focused on the ``InferenceEngine`` API. Mixed into the engine via
multiple inheritance; bind-mounted into the policy container as part
of the ``/app/lerobot_engine/`` package.

Owns:
- ``_resolve_model_dir``: auto-descend lerobot training-output roots.
- ``_load_policy_assets``: load weights + stored pre/post processors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .policy_contract import load_policy_contract, validate_policy_config

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy


logger = logging.getLogger("lerobot_engine")


class LoadingMixin:
    """Policy contract, weight, and saved-processor loading helpers."""

    @staticmethod
    def _load_policy_contract(model_path: str):
        return load_policy_contract(model_path)

    @staticmethod
    def _validate_policy_contract(contract, policy: PreTrainedPolicy) -> None:
        validate_policy_config(contract, policy.config)

    @staticmethod
    def _resolve_model_dir(model_path: str) -> str:
        """Auto-descend lerobot training-output roots.

        Accept a direct pretrained model, a checkpoint directory containing
        ``pretrained_model``, or a run root with ``checkpoints/last``.
        """
        root = Path((model_path or "").strip())
        candidates = (
            root,
            root / "pretrained_model",
            root / "checkpoints" / "last" / "pretrained_model",
        )
        for candidate in candidates:
            if (candidate / "config.json").is_file():
                if candidate != root:
                    logger.info("Resolved pretrained model directory: %s", candidate)
                return str(candidate)
        return str(root)

    @staticmethod
    def _load_policy_assets(
        model_path: str, device: torch.device
    ) -> tuple[PreTrainedPolicy, Any, Any]:
        """Load policy weights + saved pre/post processors."""
        import json

        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            raise RuntimeError(f"checkpoint is missing policy config: {config_path}")
        try:
            with config_path.open(encoding="utf-8") as handle:
                policy_type = json.load(handle).get("type")
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read policy config {config_path}: {exc}") from exc
        if not isinstance(policy_type, str) or not policy_type:
            raise RuntimeError(f"policy config has no non-empty type: {config_path}")

        logger.info("Policy type: %s", policy_type)
        PolicyClass = get_policy_class(policy_type)

        # FastWAM's text encoder must stay on the CPU. Its default config can
        # auto-select CUDA inside ``from_pretrained`` and exhaust VRAM before
        # the offload hook runs, so pin only this policy's initial load to CPU.
        if policy_type == "fastwam":
            policy_config = PreTrainedConfig.from_pretrained(model_path)
            policy_config.device = "cpu"
            policy = PolicyClass.from_pretrained(model_path, config=policy_config)
        else:
            policy = PolicyClass.from_pretrained(model_path)

        # MolmoAct2 errors out unless the action mode is set. We run the
        # continuous (flow matching) head; a checkpoint that names one keeps it.
        if policy_type == "molmoact2" and not getattr(
            policy.config, "inference_action_mode", None
        ):
            policy.config.inference_action_mode = "continuous"

        if policy_type == "fastwam":
            policy = policy.eval()
            logger.info("FastWAM weights loaded on CPU for selective offload")
        else:
            policy = policy.to(device).eval()
            logger.info("Policy weights loaded on %s", device)

        # Stored processor pipelines include the dataset-time normalizer
        # stats and image transforms so we don't re-derive (and de-sync)
        # them. Falling through to the default factory here would wipe
        # those stats and produce garbage actions.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
            },
        )
        logger.info("Pre/post processors loaded")
        return policy, preprocessor, postprocessor
