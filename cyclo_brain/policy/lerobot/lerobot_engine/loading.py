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
- ``_infer_image_resize``: read per-input-image shape hints off the policy.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, Tuple

import torch

from .image_preprocessing import infer_image_resize_targets

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy


logger = logging.getLogger("lerobot_engine")


class LoadingMixin:
    """Policy load helpers — weights, processors, resize hint."""

    @staticmethod
    def _load_compatible_policy_config(
        model_path: str, policy_class: type[PreTrainedPolicy]
    ) -> PreTrainedConfig | None:
        """Load configs saved by newer LeRobot releases on older runtimes.

        ``pretrained_revision`` only selects a Hub commit, branch, or tag while
        resolving a remote pretrained model.  Once a complete checkpoint is
        local, it has no effect on model construction or weights.  LeRobot
        versions predating that field reject the otherwise compatible config,
        so parse a temporary copy without it and leave the checkpoint intact.

        Return ``None`` when no compatibility handling is needed so the normal
        upstream ``from_pretrained`` path remains unchanged.
        """
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return None

        with config_path.open() as f:
            config_data = json.load(f)

        config_class = policy_class.config_class
        supported_fields = {field.name for field in dataclasses.fields(config_class)}
        if "pretrained_revision" not in config_data or "pretrained_revision" in supported_fields:
            return None

        config_data.pop("pretrained_revision")
        with tempfile.TemporaryDirectory(prefix="lerobot-config-compat-") as temp_dir:
            compatible_config_path = Path(temp_dir) / "config.json"
            with compatible_config_path.open("w") as f:
                json.dump(config_data, f)
            config = PreTrainedConfig.from_pretrained(temp_dir)

        logger.warning(
            "Ignoring checkpoint field 'pretrained_revision': this LeRobot runtime does not support it, "
            "and Hub revision selection is not applicable to the local checkpoint %s",
            model_path,
        )
        return config

    @staticmethod
    def _resolve_model_dir(model_path: str) -> str:
        """Auto-descend lerobot training-output roots.

        Users frequently paste the training-output root which contains
        ``pretrained_model/`` next to ``training_state/``. Strip that
        wrapper if needed so ``from_pretrained`` finds ``config.json``.
        """
        root = Path((model_path or "").strip())
        nested = root / "pretrained_model"
        if not (root / "config.json").exists() and (nested / "config.json").exists():
            logger.info("Descending into pretrained_model: %s", nested)
            return str(nested)
        return str(root)

    @staticmethod
    def _load_policy_assets(
        model_path: str, device: torch.device
    ) -> tuple[PreTrainedPolicy, Any, Any]:
        """Load policy weights + saved pre/post processors."""
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                policy_type = json.load(f).get("type", "act")
        else:
            # ACT was the original default; fall back to it for
            # checkpoints saved before ``type`` started being recorded.
            policy_type = "act"

        logger.info("Policy type: %s", policy_type)
        PolicyClass = get_policy_class(policy_type)

        # Newer checkpoints may include Hub-only metadata that an older
        # container does not know. Parse a compatible temporary copy when
        # needed, while loading the original checkpoint weights unchanged.
        policy_config = LoadingMixin._load_compatible_policy_config(model_path, PolicyClass)
        if policy_config is None:
            policy = PolicyClass.from_pretrained(model_path)
        else:
            policy = PolicyClass.from_pretrained(model_path, config=policy_config)
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

    def _infer_image_resize(self, policy: PreTrainedPolicy) -> Dict[str, Tuple[int, int]]:
        """Best-effort per-policy-key target ``(W, H)`` from config.

        Many lerobot policies advertise the expected image shape under
        ``input_features['observation.images.<cam>'].shape = (C, H, W)``.
        Pre-resizing on the host keeps mixed camera shapes aligned with the
        dataset metadata. Missing keys mean: leave that camera at native size.
        """
        try:
            features = getattr(policy.config, "input_features", {}) or {}
            return infer_image_resize_targets(features)
        except Exception:
            pass
        return {}
