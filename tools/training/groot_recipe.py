#!/usr/bin/env python3
"""Validate and run SG2 GR00T training recipes.

This wrapper keeps GR00T invocations on the GR00T policy container while
reusing the shared LeRobot recipe implementation in ``act_recipe.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import act_recipe


def _with_default_groot_container(argv: list[str]) -> list[str]:
    if "--container" in argv or any(arg.startswith("--container=") for arg in argv):
        return argv
    return [*argv, "--container", os.environ.get("GROOT_CONTAINER_NAME", "groot_server")]


def main(argv: list[str] | None = None) -> int:
    return act_recipe.main(_with_default_groot_container(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
