"""Allowlisted Cyclo Lab simulation-session control for the workstation UI.

The browser cannot supply commands, task IDs, or shell text. Every process
argument comes from ``SIMULATION_ENVIRONMENTS``. Cyclo Lab's ``--ui-session``
bringup mode owns the status document consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading
import time
from typing import Any, Dict, Optional

import docker
from docker.errors import DockerException, NotFound
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
import yaml


router = APIRouter(prefix="/simulator", tags=["simulator"])

CYCLO_LAB_CONTAINER = os.getenv("CYCLO_LAB_CONTAINER_NAME", "cyclo_lab")
S2R_TRAINING_CONTAINER = os.getenv("LEROBOT_CONTAINER_NAME", "lerobot_server_s2r")
ADDITIONAL_TRAINING_CONTAINERS = os.getenv(
    "CYCLO_ADDITIONAL_TRAINING_CONTAINERS", "lerobot_server"
)
TRAINING_CONTAINERS = tuple(dict.fromkeys(
    name
    for name in (
        S2R_TRAINING_CONTAINER,
        *(item.strip() for item in ADDITIONAL_TRAINING_CONTAINERS.split(",")),
    )
    if name
))
CYCLO_LAB_WORKDIR = os.getenv("CYCLO_LAB_WORKDIR", "/workspace/cyclo_lab")
SESSION_STATUS_PATH = "/tmp/cyclo_lab_ui_session.json"
POLICY_ROOT = Path("/workspace/model/lerobot")
POLICY_MANIFEST_NAME = "cyclo_policy.yaml"

SIMULATION_ENVIRONMENTS: Dict[str, Dict[str, Any]] = {
    "task_000458": {
        "label": "Task 000458 — stationary dual-arm",
        "action_dim": 19,
        "profiles": {
            "deterministic": "Cyclo-Real-Showroom-Task000458-FFW-SG2-v0",
            "randomized_evaluation": "Cyclo-Real-Showroom-Task000458-Random-FFW-SG2-v0",
        },
    },
    "task_000525": {
        "label": "Task 000525 — mobile dual-arm",
        "action_dim": 22,
        "profiles": {
            "deterministic": "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0",
            "randomized_evaluation": "Cyclo-Real-Showroom-Task000525-Random-FFW-SG2-v0",
        },
    },
}

_ACTIVE_STATES = {"starting", "ready", "resetting", "running", "stopping", "error"}
_TRAINING_PROCESS_MARKERS = (
    "lerobot_train",
    "train_with_state_noise.py",
    "/tools/training/",
)


class StartSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_path: str
    profile: Optional[str] = None


class ResolvePolicyRequest(BaseModel):
    policy_path: str


@dataclass
class _LocalSession:
    environment: Optional[str] = None
    profile: Optional[str] = None
    policy_path: Optional[str] = None
    exec_id: Optional[str] = None
    reset_requested: bool = False


_SESSION = _LocalSession()
_LOCK = threading.RLock()


def _docker_client():
    try:
        return docker.from_env()
    except DockerException as exc:
        raise HTTPException(status_code=503, detail=f"Docker is unavailable: {exc}") from exc


def _running_container(client, name: str):
    try:
        container = client.containers.get(name)
        container.reload()
    except NotFound as exc:
        raise HTTPException(status_code=503, detail=f"Required container '{name}' was not found") from exc
    except DockerException as exc:
        raise HTTPException(status_code=503, detail=f"Could not inspect '{name}': {exc}") from exc
    if getattr(container, "status", None) != "running":
        raise HTTPException(status_code=503, detail=f"Required container '{name}' is not running")
    return container


def _training_is_active(client) -> bool:
    """Return whether an S2R or explicitly observed external backend is training."""
    for container_name in TRAINING_CONTAINERS:
        try:
            container = client.containers.get(container_name)
            container.reload()
            if getattr(container, "status", None) != "running":
                continue
            process_table = container.top(ps_args="-eo args") or {}
        except (NotFound, DockerException):
            continue

        rows = process_table.get("Processes", [])
        if any(
            marker in " ".join(str(field) for field in row)
            for row in rows
            for marker in _TRAINING_PROCESS_MARKERS
        ):
            return True
    return False


def _read_status_document(container) -> Optional[Dict[str, Any]]:
    """Read the Cyclo Lab status file without invoking a shell in its container."""
    try:
        archive, _ = container.get_archive(SESSION_STATUS_PATH)
        payload = b"".join(archive)
        with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as bundle:
            members = [member for member in bundle.getmembers() if member.isfile()]
            if not members:
                return None
            status_file = bundle.extractfile(members[0])
            if status_file is None:
                return None
            value = json.load(status_file)
            return value if isinstance(value, dict) else None
    except (NotFound, DockerException, OSError, tarfile.TarError, json.JSONDecodeError):
        return None


def _exec_is_running(client, exec_id: Optional[str]) -> bool:
    if not exec_id:
        return False
    try:
        return bool(client.api.exec_inspect(exec_id).get("Running"))
    except DockerException:
        return False


def _exec_pid(client, exec_id: Optional[str]) -> Optional[int]:
    if not exec_id:
        return None
    try:
        pid = int(client.api.exec_inspect(exec_id).get("Pid", 0))
    except (DockerException, TypeError, ValueError):
        return None
    return pid if pid > 1 else None


def _pid_is_alive(container, value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 1:
        return False
    try:
        result = container.exec_run(["kill", "-0", str(pid)])
    except DockerException:
        return False
    return getattr(result, "exit_code", 1) == 0


def _resolve_policy_contract(policy_path: str) -> tuple[str, str, str]:
    """Resolve an allowlisted task/profile from a checkpoint-local manifest."""
    raw_path = Path(str(policy_path or "").strip())
    if not raw_path.is_absolute():
        raise HTTPException(status_code=400, detail="Policy path must be absolute")
    try:
        root = POLICY_ROOT.resolve(strict=True)
        checkpoint = raw_path.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Policy path does not exist: {exc}") from exc
    if checkpoint != root and root not in checkpoint.parents:
        raise HTTPException(status_code=400, detail="Policy path must be under /workspace/model/lerobot")
    start_directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    manifest_path = next(
        (
            directory / POLICY_MANIFEST_NAME
            for directory in (start_directory, *start_directory.parents)
            if (directory == root or root in directory.parents)
            and (directory / POLICY_MANIFEST_NAME).is_file()
        ),
        None,
    )
    if manifest_path is None:
        raise HTTPException(status_code=400, detail=f"Checkpoint is missing {POLICY_MANIFEST_NAME}")
    try:
        contract = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read policy contract: {exc}") from exc
    simulation = contract.get("simulation", {}) if isinstance(contract, dict) else {}
    deterministic_gym_id = str(simulation.get("environment", "")).strip()
    environment = next(
        (
            key for key, value in SIMULATION_ENVIRONMENTS.items()
            if value["profiles"]["deterministic"] == deterministic_gym_id
        ),
        None,
    )
    if environment is None:
        raise HTTPException(status_code=400, detail="Policy contract has an unsupported simulation.environment")
    expected_randomized = SIMULATION_ENVIRONMENTS[environment]["profiles"]["randomized_evaluation"]
    declared_randomized = simulation.get("randomized_environment")
    if declared_randomized is not None and str(declared_randomized).strip() != expected_randomized:
        raise HTTPException(status_code=400, detail="Policy contract has an unsupported simulation.randomized_environment")
    profile = str(simulation.get("default_reset", "")).strip()
    if profile not in SIMULATION_ENVIRONMENTS[environment]["profiles"]:
        raise HTTPException(status_code=400, detail="Policy contract has an unsupported simulation.default_reset")
    return environment, profile, str(checkpoint)


def _resolve_start_request(request: StartSimulationRequest) -> tuple[str, str, str]:
    environment, default_profile, policy_path = _resolve_policy_contract(request.policy_path)
    profile = request.profile or default_profile
    if profile not in SIMULATION_ENVIRONMENTS[environment]["profiles"]:
        raise HTTPException(status_code=400, detail="Profile override is not in the policy task allowlist")
    return environment, profile, policy_path


def _public_status(client, container) -> Dict[str, Any]:
    document = _read_status_document(container) or {}
    with _LOCK:
        local_environment = _SESSION.environment
        local_profile = _SESSION.profile
        local_policy_path = _SESSION.policy_path
        exec_running = _exec_is_running(client, _SESSION.exec_id)
        reset_requested = _SESSION.reset_requested

    reported_task = document.get("task") or document.get("gym_id")
    reported_match = next(
        (
            (task_key, profile_key)
            for task_key, task in SIMULATION_ENVIRONMENTS.items()
            for profile_key, gym_id in task["profiles"].items()
            if gym_id == reported_task
        ),
        (None, None),
    )
    environment = document.get("environment") or reported_match[0] or local_environment
    profile = document.get("profile") or reported_match[1] or local_profile
    configured = SIMULATION_ENVIRONMENTS.get(environment or "", {})
    gym_id = configured.get("profiles", {}).get(profile or "") or reported_task
    reported_state = str(document.get("state", "")).lower()
    pid_alive = _pid_is_alive(container, document.get("pid"))
    samples_ready = (
        int(document.get("observation_sequence", 0) or 0) > 0
        and int(document.get("camera_sequence", 0) or 0) > 0
    )

    if reported_state == "error":
        state = "error"
    elif reported_state in {"ready", "running"} and (exec_running or pid_alive) and not samples_ready:
        state = "starting"
    elif reported_state in _ACTIVE_STATES and (exec_running or pid_alive):
        state = reported_state
    elif exec_running:
        state = "resetting" if reset_requested else "starting"
    else:
        state = "stopped"

    return {
        "available": True,
        "state": state,
        "environment": environment,
        "profile": profile,
        "gym_id": gym_id,
        "reset_count": int(document.get("reset_count", 0) or 0),
        "observation_sequence": int(document.get("observation_sequence", 0) or 0),
        "camera_sequence": int(document.get("camera_sequence", 0) or 0),
        "policy_path": document.get("policy_path") or local_policy_path,
        "error": document.get("error"),
        "message": document.get("error") or document.get("message") or (
            "Waiting for fresh state and camera samples" if state == "starting" else ""
        ),
        "training_active": _training_is_active(client),
    }


def _launch_argv(environment: str, profile: str):
    gym_id = SIMULATION_ENVIRONMENTS[environment]["profiles"][profile]
    return [
        "/isaac-sim/python.sh",
        "scripts/sim2real/bringup.py",
        "--task",
        gym_id,
        "--bridge",
        "ffw_sg2",
        "--headless",
        "--enable_cameras",
        "--ui-session",
        "--session-status-file",
        SESSION_STATUS_PATH,
    ]


@router.get("/environments")
def list_environments():
    return {
        "environments": [
            {
                "key": key,
                "label": value["label"],
                "action_dim": value["action_dim"],
                "profiles": [
                    {
                        "key": profile,
                        "label": "Randomized evaluation" if profile == "randomized_evaluation" else "Deterministic",
                        "gym_id": gym_id,
                    }
                    for profile, gym_id in value["profiles"].items()
                ],
            }
            for key, value in SIMULATION_ENVIRONMENTS.items()
        ]
    }


@router.get("/status")
def get_status():
    client = _docker_client()
    container = _running_container(client, CYCLO_LAB_CONTAINER)
    return _public_status(client, container)


@router.post("/resolve")
def resolve_policy(request: ResolvePolicyRequest):
    environment, profile, policy_path = _resolve_policy_contract(request.policy_path)
    configured = SIMULATION_ENVIRONMENTS[environment]
    return {
        "environment": environment,
        "environment_label": configured["label"],
        "profile": profile,
        "gym_id": configured["profiles"][profile],
        "policy_path": policy_path,
    }


@router.post("/start")
def start_simulation(request: StartSimulationRequest):
    environment, profile, policy_path = _resolve_start_request(request)

    client = _docker_client()
    container = _running_container(client, CYCLO_LAB_CONTAINER)
    current = _public_status(client, container)
    if current["state"] in _ACTIVE_STATES:
        if current["state"] in {"stopping", "error"}:
            raise HTTPException(status_code=409, detail="Stop the current simulation session before deploying")
        if current["environment"] == environment and current["profile"] == profile:
            if policy_path:
                with _LOCK:
                    _SESSION.policy_path = policy_path
                current["policy_path"] = policy_path
            return current
        raise HTTPException(status_code=409, detail="Stop the current simulation session before changing task or profile")
    if current["training_active"]:
        raise HTTPException(status_code=409, detail="A training process is active; simulation launch is disabled")

    # A previous session may have left its small runtime status document behind.
    container.exec_run(["rm", "-f", SESSION_STATUS_PATH])
    try:
        created = client.api.exec_create(
            container.id,
            cmd=_launch_argv(environment, profile),
            workdir=CYCLO_LAB_WORKDIR,
            stdout=True,
            stderr=True,
        )
        exec_id = created["Id"]
        client.api.exec_start(exec_id, detach=True)
    except (DockerException, KeyError) as exc:
        raise HTTPException(status_code=502, detail=f"Could not launch Cyclo Lab: {exc}") from exc

    with _LOCK:
        _SESSION.environment = environment
        _SESSION.profile = profile
        _SESSION.policy_path = policy_path
        _SESSION.exec_id = exec_id
        _SESSION.reset_requested = False
    return _public_status(client, container)


@router.post("/reset")
def reset_simulation():
    client = _docker_client()
    container = _running_container(client, CYCLO_LAB_CONTAINER)
    current = _public_status(client, container)
    if current["state"] not in {"ready", "running"}:
        raise HTTPException(status_code=409, detail="Simulation must be ready before it can be reset")

    try:
        completed = subprocess.run(
            ["ros2", "topic", "pub", "--once", "/simulation/reset", "std_msgs/msg/Empty", "{}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(status_code=502, detail=f"Could not request simulation reset: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "ROS reset publication failed").strip()
        raise HTTPException(status_code=502, detail=detail)

    with _LOCK:
        _SESSION.reset_requested = True
    result = _public_status(client, container)
    result["state"] = "resetting"
    result["message"] = "Waiting for reset completion and fresh observations"
    return result


@router.post("/stop")
def stop_simulation():
    client = _docker_client()
    container = _running_container(client, CYCLO_LAB_CONTAINER)
    document = _read_status_document(container) or {}
    status_pid = document.get("pid")
    pid = status_pid if _pid_is_alive(container, status_pid) else _exec_pid(client, _SESSION.exec_id)

    if pid is not None:
        try:
            pid = int(pid)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail="Cyclo Lab reported an invalid session PID") from exc
        if pid <= 1:
            raise HTTPException(status_code=502, detail="Cyclo Lab reported an unsafe session PID")
        result = container.exec_run(["kill", "-TERM", str(pid)])
        if getattr(result, "exit_code", 0) not in (0, 1):
            raise HTTPException(status_code=502, detail="Could not stop the simulation session")
        deadline = time.monotonic() + 30.0
        while (
            _pid_is_alive(container, pid) or _exec_is_running(client, _SESSION.exec_id)
        ) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _pid_is_alive(container, pid) or _exec_is_running(client, _SESSION.exec_id):
            raise HTTPException(status_code=504, detail="Timed out waiting for simulation process to stop")
    elif _exec_is_running(client, _SESSION.exec_id):
        raise HTTPException(status_code=502, detail="Docker did not report a safe simulation exec PID")

    with _LOCK:
        _SESSION.environment = None
        _SESSION.profile = None
        _SESSION.policy_path = None
        _SESSION.exec_id = None
        _SESSION.reset_requested = False
    container.exec_run(["rm", "-f", SESSION_STATUS_PATH])
    return {
        "available": True,
        "state": "stopped",
        "environment": None,
        "profile": None,
        "policy_path": None,
        "gym_id": None,
        "reset_count": 0,
        "observation_sequence": 0,
        "camera_sequence": 0,
        "message": "UI-launched simulation session stopped",
        "error": None,
        "training_active": _training_is_active(client),
    }
