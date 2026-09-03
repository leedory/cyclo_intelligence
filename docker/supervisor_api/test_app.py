import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

APP_PATH = Path(__file__).resolve().with_name("app.py")
REPO_ROOT = APP_PATH.parents[2]

docker_stub = types.ModuleType("docker")
docker_errors_stub = types.ModuleType("docker.errors")


class DockerException(Exception):
    pass


class ImageNotFound(DockerException):
    pass


class NotFound(DockerException):
    pass


docker_stub.from_env = lambda: None
docker_errors_stub.DockerException = DockerException
docker_errors_stub.ImageNotFound = ImageNotFound
docker_errors_stub.NotFound = NotFound
sys.modules["docker"] = docker_stub
sys.modules["docker.errors"] = docker_errors_stub

original_path = list(sys.path)
sys.path = [
    path for path in sys.path
    if Path(path or ".").resolve() != REPO_ROOT
]
try:
    spec = importlib.util.spec_from_file_location("supervisor_api_app", APP_PATH)
    app = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)
finally:
    sys.path = original_path

_missing_required_mounts = app._missing_required_mounts
_mount_source_for_destination = app._mount_source_for_destination
_backend_container_image_mismatch = app._backend_container_image_mismatch
_backend_container_stale_reason = app._backend_container_stale_reason
_compose_env = app._compose_env
_host_workspace_dir = app._host_workspace_dir
_require_known_service = app._require_known_service
_validate_bt_robot_type = app._validate_bt_robot_type
_validate_robot_type = app._validate_robot_type
_write_bt_robot_type = app._write_bt_robot_type
_resolve_groot_trt_paths = app._resolve_groot_trt_paths
_trt_status = app._trt_status
_BACKENDS = app._BACKENDS
_USER_SERVICES = app._USER_SERVICES
navigation = sys.modules["supervisor_api.navigation"]
navigation_grid_cache = sys.modules["supervisor_api.navigation_grid_cache"]
simulator = sys.modules["supervisor_api.simulator"]
_GROOT_REQUIRED_MOUNTS = app._REQUIRED_BACKEND_MOUNTS["groot"]
_LEROBOT_REQUIRED_MOUNTS = app._REQUIRED_BACKEND_MOUNTS["lerobot"]


def test_navigation_parses_binary_pgm():
    data = b"P5\n# map\n2 2\n255\n" + bytes([0, 127, 254, 255])

    assert navigation._parse_pgm(data) == (
        2,
        2,
        255,
        [0, 127, 254, 255],
    )


def test_navigation_rejects_map_path_escape():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        navigation._resolve_pgm_path("../../outside.pgm")


def test_navigation_validates_map_name():
    import pytest
    from fastapi import HTTPException

    assert navigation._validate_map_name("factory-1") == "factory-1"
    with pytest.raises(HTTPException):
        navigation._validate_map_name("factory; reboot")


def test_navigation_routes_are_registered():
    paths = {route.path for route in app.app.routes if hasattr(route, "path")}

    assert "/navigation/status" in paths
    assert "/navigation/start" in paths
    assert "/navigation/maps/pgm/save" in paths
    assert "/navigation/topics/ws" in paths


def test_simulator_routes_are_registered():
    paths = {route.path for route in app.app.routes if hasattr(route, "path")}

    assert "/simulator/environments" in paths
    assert "/simulator/status" in paths
    assert "/simulator/resolve" in paths
    assert "/simulator/start" in paths
    assert "/simulator/reset" in paths
    assert "/simulator/stop" in paths


def test_simulator_launch_argv_is_fixed_and_allowlisted():
    assert simulator._launch_argv("task_000458", "randomized_evaluation") == [
        "/isaac-sim/python.sh",
        "scripts/sim2real/bringup.py",
        "--task",
        "Cyclo-Real-Showroom-Task000458-Random-FFW-SG2-v0",
        "--bridge",
        "ffw_sg2",
        "--headless",
        "--enable_cameras",
        "--ui-session",
        "--session-status-file",
        "/tmp/cyclo_lab_ui_session.json",
    ]


def test_simulator_exec_inspection_reports_running_and_safe_pid():
    client = SimpleNamespace(api=SimpleNamespace(
        exec_inspect=lambda exec_id: {"Running": exec_id == "running", "Pid": 84},
    ))

    assert simulator._exec_is_running(client, "running") is True
    assert simulator._exec_is_running(client, "stopped") is False
    assert simulator._exec_pid(client, "running") == 84


def test_simulator_training_gate_observes_s2r_and_standard_containers(monkeypatch):
    requested = []

    class FakeContainer:
        status = "running"

        def __init__(self, processes):
            self.processes = processes

        def reload(self):
            pass

        def top(self, ps_args):
            assert ps_args == "-eo args"
            return {"Processes": [[command] for command in self.processes]}

    containers = {
        "lerobot_server_s2r": FakeContainer(["python3 -m main_runtime"]),
        "lerobot_server": FakeContainer(["python3 -m lerobot.scripts.lerobot_train"]),
    }

    def get_container(name):
        requested.append(name)
        return containers[name]

    monkeypatch.setattr(
        simulator,
        "TRAINING_CONTAINERS",
        ("lerobot_server_s2r", "lerobot_server"),
    )
    client = SimpleNamespace(containers=SimpleNamespace(get=get_container))

    assert simulator._training_is_active(client) is True
    assert requested == ["lerobot_server_s2r", "lerobot_server"]


def test_simulator_start_request_rejects_environment_override():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        simulator.StartSimulationRequest(
            policy_path="/workspace/model/lerobot/run",
            environment="task_000458",
        )


def test_simulator_resolves_nearest_policy_contract(monkeypatch, tmp_path):
    root = tmp_path / "lerobot"
    selected = root / "run" / "checkpoints" / "040000" / "pretrained_model"
    selected.mkdir(parents=True)
    (root / "run" / "cyclo_policy.yaml").write_text(
        """simulation:
  environment: Cyclo-Real-Showroom-Task000458-FFW-SG2-v0
  randomized_environment: Cyclo-Real-Showroom-Task000458-Random-FFW-SG2-v0
  default_reset: randomized_evaluation
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(simulator, "POLICY_ROOT", root)

    assert simulator._resolve_policy_contract(str(selected)) == (
        "task_000458",
        "randomized_evaluation",
        str(selected),
    )


def test_simulator_rejects_manifest_gym_id_outside_catalog(monkeypatch, tmp_path):
    import pytest
    from fastapi import HTTPException

    root = tmp_path / "lerobot"
    selected = root / "run"
    selected.mkdir(parents=True)
    (selected / "cyclo_policy.yaml").write_text(
        """simulation:
  environment: Cyclo-Real-Showroom-Other-v0
  default_reset: deterministic
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(simulator, "POLICY_ROOT", root)

    with pytest.raises(HTTPException, match="unsupported simulation.environment"):
        simulator._resolve_policy_contract(str(selected))


def test_simulator_ready_requires_fresh_observation_and_camera(monkeypatch):
    task = "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0"
    document = {
        "state": "ready",
        "task": task,
        "pid": 42,
        "observation_sequence": 1,
        "camera_sequence": 0,
    }
    monkeypatch.setattr(simulator, "_read_status_document", lambda _container: document)
    monkeypatch.setattr(simulator, "_pid_is_alive", lambda _container, _pid: True)
    monkeypatch.setattr(simulator, "_exec_is_running", lambda _client, _exec_id: False)
    monkeypatch.setattr(simulator, "_training_is_active", lambda _client: False)

    status = simulator._public_status(SimpleNamespace(), SimpleNamespace())

    assert status["state"] == "starting"
    assert status["environment"] == "task_000525"
    assert status["message"] == "Waiting for fresh state and camera samples"


def test_simulator_surfaces_session_error(monkeypatch):
    document = {"state": "error", "error": "Isaac renderer failed", "pid": 42}
    monkeypatch.setattr(simulator, "_read_status_document", lambda _container: document)
    monkeypatch.setattr(simulator, "_pid_is_alive", lambda _container, _pid: True)
    monkeypatch.setattr(simulator, "_exec_is_running", lambda _client, _exec_id: True)
    monkeypatch.setattr(simulator, "_training_is_active", lambda _client: False)

    status = simulator._public_status(SimpleNamespace(), SimpleNamespace())

    assert status["state"] == "error"
    assert status["error"] == "Isaac renderer failed"
    assert status["message"] == "Isaac renderer failed"


def test_simulator_reset_publishes_only_fixed_ros_message(monkeypatch):
    calls = []
    fake_container = SimpleNamespace()
    fake_client = SimpleNamespace()
    monkeypatch.setattr(simulator, "_docker_client", lambda: fake_client)
    monkeypatch.setattr(simulator, "_running_container", lambda _client, _name: fake_container)
    monkeypatch.setattr(
        simulator,
        "_public_status",
        lambda _client, _container: {
            "state": "ready",
            "reset_count": 2,
            "observation_sequence": 10,
            "camera_sequence": 10,
        },
    )

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(simulator.subprocess, "run", fake_run)
    result = simulator.reset_simulation()

    assert calls[0][0] == [
        "ros2", "topic", "pub", "--once",
        "/simulation/reset", "std_msgs/msg/Empty", "{}",
    ]
    assert result["state"] == "resetting"


def test_simulator_stop_waits_for_pid_and_exec_before_clearing(monkeypatch):
    commands = []

    class FakeContainer:
        def exec_run(self, argv):
            commands.append(argv)
            return SimpleNamespace(exit_code=0)

    fake_container = FakeContainer()
    fake_client = SimpleNamespace()
    pid_alive = iter([True, True, False, False, False])
    exec_running = iter([True, False, False])
    monkeypatch.setattr(simulator, "_docker_client", lambda: fake_client)
    monkeypatch.setattr(simulator, "_running_container", lambda _client, _name: fake_container)
    monkeypatch.setattr(simulator, "_read_status_document", lambda _container: {"pid": 42})
    monkeypatch.setattr(simulator, "_pid_is_alive", lambda _container, _pid: next(pid_alive))
    monkeypatch.setattr(simulator, "_exec_is_running", lambda _client, _exec_id: next(exec_running))
    monkeypatch.setattr(simulator, "_training_is_active", lambda _client: False)
    monkeypatch.setattr(simulator.time, "sleep", lambda _seconds: None)
    simulator._SESSION.exec_id = "exec-1"
    simulator._SESSION.environment = "task_000458"
    simulator._SESSION.profile = "deterministic"

    result = simulator.stop_simulation()

    assert commands == [
        ["kill", "-TERM", "42"],
        ["rm", "-f", "/tmp/cyclo_lab_ui_session.json"],
    ]
    assert result["state"] == "stopped"
    assert simulator._SESSION.exec_id is None


def test_simulator_stop_uses_exec_pid_during_startup(monkeypatch):
    commands = []

    class FakeContainer:
        def exec_run(self, argv):
            commands.append(argv)
            return SimpleNamespace(exit_code=0)

    fake_container = FakeContainer()
    fake_client = SimpleNamespace()
    pid_alive = iter([False, True, False, False])
    exec_running = iter([False, False])
    monkeypatch.setattr(simulator, "_docker_client", lambda: fake_client)
    monkeypatch.setattr(simulator, "_running_container", lambda _client, _name: fake_container)
    monkeypatch.setattr(simulator, "_read_status_document", lambda _container: {})
    monkeypatch.setattr(simulator, "_pid_is_alive", lambda _container, _pid: next(pid_alive))
    monkeypatch.setattr(simulator, "_exec_pid", lambda _client, _exec_id: 84)
    monkeypatch.setattr(simulator, "_exec_is_running", lambda _client, _exec_id: next(exec_running))
    monkeypatch.setattr(simulator, "_training_is_active", lambda _client: False)
    monkeypatch.setattr(simulator.time, "sleep", lambda _seconds: None)
    simulator._SESSION.exec_id = "starting-exec"

    result = simulator.stop_simulation()

    assert commands[0] == ["kill", "-TERM", "84"]
    assert commands[-1] == ["rm", "-f", "/tmp/cyclo_lab_ui_session.json"]
    assert result["state"] == "stopped"


def test_navigation_grid_data_crc32_uses_only_map_data():
    first = {"info": {"width": 2}, "data": [-1, 0, 100, 0]}
    same_data = {"info": {"width": 4}, "data": [-1, 0, 100, 0]}
    changed = {"info": {"width": 2}, "data": [-1, 0, 99, 0]}

    marker = navigation_grid_cache.occupancy_grid_data_crc32(first)
    assert navigation_grid_cache.occupancy_grid_data_crc32(same_data) == marker
    assert navigation_grid_cache.occupancy_grid_data_crc32(changed) != marker


def test_navigation_grid_cache_serializes_only_changed_data():
    cache = navigation_grid_cache.OccupancyGridCache("/map")

    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 1]})
    marker, payload = cache.serialized_if_changed(None)
    assert json.loads(payload) == {
        "available": True,
        "data": {"info": {"width": 2}, "data": [0, 1]},
    }
    assert cache.serialized_if_changed(marker) == (marker, None)

    cache.cache_ros_message({"info": {"width": 99}, "data": [0, 1]})
    metadata_marker, metadata_payload = cache.serialized_if_changed(marker)
    assert metadata_marker != marker
    assert json.loads(metadata_payload)["data"]["info"]["width"] == 99

    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 2]})
    changed_marker, changed_payload = cache.serialized_if_changed(metadata_marker)
    assert changed_marker != metadata_marker
    assert json.loads(changed_payload)["data"]["data"] == [0, 2]


def test_navigation_grid_websocket_sends_cached_original_topic(monkeypatch):
    cache = navigation_grid_cache.OccupancyGridCache("/map")
    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 100]})
    monkeypatch.setitem(navigation_grid_cache.GRID_CACHES, "/map", cache)

    started = []
    monkeypatch.setattr(
        navigation,
        "ensure_ros_grid_subscriber_started",
        lambda: started.append(True),
    )

    class FakeWebSocket:
        def __init__(self):
            self.accepted = False
            self.messages = []

        async def accept(self):
            self.accepted = True

        async def send_text(self, payload):
            self.messages.append(json.loads(payload))

        async def receive(self):
            return {"type": "websocket.disconnect"}

    websocket = FakeWebSocket()
    asyncio.run(asyncio.wait_for(
        navigation.navigation_grid_websocket(websocket, "/map"),
        timeout=1.0,
    ))

    assert websocket.accepted is True
    assert started == [True]
    assert websocket.messages == [{
        "available": True,
        "data": {"info": {"width": 2}, "data": [0, 100]},
    }]


def test_navigation_ros_exec_environment_matches_server(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    assert navigation._ros_exec_environment() == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def test_navigation_goal_passes_ros_environment(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        captured["environment"] = environment
        return 0, "Goal accepted"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    result = navigation.send_goal(
        navigation.NavigateGoalRequest(
            pose={
                "header": {"frame_id": "map"},
                "pose": {
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "orientation": {
                        "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0,
                    },
                },
            }
        )
    )

    assert result.ok
    assert captured["command"][:4] == [
        "bash", "--noprofile", "--norc", "-c"
    ]
    assert captured["environment"] == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def _container_with_mounts(*destinations):
    return SimpleNamespace(
        attrs={
            "Mounts": [
                {"Destination": destination}
                for destination in destinations
            ]
        }
    )


def test_missing_required_mounts_reports_stale_groot_container():
    container = _container_with_mounts("/legacy_model_mount/groot")

    assert _missing_required_mounts("groot", container) == list(_GROOT_REQUIRED_MOUNTS)


def test_missing_required_mounts_accepts_current_groot_container():
    container = _container_with_mounts(*_GROOT_REQUIRED_MOUNTS)

    assert _missing_required_mounts("groot", container) == []


def test_missing_required_mounts_accepts_current_lerobot_container():
    container = _container_with_mounts(*_LEROBOT_REQUIRED_MOUNTS)

    assert _missing_required_mounts("lerobot", container) == []


def test_backend_container_image_mismatch_detects_old_container_image():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.4-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(attrs={"Image": "sha256:old"})
    spec = {"image": "robotis/groot-zenoh:1.3.4-arm64"}

    assert _backend_container_image_mismatch(
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
    )


def test_backend_container_image_mismatch_accepts_current_container_image():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.4-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(attrs={"Image": "sha256:new"})
    spec = {"image": "robotis/groot-zenoh:1.3.4-arm64"}

    assert not _backend_container_image_mismatch(
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
    )


def test_backend_container_stale_reason_detects_workspace_mount_mismatch():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.4-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(
        attrs={
            "Image": "sha256:new",
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": "/home/robot/old_workspace",
                },
                *[
                    {"Destination": destination}
                    for destination in _GROOT_REQUIRED_MOUNTS
                    if destination != "/workspace"
                ],
            ],
        }
    )
    spec = {"image": "robotis/groot-zenoh:1.3.4-arm64"}

    assert _backend_container_stale_reason(
        "groot",
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
        "/mnt/ssd/cyclo_intelligence/workspace",
    ) == "workspace_mount_mismatch"


def test_backend_container_stale_reason_accepts_repo_symlink_workspace_mount(
    monkeypatch,
    tmp_path,
):
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.4-arm64"
            return SimpleNamespace(id="sha256:new")

    host_repo = tmp_path / "host_repo"
    container_repo = tmp_path / "container_repo"
    ssd_workspace = tmp_path / "ssd" / "cyclo_intelligence" / "workspace"
    (host_repo / "docker").mkdir(parents=True)
    (container_repo / "docker").mkdir(parents=True)
    ssd_workspace.mkdir(parents=True)
    (container_repo / "docker" / "workspace").symlink_to(ssd_workspace)

    monkeypatch.setattr(app, "_HOST_PROJECT_DIR_CACHE", str(host_repo / "docker"))
    monkeypatch.setattr(app, "_CYCLO_REPO_MOUNT", str(container_repo))

    container = SimpleNamespace(
        attrs={
            "Image": "sha256:new",
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": str(host_repo / "docker" / "workspace"),
                },
                *[
                    {"Destination": destination}
                    for destination in _GROOT_REQUIRED_MOUNTS
                    if destination != "/workspace"
                ],
            ],
        }
    )
    spec = {"image": "robotis/groot-zenoh:1.3.4-arm64"}

    assert _backend_container_stale_reason(
        "groot",
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
        str(ssd_workspace),
    ) is None


def test_mount_source_for_destination_resolves_workspace_host_path():
    mounts = [
        {"Destination": "/root/ros2_ws/src/cyclo_intelligence", "Source": "/repo"},
        {"Destination": "/workspace", "Source": "/mnt/ssd/cyclo_intelligence/workspace"},
    ]

    assert _mount_source_for_destination(mounts, "/workspace") == (
        "/mnt/ssd/cyclo_intelligence/workspace"
    )


def test_host_workspace_dir_prefers_actual_mount_over_legacy_env(monkeypatch):
    container = SimpleNamespace(
        attrs={
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": "/repo/docker/workspace",
                }
            ]
        }
    )
    client = SimpleNamespace(
        containers=SimpleNamespace(get=lambda _name: container)
    )

    monkeypatch.setenv("HOSTNAME", "self")
    monkeypatch.setenv(
        "CYCLO_WORKSPACE_DIR",
        "/mnt/ssd/cyclo_intelligence/workspace",
    )
    monkeypatch.setattr(app, "_docker_client", lambda: client)
    app._HOST_WORKSPACE_DIR_CACHE = None
    try:
        assert _host_workspace_dir() == "/repo/docker/workspace"
    finally:
        app._HOST_WORKSPACE_DIR_CACHE = None


def test_resolve_groot_trt_paths_defaults_engine_inside_model():
    model, engine = _resolve_groot_trt_paths(
        "/workspace/model/groot/example",
        "",
    )

    assert model == "/workspace/model/groot/example"
    assert engine == "/workspace/model/groot/example/dit_model_bf16.trt"


def test_trt_status_reports_ready_engine(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"
    engine.write_bytes(b"engine")

    status = _trt_status(str(model), str(engine))

    assert status.status == "ready"
    assert status.engine_size_bytes == len(b"engine")


def test_trt_status_reports_missing_engine(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"

    status = _trt_status(str(model), str(engine))

    assert status.status == "missing"


def test_trt_status_reports_stale_oom_build_from_log(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"
    (model / "dit_model_bf16.trt.json").write_text(
        '{"status": "building", "started_at": 1.0, "updated_at": 2.0}'
    )
    (model / "dit_model_bf16.trt.build.log").write_text(
        "=== TensorRT build exited rc=137 at 2026-06-19 06:29:02 ===\n"
    )

    status = _trt_status(str(model), str(engine))

    assert status.status == "failed"
    assert status.returncode == 137
    assert "out-of-memory" in status.message


def test_compose_uses_repo_local_workspace_mounts():
    compose = (REPO_ROOT / "docker" / "docker-compose.yml").read_text()

    assert "CYCLO_WORKSPACE_DIR" not in compose
    assert "CYCLO_HUGGINGFACE_DIR" not in compose
    assert compose.count("./workspace:/workspace") == 3
    assert compose.count("./huggingface:/root/.cache/huggingface") == 3


def test_container_helper_does_not_export_workspace_mount_overrides():
    helper = (REPO_ROOT / "docker" / "container.sh").read_text()

    assert "export CYCLO_WORKSPACE_DIR" not in helper
    assert "export CYCLO_HUGGINGFACE_DIR" not in helper
    assert "CYCLO_SSD_ROOT" not in helper
    assert "CYCLO_STORAGE_MODE" not in helper
    assert "setup_storage" not in helper
    assert "prepare_host_mounts" in helper
    assert "rsync " not in helper
    assert "rsync -aHP" not in helper


def test_bt_node_is_known_user_service():
    _require_known_service("bt_node")


def test_bt_node_robot_type_file_is_written(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))

    _write_bt_robot_type("ffw_sg2_rev1")

    assert target.read_text() == "ffw_sg2_rev1\n"


def test_bt_node_robot_type_defaults_to_sg2():
    assert _validate_bt_robot_type("") == "ffw_sg2_rev1"


def test_bt_node_robot_type_rejects_other_robots():
    try:
        _validate_bt_robot_type("omy_f3m")
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("bt_node should reject unsupported robot types")


def test_bt_node_start_defaults_to_sg2(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(rc=0, stdout="started", stderr="")

    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))
    monkeypatch.setattr(app, "_run", fake_run)

    result = asyncio.run(app.service_start("bt_node"))

    assert result.ok is True
    assert target.read_text() == "ffw_sg2_rev1\n"
    assert calls == [("s6-rc", "-u", "change", "bt_node")]


def test_bt_node_start_rejects_other_robots(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(rc=0, stdout="started", stderr="")

    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))
    monkeypatch.setattr(app, "_run", fake_run)

    try:
        asyncio.run(app.service_start(
            "bt_node",
            app.ServiceActionRequest(robot_type="omy_f3m"),
        ))
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("bt_node should reject unsupported robot types")

    assert not target.exists()
    assert calls == []


def test_robot_type_validation_rejects_shell_metacharacters():
    try:
        _validate_robot_type("omy_f3m;echo bad")
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid robot_type should be rejected")


def test_unknown_user_service_is_rejected():
    try:
        _require_known_service("not_a_service")
    except app.HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("unknown service should be rejected")


def test_zenoh_router_is_not_user_managed_service():
    assert "zenoh_router" not in _USER_SERVICES


def test_groot_backend_uses_current_release_image():
    assert (
        _BACKENDS["groot"]["image"]
        == f"robotis/groot-zenoh:1.3.4-{app._BACKEND_ARCH}"
    )


def test_lerobot_backend_uses_s2r_container_namespace():
    assert _BACKENDS["lerobot"]["container"] == "lerobot_server_s2r"


def test_backend_status_model_exposes_stale_image_status():
    status = app.BackendStatus(
        name="groot",
        image="robotis/groot-zenoh:1.3.4-arm64",
        image_pulled=True,
        image_status="stale",
        container_state="exited",
        raw_state="stale_image",
    )

    assert status.image_status == "stale"


def test_host_project_dir_falls_back_to_compose_container_name(monkeypatch):
    class FakeContainers:
        def __init__(self):
            self.requested = []

        def get(self, name):
            self.requested.append(name)
            if name == "cyclo_intelligence_s2r":
                return SimpleNamespace(
                    attrs={
                        "Mounts": [
                            {
                                "Destination": app._CYCLO_REPO_MOUNT,
                                "Source": "/home/rc/workspace/cyclo_intelligence_s2r",
                            }
                        ]
                    }
                )
            raise NotFound(name)

    fake_containers = FakeContainers()
    fake_client = SimpleNamespace(containers=fake_containers)

    monkeypatch.setenv("HOSTNAME", "ubuntu")
    monkeypatch.setattr(app, "_docker_client", lambda: fake_client)
    app._HOST_PROJECT_DIR_CACHE = None

    try:
        assert (
            app._host_project_dir()
            == "/home/rc/workspace/cyclo_intelligence_s2r/docker"
        )
        assert fake_containers.requested == ["ubuntu", "cyclo_intelligence_s2r"]
    finally:
        app._HOST_PROJECT_DIR_CACHE = None


def test_compose_env_uses_current_container_mounts(monkeypatch):
    class FakeContainers:
        def __init__(self):
            self.requested = []

        def get(self, name):
            self.requested.append(name)
            if name != "cyclo_intelligence_s2r":
                raise NotFound(name)
            return SimpleNamespace(
                attrs={
                    "Mounts": [
                        {
                            "Destination": "/workspace",
                            "Source": "/mnt/ssd/cyclo_intelligence_s2r/workspace",
                        },
                        {
                            "Destination": "/root/.cache/huggingface",
                            "Source": "/mnt/ssd/cyclo_intelligence_s2r/huggingface",
                        },
                    ]
                }
            )

    fake_containers = FakeContainers()
    fake_client = SimpleNamespace(containers=fake_containers)

    monkeypatch.setenv("HOSTNAME", "container-id")
    monkeypatch.delenv("CYCLO_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("CYCLO_HUGGINGFACE_DIR", raising=False)
    monkeypatch.setattr(app, "_docker_client", lambda: fake_client)
    app._HOST_WORKSPACE_DIR_CACHE = None
    app._HOST_HUGGINGFACE_DIR_CACHE = None

    try:
        env = _compose_env()
        assert (
            env["CYCLO_WORKSPACE_DIR"]
            == "/mnt/ssd/cyclo_intelligence_s2r/workspace"
        )
        assert (
            env["CYCLO_HUGGINGFACE_DIR"]
            == "/mnt/ssd/cyclo_intelligence_s2r/huggingface"
        )
        assert env["ARCH"] == app._BACKEND_ARCH
        assert env["CYCLO_COMPOSE_PROJECT_NAME"] == "cyclo_intelligence_s2r"
        assert env["CYCLO_MAIN_CONTAINER_NAME"] == "cyclo_intelligence_s2r"
        assert env["LEROBOT_CONTAINER_NAME"] == "lerobot_server_s2r"
        assert env["LEROBOT_RUNTIME_NAMESPACE"] == "lerobot_s2r"
        assert fake_containers.requested == [
            "container-id",
            "cyclo_intelligence_s2r",
            "container-id",
            "cyclo_intelligence_s2r",
        ]
    finally:
        app._HOST_WORKSPACE_DIR_CACHE = None
        app._HOST_HUGGINGFACE_DIR_CACHE = None
