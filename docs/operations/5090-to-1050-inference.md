# 5090 학습 결과를 1050 로봇에서 추론하는 흐름

이 문서는 두 종류의 연결을 구분한다.

1. **모델 전달:** 5090에서 생성한 checkpoint를 SSH/rsync로 1050의
   `/workspace` mount 아래에 배치한다.
2. **실시간 제어:** policy runtime과 로봇의 ROS 2 graph를 Zenoh로 연결한다.

모델이 1050에 있다는 사실만으로 실시간 연결이 생기지 않으며, 반대로 Zenoh
graph가 연결돼도 checkpoint가 올바른 계약으로 finalize되지 않았다면 LOAD하면
안 된다.

## 저장소가 보장하는 계약

5090에서는 `tools/training/act_recipe.py`가 recipe와 LeRobot dataset metadata를
검증하고 학습 argv를 만든다. 학습 중 다음 두 파일을 run root에 기록하며, 완료된
각 `checkpoints/<step>/pretrained_model/`에도 `cyclo_policy.yaml`을 기록한다.

- `resolved_recipe.yaml`: 재현용 전체 학습 설정
- `cyclo_policy.yaml`: 배포에 필요한 task, robot type, 15 Hz, state/action feature
  순서, inactive action group, camera와 simulator 계약

Task 000002의 canonical action/state는 양팔·양그리퍼 16개 뒤에 모바일 베이스
전체 3축이 오는 19D이다.

```text
[arm_left 8, arm_right 8, linear_x, linear_y, angular_z]
```

`linear_y`만 학습한 17D checkpoint를 `[0, y, 0]`으로 확장하는 호환 경로는
지원하지 않는다. 기존 17D Task 000002 checkpoint는 재학습 대상이며 1050에
배포하면 안 된다.

## 1. 5090에서 생성하고 확인하기

canonical recipe는 task당 하나다.

```bash
python3 tools/training/act_recipe.py validate training/task_000002/act.yaml
python3 tools/training/act_recipe.py plan training/task_000002/act.yaml
python3 tools/training/act_recipe.py run training/task_000002/act.yaml
```

기본 Docker 실행에서 recipe의 `/workspace/...`는 5090 checkout의
`docker/workspace/...`와 대응한다. Task 000002의 기본 output은 다음과 같다.

```text
5090 host: docker/workspace/model/lerobot/task_000002_coffee_transport_headcam_full_base_act
container: /workspace/model/lerobot/task_000002_coffee_transport_headcam_full_base_act
```

전달할 단위는 숫자 step 디렉터리 안의 `pretrained_model/` 전체다. 최소한 model
weights, `config.json`, `train_config.json`, `cyclo_policy.yaml`이 모두 있어야
한다. `checkpoints/last` symlink만 단독 복사하지 않는다.

## 2. 1050에 받기

2026-09-11 read-only 확인 기준 1050의 checkout은
`/home/robotis/cyclo_intelligence`이고, Compose는
`docker/workspace:/workspace`를 mount한다. `docker/workspace`는 실제로
`/mnt/ssd/cyclo_intelligence/workspace`를 가리키는 symlink다. 따라서 모델의
host-side 목적지는 아래 계층이다.

```text
/home/robotis/cyclo_intelligence/docker/workspace/model/lerobot/<run>/...
```

저장소에는 5090→1050 model upload 자동화가 없다. 전송은 별도 운영 단계이며,
먼저 `rsync --dry-run`으로 source와 destination을 확인한 뒤 승인된 checkpoint만
복사해야 한다. 전송 후 1050에서 파일 수와 checksum을 비교하고
`cyclo_policy.yaml`의 `robot`, `state.names`, `action.names`, `action.inactive`를
확인한다.

1050의 LeRobot service는 같은 mount를 `/workspace`로 보기 때문에 LOAD 요청의
`model_path`는 host 경로가 아니라 다음과 같은 container 경로여야 한다.

```text
/workspace/model/lerobot/<run>/checkpoints/<step>/pretrained_model
```

## 3. 1050 runtime이 하는 일

policy container는 공통 two-process runtime을 사용한다.

```text
/<backend>/inference_command
        │
        ▼
Main Runtime ── /<backend>/engine_command ──▶ Engine Process
     │                                      model load + inference
     └── action buffer + RobotClient ──────▶ configured command topics
```

- `LOAD`는 model과 `cyclo_policy.yaml`을 읽고 완전한 robot action group 순서를
  반환한다.
- Engine은 observation을 읽고 `(T, D)` action chunk를 반환할 뿐 로봇 command를
  직접 publish하지 않는다.
- Main Runtime이 buffer, 100 Hz control loop와 RobotClient publish를 소유한다.
- `publish_to_robot=false`는 preview만 허용한다.
- `publish_to_robot=true`는 실제 command topic publish를 허용하므로, 계약과
  네트워크 검증 전에는 사용하지 않는다.

## 4. Zenoh 연결과 현재 확인 결과

양쪽 Compose는 `network_mode: host`를 사용하며 runtime 기본값은
`ROS_DOMAIN_ID=30`, `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, port 7447이다. Dockerfile의
원격 router 설정은 예시일 뿐 이미지 기본값에서는 활성화되지 않는다.

2026-09-11에 실행 중인 상태를 read-only로 확인한 결과:

- 5090 (`robotis-ai-omen`, `192.168.6.102`)은 `*:7447`에서 router가 listen하며
  로컬 graph에서 `/lerobot_s2r/inference_command`와
  `/lerobot_s2r/engine_command`가 보였다.
- 1050 (`ffw-SNPR48A1050`, `192.168.6.2`)은 `127.0.0.1:7447`에서 router가
  listen했다.
- 1050의 graph에서는 `/get_robot_info`, `/get_robot_types`, `/set_robot_type`,
  `/task/command`가 보였지만 5090의 `/lerobot_s2r/*` service는 보이지 않았다.

즉, **현재 두 Zenoh graph가 연결됐다는 증거는 없다.** 물리 로봇 publish 테스트
전에 router topology를 하나로 결정해야 한다. 가능한 방향은 다음 둘이다.

- 1050 runtime을 5090의 외부 listen router로 연결한다.
- 1050 router를 외부에서 접근 가능하게 구성한 뒤 5090 policy runtime을 1050에
  연결한다.

어느 쪽이든 IP를 소스 코드에 넣지 말고 deployment 환경의
`ZENOH_CONFIG_OVERRIDE`로 설정한다. 방화벽·router listen 범위·로봇이 네트워크
단절 시 안전 정지하는 동작까지 함께 확인한 후 선택해야 한다.

## Read-only 점검 순서

다음 결과가 양쪽에서 일치하기 전에는 `publish_to_robot=true`로 전환하지 않는다.

1. 1050에 복사된 checkpoint와 5090 source의 checksum이 일치한다.
2. Task 000002 manifest가 full-base 19D이고 runtime LOAD가 action keys
   `[arm_left, arm_right, mobile]`을 반환한다.
3. 양쪽 `ros2 service list`에서 선택한 backend service가 보인다.
4. 1050의 `/get_robot_info`와 robot config가 `ffw_sg2_rev1` 및 완전한 mobile
   group `[linear_x, linear_y, angular_z]`를 보고한다.
5. 먼저 `publish_to_robot=false`로 LOAD/START/STOP/UNLOAD lifecycle과 camera/state
   readiness를 확인한다.
6. 실제 publish는 별도의 로봇 안전 절차와 승인 아래 짧은 정지 자세 시험부터
   수행한다.

1050 checkout에는 현재 수정 파일과 `.pre_*` 백업이 남아 있다. 이 문서는 그
미커밋 파일을 배포 기준으로 간주하지 않는다. 먼저 `leedory/cyclo_intelligence`
의 정리된 branch와 1050 전용 변경을 별도 review한 뒤 적용해야 한다.
