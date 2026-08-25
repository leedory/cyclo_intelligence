# Task_000459 smoke 데이터 취득 및 LeRobot 변환

## 목적과 범위

`Task_000459_Pick_Peanut_Mix_WhiteShelf_GENERATED_SIM`은 ACT 파이프라인의
입력, 변환, 학습 실행을 빠르게 확인하기 위한 **smoke/overfit용 simulation
데이터**다. 실제 로봇 배포 성능이나 sim-to-real 일반화 성능을 검증하는
데이터셋이 아니다.

- 작업 instruction: `Pick up the Peanut Mix with right gripper.`
- native 원본: `/home/robotis-ai/cyclo_lab/data/generated/Task_000459_Pick_Peanut_Mix_WhiteShelf_GENERATED_SIM`
- stage 경로: `docker/workspace/native_sources/Task_000459_Pick_Peanut_Mix_WhiteShelf_GENERATED_SIM`
- LeRobot v3 목표 경로: `docker/workspace/dataset/robotis/task_000459_peanut_mix_generated_smoke_act_v30`
- 실행 스크립트: `tools/act_sim_real/run_generated_smoke.sh`
- 변환기: `tools/act_sim_real/convert_native_peanut_smoke.py`

"smoke"라는 이름은 데이터 품질이 낮다는 뜻이 아니라, 같은 검증 궤적을
고정된 조건에서 재생한 소규모 데이터로 코드 경로를 검증한다는 뜻이다.

## 어떻게 취득했는가

생성기는 Cyclo Lab의
`scripts/sim2real/imitation_learning/generate_peanut_pick_dataset.py`다.
원본은 ROS MCAP이 아니라, 이전에 저장한 검증된 teleoperation capture의
`trajectory_chunks/chunk_*.npz`이다.

1. 생성기는 capture에서 `sim_action`, 시간, 로봇/대상 상태를 읽는다.
2. 오른팔의 home 위치 이탈과 gripper 닫힘을 찾아 첫 번째 peanut-pick
   구간(approach, close, extraction, hold)을 찾는다.
3. 이 구간을 phase별로 재표본화한다. 시작 안정화 30 step, approach 105,
   pre-grasp 안정화 8, close 12, close dwell 10, extraction 85, hold 30 step으로
   구성되어 각 episode는 280 control step, 15 Hz, 약 18.67초다.
4. Isaac Lab/Cyclo Lab showroom 환경을 매 episode마다 `reset()`한 뒤, 위 action을
   재생한다. 카메라는 `cam_head`, `cam_wrist_left`, `cam_wrist_right`를 모두
   켜고 매 control step JPEG(RGB, quality 90)를 저장한다.
5. 각 step에서 simulator의 SG2 joint position, canonical action, peanut packet root state, 오른팔 EEF body state 및 timestamp를 `trajectory.npz`에 저장한다. 이때 base `linear_x`, `linear_y`, `angular_z` state는 simulator에서 읽지 않고 `0, 0, 0`을 붙인다.
6. 마지막에 다음 조건을 모두 만족하는 episode만 성공으로 표기한다.

   - 최종 gripper command >= 0.9
   - 최종 packet 수평/3D displacement >= 0.15 m
   - 최대 lift >= 0.015 m
   - 최종 packet z >= 0.5 m

이 archive의 manifest는 12개의 독립 reset/replay episode가 모두 성공했다고
기록한다. 다만 action의 공간 변형은 넣지 않았다. 현재 생성 설정은 packet-0의
검증된 pick 궤적을 사용하며 robot root pose, 대상 pose, 벽 배경은 고정이다.
코드의 phase-retiming 함수도 현재 `scale = 1.0`이므로 실제로는 timing variation도
추가하지 않는다. 따라서 12개 episode는 서로 다른 배치 조건의 demonstration이
아니라 같은 성공 motion의 독립 simulation replay다.

## native archive 형식

각 `episodes/episode_XXXXXX`에는 다음이 있다.

| 파일/디렉터리 | 내용 |
|---|---|
| `trajectory.npz` | 15 Hz, 280 step의 named state/action 및 부가 상태 |
| `frames/<camera>/frame_XXXXXX.jpg` | head/left wrist/right wrist RGB JPEG |
| `metrics.json` | 성공 여부, 대상 이동/상승량, phase index |

`trajectory.npz`의 주 feature는 다음과 같다.

- `observation_state`: `[280, 22]` — 좌/우 arm 7축과 gripper, head 2축, lift는 `robot.data.joint_pos`에서 `env.step()` **후** 읽은 simulator 값이다. 마지막 base `linear_x`, `linear_y`, `angular_z` 3개는 현재 생성 코드가 넣는 고정 영벡터다.
- `action`: `[280, 22]` — 위와 같은 canonical recorder 순서의 action
- `packet_root_state_w`, `eef_body_state_w`: 각각 `[280, 13]`; 성공 검증 및
  분석용이며 ACT 입력으로 직접 변환하지 않는다.
- `timestamps_s`: 15 Hz timestamp
- `observation_state_names`, `action_names`: 위 vector 순서를 보존하는 이름 배열

이 형식은 `cyclo_lab_native_simulation_dataset/v1`이다. **MCAP도, 원래부터
LeRobot 형식인 데이터도 아니다.**
### Leader 없이 action이 생긴 이유

이번 generated smoke 생성 과정에서는 leader를 연결하거나 사람의 입력을 실시간으로 읽지 않는다. 생성기는 `--capture-dir/trajectory_chunks/chunk_*.npz`에서 이미 저장되어 있던 `sim_action`을 읽고, 그 중 packet-0의 첫 pick 구간을 잘라 simulator에 반복 적용한다. 이후 `sim_to_canonical_action()`이 Cyclo Lab의 action 순서를 recorder 22-D 순서로 재배열하고, LeRobot 변환 단계가 그 중 오른팔/lift/base의 12-D만 선택한다.

따라서 LeRobot `action`은 **현재 smoke run에서 새로 취득한 leader action이 아니라, 기존 teleop capture에서 유래한 replay command**다. 현재 배포된 generated archive의 manifest에는 원래 `--capture-dir` 경로와 그 capture를 어떤 장치/입력 방식으로 만든 것인지는 남아 있지 않다. 이 archive만으로는 그 선행 capture가 physical leader, keyboard, 다른 SDK 중 무엇으로 조작되었는지 단정할 수 없다.


## LeRobot v3로 어떻게 변환하는가

다음 명령은 원본을 workspace에 snapshot하고, 컨테이너 안에서 입력을 검증한 뒤
LeRobot v3 dataset을 생성한다. `all`은 GPU 학습을 시작하지 않는다.

```bash
cd /home/robotis-ai/cyclo_intelligence_s2r
docker/container.sh start-lerobot
tools/act_sim_real/run_generated_smoke.sh all
```

변환 흐름은 아래와 같다.

1. `stage-source`가 native archive를 `rsync`로 `docker/workspace/native_sources/...`에
   복사하고 변환기를 workspace에 설치한다. 이는 실행 중인 `lerobot_server`가
   `/workspace` mount를 통해 동일한 입력을 보도록 하기 위한 단계다.
2. `validate`는 manifest format과 task id (`000459`), episode count를 확인한다.
   모든 `metrics.json.success`가 `true`인지, state name 순서가 episode마다 같은지,
   timestamp가 엄격히 증가하는지, 모든 선택 frame이 존재하는지 검증한다.
3. 15 Hz source timestamp를 10 Hz grid에 가장 가까운 source index로 내려샘플링한다.
   index가 중복되거나 역순이면 실패시킨다. 280 step/18.67초 source는 episode당
   187개의 10 Hz policy frame이 된다.
4. camera key를 기존 ACT camera schema에 맞춘다.

   | Native camera | LeRobot key | 처리 |
   |---|---|---|
   | `cam_head` | `observation.images.cam_left_head` | rotation 없음 |
   | `cam_wrist_left` | `observation.images.cam_left_wrist` | 시계 방향 270도 회전 |
   | `cam_wrist_right` | `observation.images.cam_right_wrist` | 시계 방향 270도 회전 |

   모든 이미지는 RGB CHW `[3, 480, 640]`로 resize된다. `LeRobotDataset`이 이를
   AV1 video로 인코딩한다.
5. LeRobot feature는 state 22차원, action 12차원, camera 3개로 생성된다.
   action에는 아래 오른팔/이동 base command만 순서대로 남긴다.

   ```text
   arm_r_joint1..7, gripper_r_joint1, lift_joint,
   linear_x, linear_y, angular_z
   ```

6. episode별로 `dataset.add_frame()` 후 `save_episode()`를 호출하고, 끝에서
   `finalize()`로 parquet/video/meta/statistics를 완성한다. 정상 완료 시
   `meta/cyclo_source_manifest.json`도 source frame 수와 policy frame 수를 기록한다.

변환기는 output 디렉터리가 비어 있지 않으면 overwrite하지 않는다. 따라서 실패한
부분 dataset을 다시 변환할 때는 기존 output을 보존/이동하거나 새 `DATASET_ROOT`를
지정해야 한다. 덮어쓰기로 원본과 정상 dataset을 잃지 않도록 한 안전장치다.

## 현재 확인한 변환 상태

native archive와 stage copy에는 12 episode가 있다. 그러나 현재
`task_000459_peanut_mix_generated_smoke_act_v30/meta/info.json`은 **7 episode,
1,309 frame**이고 `meta/cyclo_source_manifest.json`은 아직 없다. 이는 12개를
끝까지 변환하고 `finalize()`/manifest 기록까지 완료한 결과가 아니다.

따라서 이 출력은 학습용으로 확정하지 말고 변환 완료를 먼저 확인해야 한다. 정상
완료 기준은 `total_episodes == 12`, 약 `2,244` policy frame(12 x 187), 그리고
`meta/cyclo_source_manifest.json`의 존재다. 변환기는 non-empty output을 재사용하지
않으므로, 재실행 전에 실패한 7-episode output을 별도 보관하거나 새 output 경로를
지정해야 한다.

## 기존 Task_000458 teleop 데이터와의 차이

기존 ACT 데이터는 `Task_000458_Pick_Peanut_Mix_WhiteShelf_SimtoReal_*_MCAP`의
simulation MCAP와 real-robot MCAP을 사용하는 파이프라인이다.
`convert_sim_real.py`가 repository의 `RosbagToLerobotV30Converter`를 호출해 ROS
topic, 기록 timestamp, recording video를 정렬/변환한다. real 쪽은 video가 완전한
episode 2--13만 사용하며, sim 12개와 real 12개를 합친 경우 총 24 episode다.

| 항목 | Task_000459 generated smoke | Task_000458 teleop sim/real |
|---|---|---|
| 원천 형식 | native Cyclo Lab archive (`npz` + JPEG + JSON) | ROS MCAP + metadata + recording MP4 |
| 행동의 출처 | 검증된 packet-0 teleop pick을 simulator에서 재생 | 실제 teleoperation/recording 시 수집된 ROS stream |
| 물리 실행 | 매 episode Isaac Lab reset 후 action replay | sim recording 또는 실제 로봇의 기록 |
| 성공 선택 | simulator packet displacement/lift/gripper 조건으로 생성 시 검증 | 녹화 episode와 카메라/토픽 완전성 기준으로 변환 |
| 환경 다양성 | 없음: 대상/root/background 고정, 같은 motion replay | 녹화별 시간·제어·관측 variation, real에는 실제 센서/로봇 효과 포함 |
| source rate | 15 Hz native step | ROS/recording timestamp; converter가 10 Hz로 동기화 |
| 이미지 원천 | simulator RGB JPEG | recording video(MP4) |
| LeRobot camera | 3개, 480x640, 키를 common schema로 rename/rotate | 같은 3개 common camera, 480x640; sim-only `cam_external` 제외 |
| LeRobot state | **22-D** (left/right arm, gripper, head, lift, base) | **12-D** (right arm/gripper, lift, base) |
| LeRobot action | 12-D right arm/gripper/lift/base | 동일한 12-D right arm/gripper/lift/base |
| 합산 가능성 | 기존 000458 dataset과 그대로 합칠 수 없음 | sim-only, real-only, sim+real 세 변형을 같은 12-D schema로 학습 가능 |

두 파이프라인은 최종 action key와 image key/크기는 맞지만, observation state가
`22` 대 `12`로 다르다. ACT는 dataset metadata에서 input feature shape를 읽으므로
두 dataset을 단순 병합하면 state feature 불일치로 실패하거나 잘못된 입력 schema를
만들게 된다. 병합 또는 checkpoint fine-tuning을 하려면 먼저 한쪽을 공통 schema로
변환해야 한다. 일반적으로 smoke state에서 left arm, left gripper, head를 제거해
12-D로 맞추거나, 반대로 Task_000458 converter를 22-D state로 일관되게 재생성하는
명시적 변환 작업이 필요하다.

또한 smoke dataset은 고정 replay라 training loss를 빠르게 낮추는지 확인하는 데는
유용하지만, 데이터 다양성 때문에 발생하는 일반화 성능을 판단할 근거가 될 수 없다.
배포용 ACT는 기존 real teleop 데이터와 추가로 수집한 다양한 성공 real demonstration을
중심으로 검증해야 한다.

## 학습 실행 전 확인

정상 완료된 smoke dataset을 대상으로만 아래처럼 학습을 시작한다.

```bash
cd /home/robotis-ai/cyclo_intelligence_s2r
tools/act_sim_real/run_generated_smoke.sh train
```

기본 설정은 ACT, CUDA, batch 16, 50,000 step, `chunk_size=45`,
`n_action_steps=45`, 10 Hz, 480x640 camera 3개다. 밝기 및 작은 affine 증강도
활성화되어 있다. 이 설정은 smoke 파이프라인의 실행 검증용 기준이며, 12개의 거의
동일한 replay로 실제 배포 하이퍼파라미터를 결정해서는 안 된다.
