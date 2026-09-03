# LeRobot 학습·배포 I/O 계약 가이드

이 문서는 Cyclo 로봇에서 LeRobot policy를 학습할 때 데이터셋, 학습
config, checkpoint가 동일한 I/O 계약을 갖도록 하기 위한 기준이다. 모델
config의 차원만으로는 관절 이름과 순서를 복원할 수 없으므로, 학습 시점에
명시적인 layout을 보존해야 한다.

## 원칙

- 로봇 공통 YAML은 **하드웨어의 기본값**만 가진다.
- task 또는 policy마다 다른 관절·카메라 처리는 policy별
  `cyclo_policy_io.json`에만 기록한다.
- `observation.state`와 `action`은 각각 이름, 순서, 차원을 갖는 계약이다.
  차원만 같다고 호환되는 것은 아니다.
- 학습 시 적용한 이미지 회전·resize는 inference에서도 동일해야 한다.

## Peanut 19D 기준 레이아웃

`task_000458_peanut02_seed_v2_original_act` 및
`task_000458_peanut02_seed_v2_original_plus_augment_random_16x_act`의
state/action 레이아웃은 다음과 같다.

| Index | Modality | Joint names |
| --- | --- | --- |
| 0–7 | `arm_left` | `arm_l_joint1` … `arm_l_joint7`, `gripper_l_joint1` |
| 8–15 | `arm_right` | `arm_r_joint1` … `arm_r_joint7`, `gripper_r_joint1` |
| 16–17 | `head` | `head_joint1`, `head_joint2` |
| 18 | `lift` | `lift_joint` |

따라서 policy config에는 아래와 같이 state/action 모두 19D로 기록한다.

```json
{
  "input_features": {
    "observation.state": {"type": "STATE", "shape": [19]},
    "observation.images.rgb.cam_left_head": {"type": "VISUAL", "shape": [3, 480, 640]},
    "observation.images.rgb.cam_left_wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
    "observation.images.rgb.cam_right_wrist": {"type": "VISUAL", "shape": [3, 480, 640]}
  },
  "output_features": {
    "action": {"type": "ACTION", "shape": [19]}
  }
}
```

`mobile` (`linear_x`, `linear_y`, `angular_z`)는 이 19D policy에 포함하지
않는다. mobile을 제어할 policy를 새로 학습할 경우에는, mobile의 포함 여부와
순서를 데이터셋부터 다시 정의하고 그 차원으로 재학습해야 한다. 기존 19D
checkpoint에 mobile을 padding, truncation, 또는 끼워 넣어서는 안 된다.

## 데이터 변환 단계

1. converter가 `observation.state`와 `action`을 위 순서로 생성하도록
   `joint_names`를 명시한다. 이름은 LeRobot dataset metadata의 feature names에
   함께 저장한다.
2. 첫 episode를 변환한 직후 metadata를 검사한다. state/action의 names와
   길이가 각각 19인지 확인한다.
3. state와 action의 이름이 다르거나 차원이 다르면 의도적으로 다른 경우만
   허용하고, policy config도 각각의 실제 차원으로 만든다.
4. 값이 고정된 관절(예: 녹화 중 움직이지 않은 head)은 제거하지 않는다.
   제거하려면 state와 action, policy config, deployment I/O mapping을 모두
   변경한 뒤 재학습한다.

## 카메라 전처리 단계

Peanut 19D policy는 다음 세 입력만 사용한다.

| Dataset/policy key | Robot camera | Rotation | Final shape |
| --- | --- | --- | --- |
| `observation.images.rgb.cam_left_head` | `cam_left_head` | 0° | `3×480×640` |
| `observation.images.rgb.cam_left_wrist` | `cam_left_wrist` | 270° | `3×480×640` |
| `observation.images.rgb.cam_right_wrist` | `cam_right_wrist` | 270° | `3×480×640` |

변환기는 rotation 후 resize를 수행해야 하며, inference도 같은 순서를
따라야 한다. 새 policy에서 wrist 방향이 다르면 공통 robot YAML을 바꾸지 말고
그 policy의 `camera_rotation_deg`만 변경한다.

## Checkpoint export

학습 저장 단계에서 각 `checkpoints/<step>/pretrained_model/`에
`cyclo_policy_io.json`을 함께 저장한다. 이 파일은 policy가 동작하는 데 필요한
관절/카메라 의미 정보를 보존하는 배포 artifact다.

Peanut 19D의 최소 예시는 다음과 같다.

```json
{
  "format_version": 1,
  "camera_rotation_deg": {
    "cam_left_wrist": 270,
    "cam_right_wrist": 270
  },
  "observation_state_modalities": ["arm_left", "arm_right", "head", "lift"],
  "observation_state_joint_names": [
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
    "head_joint1", "head_joint2", "lift_joint"
  ],
  "action_modalities": ["arm_left", "arm_right", "head", "lift"],
  "action_joint_names": [
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
    "head_joint1", "head_joint2", "lift_joint"
  ]
}
```

현재 engine은 checkpoint sidecar가 없을 때에만 model-name registry를 fallback으로
사용한다. registry는 임시 호환 수단이므로, 새 학습 결과에는 sidecar를 포함하는
방식을 표준으로 삼는다.

## 학습 전·후 검증 gate

학습을 시작하기 전과 checkpoint를 배포하기 전에 다음을 자동 검사한다.

- dataset metadata의 state/action names가 policy I/O JSON의 names와 정확히 같은가
- name 개수와 `config.json`의 state/action shape가 같은가
- policy가 요구하는 모든 image key가 dataset에 존재하는가
- 각 image의 rotation 후 shape가 policy shape와 같은가
- `cyclo_policy_io.json`이 모든 배포 checkpoint에 존재하고 JSON 문법이 유효한가
- runtime이 정책별 mapping을 적용한 결과가 padding/truncation 없이 policy 차원과 같은가

이 gate를 통과하지 못한 checkpoint는 inference 목록에 노출하지 않는 것을 권장한다.

## 이번 두 policy의 임시 배포 설정 제거

현재 두 Peanut policy는 checkpoint 내부 sidecar 대신 engine registry에 개별 매핑이
있다. 제거하려면 아래 두 파일만 삭제하면 되며, 다른 policy와 공통 robot 설정에는
영향이 없다.

- `lerobot_engine/model_io_mappings/task_000458_peanut02_seed_v2_original_act.json`
- `lerobot_engine/model_io_mappings/task_000458_peanut02_seed_v2_original_plus_augment_random_16x_act.json`
