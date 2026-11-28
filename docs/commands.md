# dxconv 명령어 정리

모든 명령은 `dxconv <명령> [옵션]` 형식. 옵션 전체는 `dxconv <명령> --help`.

| 명령 | 하는 일 | 주로 쓰는 때 |
|---|---|---|
| `convert` | pt → ONNX → dxnn 한 번에 | **평소 변환** |
| `sweep` | 설정 조합을 여러 번 변환·평가해 비교 (실험용) | 추천값 재검증 (기본 사용에는 불필요) |
| `env` | DX-COM / DX-RT 설치 상태 확인 | 설치 직후, 문제 생겼을 때 |
| `remote setup` / `remote check` | NPU 장비 연결 준비 / 점검 | 장비를 처음 연결할 때 |
| `dataset split` / `dataset check` | 데이터 분할 미리 보기 / 데이터셋 점검 | 변환 전에 데이터 확인 |
| `baseline` | PC에서 PT / FP32 ONNX 정확도 측정 | NPU 결과와 비교할 기준값이 필요할 때 |
| `recommend show` / `recommend update` | 추천값 확인 / `sweep` 결과로 추천값 갱신 | 추천값 관리 |
| `export` / `compile` | 변환을 단계별로 따로 실행 | 디버깅, ONNX가 이미 있을 때 |

공통 옵션

| 옵션 | 뜻 |
|---|---|
| `--debug` | 오류 시 스택트레이스 출력 (`dxconv --debug convert ...`처럼 명령 앞에) |
| `--out <폴더>` | 결과를 저장할 폴더 (기본 `runs`) |
| `--model <이름>` | 결과 폴더·프리셋 이름. 생략하면 pt 파일 이름 |
| `--config <yaml>` | 설정 파일로 여러 값을 한 번에 지정 |
| `--set 키=값` | 설정 하나 바꾸기 (여러 번 사용 가능). 예: `--set compile.calibration_num=200` |

설정 우선순위: **명령행 옵션 > `--set` > `--config` > 모델 프리셋 > 추천값 > 기본값**

---

## 1. `convert`

```bash
dxconv convert --pt <pt 파일> [입력 옵션] [장비 옵션] [설정 옵션]
```

### 입력 옵션 (모두 선택)

| 옵션 | 뜻 | 없을 때 |
|---|---|---|
| `--images <폴더>` | 이미지 폴더. 캘리브레이션 / 튜닝 / 평가로 자동 분할 | 기본 사진(coco128)으로 캘리브레이션 |
| `--labels <폴더>` | YOLO 라벨 폴더 (`<이미지 이름>.txt`). `--images`와 함께 | 정확도 검증 생략 (FPS만) |
| `--calib-dir <폴더>` | 캘리브레이션에만 쓸 사진 폴더 (분할 없이 그대로) | — |
| `--dataset <yaml>` | 분할 목록을 직접 정한 데이터셋 설정 | — |

`--calib-dir`는 `--images` / `--dataset`과 함께 쓸 수 없음

### 장비 옵션 (포맷 결정, 하나만)

| 옵션 | 뜻 |
|---|---|
| `--remote <장비 이름>` | 장비에 접속해 DX-RT 버전을 읽고 포맷 결정 + 변환 후 장비에서 검증 |
| `--target-dxrt <버전>` | 배포 장비 DX-RT 버전만 지정 (예: `3.2.0` → v8). 검증 없음 |
| `--dxnn-format <9\|8>` | 포맷 직접 지정 |
| `--dxcom-version <버전>` | DX-COM 버전 직접 지정 (예: `2.4.1`) |

아무것도 안 주면 설정 파일의 DX-COM 버전(기본 2.5.0, v9)

### 설정 옵션 (보통 안 바꿔도 됨)

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--imgsz <크기>` | 640 | 입력 해상도 (32의 배수). 낮추면 빨라지고 정확도 하락 |
| `--opset <번호>` | 17 | ONNX opset. DX-COM 지원 11~21, 권장 17. 실측: 11~21 모두 NPU 데이터 동일 |
| `--calib-method <방식>` | 이미지 있으면 minmax, 없으면 ema | 캘리브레이션 방식 (`minmax` / `ema` / `iqr`) |
| `--calib-num <장수>` | 이미지 있으면 50, 없으면 100 | 캘리브레이션에 쓸 사진 수 |
| `--opt-level <0\|1>` | 1 | DX-COM 최적화 수준. 1이 약 20% 빠름 |
| `--aggressive-partitioning` / `--no-aggressive-partitioning` | 끔 | 연산을 최대한 NPU에 올림 (YOLO26에서는 끄는 쪽이 빠름) |
| `--export-html` | 끔 | DX-COM 컴파일 요약 HTML을 run 폴더에 저장 |
| `--timeout <초>` | 없음 | 컴파일 제한 시간 |
| `--dry-run` | — | 컴파일 없이 설정과 DX-COM config만 만들어 확인 |

### 예시

```bash
# pt만: 기본 사진으로 변환 + 장비 FPS
dxconv convert --pt fire.pt --remote npu1

# 이미지 + 라벨: 내 데이터로 캘리브레이션 + PT 대비 정확도·FPS 검증
dxconv convert --pt fire.pt --images ./data/images --labels ./data/labels --remote npu1

# 이미지만: 내 사진으로 캘리브레이션 (정확도 검증 없이 FPS만)
dxconv convert --pt fire.pt --images ./site_photos --remote npu1

# 장비 없이 DX-RT 3.2.0 장비용 (v8) 생성
dxconv convert --pt fire.pt --images ./data/images --target-dxrt 3.2.0

# 해상도 480으로 (FPS 우선)
dxconv convert --pt fire.pt --images ./data/images --labels ./data/labels --remote npu1 --imgsz 480

# 캘리브레이션 방식 직접 지정
dxconv convert --pt fire.pt --images ./data/images --labels ./data/labels --remote npu1 --calib-method ema --calib-num 100
```

### 출력

```
대상 DX-RT 3.5.0 → dxnn v9 (DX-COM 2.5.0)
자동 분할: ... (이미지 1000장 → 캘리브레이션 200 / 튜닝 200 / 평가 600)
▶ PT → ONNX
▶ ONNX → DXNN
▶ NPU 검증
  NPU mAP50-95 84.73  mAP50 97.67  FPS 205.9  (PT 84.88, Δ mAP50-95 -0.15%p, Δ mAP50 +0.01%p)
완료
```

결과 폴더 `<out>/convert/<model>_<날짜_시각>/`

| 파일 | 내용 |
|---|---|
| `<model>.dxnn` | 최종 결과물 |
| `<model>.onnx` | 중간 ONNX |
| `report.json` | 입력, 버전, 설정, dxnn 실제 설정 비교, NPU 검증 결과, 경고 |
| `resolved.yaml` | 실제 적용된 설정 |
| `compile.log` | DX-COM 출력 |
| `calib_subset/` | 실제 사용한 캘리브레이션 사진 |

---

## 2. `sweep` (설정 비교 실험, 실험용)

기본 사용에는 필요 없음. 6개 모델(ALVIS Fire·Falldown, COCO YOLO26 n/s/m/l)에서 추천값 1회 변환과 차이 0~0.5%p로 추천값이 같거나 좋았음. 새 DX-COM 버전이나 새 종류의 모델에서 추천값을 재검증할 때 사용

```bash
dxconv sweep --pt <pt 파일> --images <폴더> --labels <폴더> --remote <장비 이름> [옵션]
```

- 라벨 필수 (조합을 정확도로 비교), 모델당 약 11~30분 (모델 크기에 따라)
- 기본 비교 (`configs/sweep/default_space.yaml`) : ema / minmax × 사진 묶음 3개 (각 100장) = 6회 변환 → 방식별 Tuning mAP50-95 평균이 높은 방식 → 그 방식 중 Tuning 최대 묶음
- 전체 비교 (`--space configs/sweep/full_space.yaml`) : A opset × simplify → B 양자화 방식 × 장수 → C opt_level × aggressive
- 고른 조합은 Evaluation 분할로 다시 측정해 `summary.md`에 PT와 함께 기록

| 옵션 | 뜻 |
|---|---|
| `--save-preset` | 고른 값을 `configs/presets/<model>.yaml`로 저장 → 이후 `convert`에 자동 적용 |
| `--objective "<식>"` | 전체 비교 C 단계 선택 기준. 예: `"PT 대비 mAP 하락 <= 1%p 중 FPS 최대"`, `"FPS >= 20*15 중 mAP 최대"` |
| `--strict-objective` | 조건을 만족하는 조합이 없으면 멈춤 (기본: 하락이 가장 작은 조합으로 대체 선택) |
| `--space <yaml>` | 비교할 조합 파일 (기본 `configs/sweep/default_space.yaml`) |
| `--stage A\|B\|C` | 한 단계만 실행 (전체 비교일 때) |
| `--compile-only` | 장비 없이 컴파일만. 평가는 `sweep import-eval`로 입력 |
| `--retry-failed` | 실패했던 조합 다시 컴파일 |

```bash
dxconv sweep --pt fire.pt --images ./data/images --labels ./data/labels --remote npu1
dxconv sweep status --model fire
```

- `sweep status` : 진행 상황, 결과 표 다시 보기
- 중간에 멈춰도 같은 명령을 다시 실행하면 이어서 진행 (같은 설정은 다시 컴파일하지 않음)
- 결과 : `<out>/sweep/<model>/summary.md`, `sweep.db`

---

## 3. `env`

```bash
dxconv env            # 표로 출력
dxconv env --json     # JSON
```

DX-COM 버전과 출처(pip 설치 / PATH), DX-RT 버전, 선택될 DX-COM 버전 표시

---

## 4. `remote`

```bash
cp configs/remote.example.yaml configs/remote.yaml
dxconv remote setup --host npu1
dxconv remote check --host npu1
```

- `remote.yaml` : 장비 IP(`hostname`), 계정(`username`), SSH 포트(`port`) 입력
- `setup` : 장비에 평가 스크립트 복사 (작업 폴더 자동 생성)
- `check` : DX-RT, 드라이버, 펌웨어, 읽을 수 있는 dxnn 포맷, 평가 python, 디스크 여유
- 접속 실패 시 원인별 안내 (시간 초과 → 주소·포트·네트워크, `Permission denied` → `ssh-copy-id`)

`--host`는 `configs/remote.yaml`에 적은 장비 이름. 이후 `--remote npu1`로 사용

---

## 5. `dataset`

```bash
dxconv dataset split --images ./data/images --labels ./data/labels    # 분할 미리 보기
dxconv dataset check <데이터셋 이름 또는 yaml>                          # 목록·이미지·라벨 점검
```

분할 비율은 `configs/defaults.yaml`의 `split`

---

## 6. `baseline`

```bash
dxconv baseline --pt fire.pt --images ./data/images --labels ./data/labels
dxconv baseline --pt fire.pt --images ... --labels ... --onnx a.onnx --onnx b.onnx   # ONNX 여러 개 비교
```

NPU 없이 PC에서 PT와 FP32 ONNX 정확도 측정. 결과: `<out>/baseline/<model>_<날짜>/baseline_table.md`

---

## 7. `recommend`

```bash
dxconv recommend show                                              # 포맷별 추천값과 근거
dxconv recommend update runs/sweep/model1 runs/sweep/model2         # sweep 결과로 갱신 (키별 최다 선택값)
```

---

## 8. `export` / `compile` (단계별 실행)

```bash
dxconv export --pt fire.pt                                           # pt → ONNX만
dxconv compile --onnx fire.onnx --images ./data/images --remote npu1  # ONNX → dxnn만
```

옵션은 `convert`와 같음 (`export`는 `--opset`, `--imgsz`, `--set`, `--dry-run`)
