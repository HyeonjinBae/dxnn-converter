# dxconv

Ultralytics YOLO 모델(`.pt`)을 DEEPX NPU용 모델(`.dxnn`)로 바꿔 주는 도구

pt를 넣으면 배포할 장비의 DX-RT 버전에 맞는 컴파일러(DX-COM)를 골라, 실험으로 정한 설정으로 pt → ONNX → dxnn까지 변환

이미지/라벨 폴더는 가지고 있는 경우 사용 가능 (양자화 정확도가 조금 개선되고, 정확도 하락을 직접 측정 가능)

- 대상 : end2end(one2one head) Ultralytics YOLO 탐지 모델 (모델별 사전 설정 불필요)
- 컴파일러 : DEEPX DX-COM 2.5.0 (dxnn v9) / 2.4.1 (dxnn v8)
- 실행 PC : Linux x86-64 / Windows (WSL2 Ubuntu) / macOS (DX-COM은 docker로), Python 3.10 이상
- 변환은 NPU가 없는 PC에서, 정확도·FPS 확인만 NPU 장비에 SSH로 접속해서 진행

| 문서 | 내용 |
|---|---|
| [`docs/user_guide.md`](docs/user_guide.md) | 사용설명서 (설치, 모드, 결과 확인, 제약, 실측 참고) |
| [`docs/commands.md`](docs/commands.md) | 명령어와 옵션 전체 |
| [`docs/paths.md`](docs/paths.md) | 경로·설정 값 입력 위치 (명령 인자, 장비 설정, 환경변수) |
| [`docs/windows_wsl2.md`](docs/windows_wsl2.md) | Windows PC 설치·실행 방법과 확인 결과 |

확인한 환경 (코드 수정 없이 `configs/remote.yaml`만 바꿔서 동작)

| 변환 PC | DX-COM | 추천 모드 | 정밀 모드 | NPU 장비 |
|---|---|---|---|---|
| Windows + WSL2 Ubuntu 24.04 (x86-64) | pip `dx-com` | 1분 39초 ~ 3분 | 약 11분 | DX-M1 (DX-RT 3.5.0) |
| macOS (Apple Silicon) | docker `dxcom:2.5.0` | 3 ~ 4분 | 약 17분 | DX-M1 (DX-RT 3.5.0) |

## 사용법

```bash
# pt만 : 기본 사진으로 양자화해서 1번 변환 + NPU 장비에서 FPS 측정
dxconv convert --pt model.pt --remote <장비 이름>

# 추천 모드 : 보유 이미지로 양자화 + NPU 장비에서 PT 대비 정확도·FPS 확인 (몇 분)
dxconv convert --pt model.pt --images <이미지 폴더> --labels <YOLO 라벨 폴더> --remote <장비 이름>

# 정밀 모드 : 양자화 방식(ema/minmax)마다 사진 묶음 3개로 변환·평가 후 평균이 좋은 쪽 선택 (11~17분)
dxconv sweep --pt model.pt --images <이미지 폴더> --labels <YOLO 라벨 폴더> --remote <장비 이름> --save-preset

# 장비가 없는 경우 : 배포할 장비의 DX-RT 버전만 지정해 포맷만 맞춤
dxconv convert --pt model.pt --images <이미지 폴더> --labels <YOLO 라벨 폴더> --target-dxrt 3.2.0
```

pt를 넣은 후 과정

1. 데이터 분할 : 이미지/라벨 폴더를 비율로 Calibration / Tuning / Evaluation으로 나눔 (이미지가 없으면 기본 사진 coco128로 Calibration)
2. 포맷 결정 : 장비 DX-RT를 읽어(`--remote`) 또는 받아서(`--target-dxrt`) 장비가 읽을 수 있는 가장 새 dxnn 포맷과 DX-COM 버전 선택
3. 설정 결정 : 포맷별 추천값(`configs/recommend.yaml`) 적용
   - 이미지가 없으면 일반 사진용 값(`settings_no_data`) 사용
   - 모델 프리셋이 있으면 그 값이 우선
4. 변환 + 설정 확인 : dxnn 안에 실제로 기록된 설정이 요청한 값과 같은지 확인
5. NPU 검증 : 장비에서 Evaluation 분할로 mAP와 FPS를 측정하고 PT 대비 하락 폭 표시 (라벨이 없으면 FPS만)

## 설치

Linux x86-64 / Windows (WSL2 Ubuntu)

```bash
git clone https://github.com/HyeonjinBae/dxnn-converter.git ~/dxconv && cd ~/dxconv
./scripts/install_host.sh          # 가상환경 + dxconv + DX-COM 2.5.0 (pip install dx-com)
source .venv/bin/activate
dxconv env                         # DX-COM 버전과 설치 위치 확인
```

- Windows : PowerShell(관리자)에서 `wsl --install -d Ubuntu-24.04` 후 Ubuntu 창에서 위 순서대로 진행 ([`docs/windows_wsl2.md`](docs/windows_wsl2.md))
- v8 포맷이 필요하면 DX-COM 2.4.1 : `DXCOM_VERSION=2.4.1 ./scripts/install_host.sh` (가상환경 하나에 버전 하나)

macOS (Apple Silicon 포함) : DX-COM이 Linux 전용이라 docker로 실행 (로컬에 `dxcom`이 없으면 `dxcom:<버전>` 이미지 자동 사용)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
docker build --platform linux/amd64 -t dxcom:2.5.0 -f docker/Dockerfile.dxcom docker/
docker build --platform linux/amd64 -t dxcom:2.4.1 --build-arg DXCOM_VERSION=2.4.1 -f docker/Dockerfile.dxcom docker/   # v8 필요 시
```

- Docker Desktop → Settings → File sharing에 작업 폴더(`/Users` 등) 추가
- docker 안에는 GPU가 없어 양자화를 CPU로 진행 (`No supported GPU/CUDA environment` 경고는 정상)

설정 폴더(`configs/`)를 찾는 순서 : 환경변수 `DXCONV_CONFIG_DIR` → 저장소 안 `configs/` → 현재 폴더의 `configs/`

## NPU 장비 연결 (`--remote`)

```bash
cp configs/remote.example.yaml configs/remote.yaml      # 장비 주소(hostname)와 계정(username)만 필수, git 제외
ssh-copy-id -p <포트> <계정>@<주소>                       # SSH 키 등록 (비밀번호 접속은 미지원)
dxconv remote setup --host <장비 이름>                    # 장비에 평가 스크립트 복사
dxconv remote check --host <장비 이름>                    # DX-RT·드라이버·펌웨어 버전, 읽을 수 있는 dxnn 포맷, 평가용 python
```

- 장비 계정·주소·경로 : 코드에 없고 `configs/remote.yaml`에서만 읽음 (항목별 입력 값과 찾는 법 : [`docs/paths.md`](docs/paths.md))
- 평가용 python : dx_engine이 있는 python을 장비에서 자동 탐지, 여러 개면 PC와 OpenCV 버전이 같은 것 선택
- 장비 역할 : 추론만 담당 (`device_agent/`), mAP는 PC에서 PT 정확도와 같은 코드로 계산
- FPS : DX-RT `run_model -b` 값 (NPU 코어를 모두 쓴 최대 처리량)
- 동시 사용 : 여러 PC가 같은 장비를 써도 측정은 한 번에 하나씩 (장비의 잠금 파일)
- OpenCV 버전 : 장비와 PC가 다르면 경고 (이미지 크기 조정 결과 차이로 mAP 소폭 변동)

## dxnn 포맷

| 포맷 | DX-COM | 필요한 DX-RT | docker 이미지 |
|---|---|---|---|
| v9 | 2.5.0 | 3.5.0 이상 | `dxcom:2.5.0` |
| v8 | 2.4.1 | 3.1.0 이상 | `dxcom:2.4.1` |

- 버전을 정하는 순서 : 명령 옵션(`--dxcom-version`, `--dxnn-format`, `--target-dxrt`, `--remote`) → 설정 파일 → 설치된 DX-COM
- DX-COM 2.4.1 : CLI가 config의 캘리브레이션 방식·장수를 무시 → 2.4는 Python API(`dx_com.compile`)로 컴파일

## 설정

나중에 오는 것이 우선

```
defaults.yaml < recommend.yaml(목표 포맷) < presets/<model>.yaml < presets/<model>@dxcom<버전>.yaml < --config < --set < 명령 옵션
```

| 파일 | 내용 |
|---|---|
| `configs/defaults.yaml` | 기본값 (export, compile, 전처리, 분할 비율, 평가) |
| `configs/recommend.yaml` | 포맷별 추천값과 근거 (`dxconv recommend show` / `update`) |
| `configs/presets/<model>.yaml` | 모델별 값 (선택), `sweep --save-preset`으로 생성, git 제외 (PC마다 다름) |
| `configs/compat.yaml` | DX-COM 버전별 dxnn 포맷, 필요한 DX-RT, opset 범위 |
| `configs/sweep/default_space.yaml` | 정밀 모드에서 비교할 조합 (기본 : 양자화 방식 × 사진 묶음) |
| `configs/sweep/full_space.yaml` | 정밀 모드 전체 비교 (opset, 양자화, 최적화 수준까지) |
| `configs/remote.example.yaml` | NPU 장비 설정 예시 |
| `configs/datasets/example.yaml` | 분할을 목록 파일로 직접 정할 때 쓰는 예시 |

- 모델 이름 : `--model`을 안 주면 pt 파일 이름 사용
- 병합 : 파일끼리 겹치는 키만 덮어씀, 최종 설정은 형식 검사 (모르는 키나 잘못된 값이면 오류)
- 최종 설정 확인 : 결과 폴더의 `resolved.yaml`

## 정밀 모드 (`sweep`)

기본 : 양자화 방식만 비교 (`configs/sweep/default_space.yaml`)

| 비교 | 조합 | 고르는 방법 |
|---|---|---|
| 양자화 방식 × 사진 묶음 | ema / minmax × 사진 묶음 3개 (각 100장) = 6번 변환 | 방식별 Tuning mAP50-95 평균이 높은 쪽 → 그 방식 중 Tuning 결과가 가장 좋은 묶음 |

- 평균으로 고르는 이유 : 같은 설정도 양자화에 쓴 사진에 따라 정확도가 0.2~1%p 흔들림
- 고정 값 : opset·simplify·최적화 수준·aggressive partitioning (실험에서 결과가 같거나 항상 같은 값이 최고)
- 전체 비교 : `--space full_space.yaml` (opset × simplify → 양자화 방식 × 장수 → 최적화 수준 × aggressive, 30~35분)
- 최종 기록 : 고른 조합을 Evaluation 분할로 다시 측정해 `summary.md`에 PT와 함께 기록
- 고르는 기준 변경 : 전체 비교의 마지막 단계(속도 설정)는 `--objective`로 지정
  - 기본 : `"PT 대비 mAP 하락 <= 1%p 중 FPS 최대"`
  - 예 : `"FPS >= 20*15 중 mAP 최대"` (카메라 20대 × 15fps), 조건이 여러 개면 쉼표로 연결
  - 조건을 만족하는 조합이 없으면 하락이 가장 작은 조합(0.1%p 이내면 FPS가 높은 쪽) 선택 후 `[대체 선택]`으로 표시 (멈추려면 `--strict-objective`)
- 캐시 : 같은 설정은 다시 컴파일하지 않음, 중간에 멈춰도 다시 실행하면 이어서 진행
- 결과 : `<out>/sweep/<model>/` (`sweep.db`, `summary.md`, `combos/<해시>/`), v8은 `sweep/<model>@dxcom2.4.1/`
- 추천값 갱신 : `dxconv recommend update <sweep 폴더...>` (여러 모델 결과에서 가장 많이 뽑힌 값)
- 장비가 없을 때 : `--compile-only`로 컴파일만 하고, 장비에서 측정한 결과는 `dxconv sweep import-eval`로 입력

## 자동 분할 (`--images`, `--labels`)

| 분할 | 기본 비율 | 1000장일 때 | 300장일 때 |
|---|---|---|---|
| Calibration (양자화용) | 20%, 최대 200 | 200 | 60 |
| Tuning (정밀 모드에서 조합 고르기) | 20%, 최대 500 | 200 | 60 |
| Evaluation (최종 보고, 고르는 데는 미사용) | 나머지 | 600 | 180 |

- 사용 장수 : 추천 모드 50장 (이미지가 없으면 기본 사진 100장), 정밀 모드 묶음마다 100장 (이미지가 적으면 있는 만큼)
- 라벨 : `<라벨 폴더>/<이미지 파일 이름>.txt` (YOLO 형식), 클래스 번호가 **모델 클래스 번호와 같아야 함**
- 클래스 비율 : 분할마다 같도록 나눔 (이미지마다 가장 드문 클래스 기준)
- 중복 이미지 : 내용이 같은 이미지는 하나만 사용
- 재현성 : 같은 폴더·같은 설정이면 항상 같은 분할
- 미리 보기 : `dxconv dataset split --images ... --labels ...`

## 결과 폴더

```
<out>/convert/<model>_<YYYYMMDD_HHMMSS>/
├── <model>.dxnn          최종 결과물
├── <model>.onnx
├── report.json           입력 해시, 버전, 최종 설정, dxnn 실제 설정 비교, NPU 검증 결과, 경고, 소요 시간
├── resolved.yaml         실제로 적용된 설정
├── dxcom_config.json     DX-COM에 넘긴 설정
├── compile.log           DX-COM 출력
└── calib_subset/         실제로 양자화에 쓴 사진 (letterbox 처리본)
```

- 실패한 경우 : `report.json`에 `success: false`와 오류 내용 기록
- 자세한 오류 위치 : `dxconv --debug ...`

## 폴더 구성

```
configs/        기본값, 추천값, 호환성 표, 정밀 모드 조합, 예시 설정
dxconv/         명령(cli.py), 설정(config.py), 실행 흐름(pipeline.py), 자동 분할(datasplit.py), 추천값(recommend.py)
  env/          DX-COM/DX-RT 찾기, 버전 선택
  export/       pt 검사, export 규칙, ONNX 생성
  compile/      캘리브레이션, DX-COM 설정 생성, 컴파일, dxnn 실제 설정 읽기, 버전별 처리
  evaluate/     PT/ONNX 정확도, mAP 계산, NPU 원격 평가
  sweep/        정밀 모드 (조합 비교, 고르는 기준, 캐시, 결과 DB)
  remote/       SSH 장비 연결
device_agent/   NPU 장비에서 실행하는 평가 스크립트 (dxconv 없이 동작)
docker/         DX-COM docker 이미지
scripts/        변환 PC 설치 스크립트 (install_host.sh)
tests/          pytest
```

## 테스트

```bash
pytest                    # 실제 ultralytics / DX-COM 호출 테스트 제외
pytest -m integration     # 실제 ultralytics / DX-COM 호출
ruff check .
```
