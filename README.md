# dxconv

Ultralytics YOLO 모델(`.pt`)을 DEEPX NPU용 모델(`.dxnn`)로 변환하는 도구

- pt를 넣으면 배포할 NPU 장비의 DX-RT 버전에 맞춰 pt → ONNX → dxnn까지 변환
- 변환 설정은 실험으로 정한 추천값을 자동 적용 (모델별 사전 설정 불필요)
- 이미지·라벨 폴더가 있으면 그 사진으로 양자화하고, NPU 장비에서 PT 대비 정확도 하락과 FPS까지 측정
- 소요 시간 : 모델 1개 변환 + NPU 검증 약 2~3분 (YOLO26s, Windows WSL2 기준)

## 요구 사항

| 항목 | 내용 |
|---|---|
| 변환 PC | Linux x86-64 (Ubuntu 22.04 / 24.04) 또는 Windows 10 22H2 이상 / 11 + WSL2 Ubuntu |
| Python | 3.10 이상 |
| DX-COM | 2.5.0 (설치 스크립트가 함께 설치) |
| NPU 장비 (검증 시) | DEEPX NPU + DX-RT 3.5.0 이상, 변환 PC에서 SSH 접속 가능 |
| 대상 모델 | end2end(one2one head) Ultralytics YOLO 탐지 모델 (YOLO26 등) |

- DX-COM이 Linux x86-64 전용이라 Windows는 WSL2(Windows 내장 Linux)에서 실행
- NPU 장비가 없어도 변환은 가능 (정확도·FPS 검증만 생략)

## 설치

### Linux

```bash
sudo apt update
sudo apt install -y python3-venv git libgl1 libglib2.0-0t64
git clone https://github.com/HyeonjinBae/dxnn-converter.git ~/dxconv
cd ~/dxconv
./scripts/install_host.sh
source .venv/bin/activate
dxconv env
```

- `install_host.sh` : 가상환경(`.venv`) + dxconv + DX-COM 2.5.0 (`pip install dx-com`) 설치
- 성공 기준 : `dxconv env`에 DX-COM 2.5.0 (출처 pip) 표시
- "DX-RT 없음 / NPU 없음" 표시 : 정상 (변환 PC에는 NPU 불필요)
- Ubuntu 22.04 : `libglib2.0-0t64` 대신 `libglib2.0-0`
- 기본 python3가 3.10 미만이면 : `PYTHON=python3.12 ./scripts/install_host.sh`
- 새 터미널을 열 때마다 : `cd ~/dxconv && source .venv/bin/activate`

### Windows (WSL2)

1. PowerShell을 관리자 권한으로 열고 실행 후 재부팅

```powershell
wsl --install -d Ubuntu-24.04
```

2. 재부팅 후 `wsl -l -v`에 Ubuntu가 없으면 같은 명령을 한 번 더 실행
3. 시작 메뉴 → Ubuntu 실행 → Ubuntu 사용자 이름·비밀번호 설정
4. Ubuntu 창에서 `cd ~` 후 위 Linux 설치 순서대로 진행

- Windows 파일 경로, 문제 해결 : [`docs/windows_wsl2.md`](docs/windows_wsl2.md)

## NPU 장비 연결

정확도·FPS를 NPU 장비에서 측정할 때만 필요

### 1. 설정 파일 작성

```bash
cp configs/remote.example.yaml configs/remote.yaml
nano configs/remote.yaml
```

```yaml
hosts:
  npu1:
    hostname: 192.168.0.10
    username: user
    port: 22
```

- `npu1` : 장비 이름 (명령에서 `--remote npu1`로 사용, 자유롭게 지정)
- `hostname` : 장비 IP 주소
- `username` : 장비 로그인 계정
- `port` : SSH 포트 (22가 아니면 반드시 수정, 틀리면 접속 시간 초과)
- 작업 폴더, 평가용 Python : 장비에서 자동 탐지 (직접 지정 방법 : [`docs/paths.md`](docs/paths.md))
- `configs/remote.yaml` : git 제외 파일, 장비 정보는 이 파일에만 기록

### 2. SSH 키 등록 (처음 한 번)

입력을 기다리는 명령이 있으니 한 줄씩 실행

```bash
ls ~/.ssh/id_ed25519.pub
ssh-keygen -t ed25519
ssh-copy-id -p <포트> <계정>@<장비 IP>
ssh -p <포트> <계정>@<장비 IP> echo ok
```

- 첫 줄에서 파일이 보이면 `ssh-keygen`은 건너뜀 (키가 이미 있음)
- `ssh-keygen` : 엔터 3번
- `ssh-copy-id` : `yes` 입력 후 장비 비밀번호 1번 입력
- 마지막 줄이 비밀번호 없이 `ok`를 출력하면 성공
- 비밀번호 접속은 지원하지 않음 (SSH 키만)

### 3. 장비 점검

```bash
dxconv remote setup --host npu1
dxconv remote check --host npu1
```

- `setup` : 장비에 평가 스크립트 복사 (작업 폴더 자동 생성)
- `check` : DX-RT 버전, 읽을 수 있는 dxnn 포맷, 평가용 Python, 디스크 여유 확인 → 결과 `OK`면 준비 완료
- 접속 실패 시 원인(포트, 키 미등록, 주소)과 해결 방법을 안내

## 사용법

```bash
dxconv convert --pt model.pt --remote npu1
```

- pt만 있을 때 : 기본 사진(coco128, 처음 한 번 자동 다운로드)으로 양자화 → NPU FPS 측정

```bash
dxconv convert --pt model.pt --images <이미지 폴더> --labels <라벨 폴더> --remote npu1
```

- 이미지·라벨이 있을 때 : 그 사진으로 양자화 → NPU에서 PT 대비 mAP 하락과 FPS 측정
- 라벨 없이 `--images`만 주면 : 그 사진으로 양자화, FPS만 측정

```bash
dxconv convert --pt model.pt --target-dxrt 3.5.0
```

- NPU 장비가 없을 때 : 배포할 장비의 DX-RT 버전만 지정해 포맷을 맞춤 (검증 생략)
- 결과 폴더를 바꾸려면 `--out <폴더>` (기본 `runs/`)
- 명령어와 옵션 전체 : [`docs/commands.md`](docs/commands.md)

## 입력 데이터

| 입력 | 조건 |
|---|---|
| `--pt` | end2end Ultralytics YOLO 탐지 모델 |
| `--images` | jpg / jpeg / png / bmp, 하위 폴더까지 읽음, 학습 데이터와 비슷한 장면 300장 이상 권장 |
| `--labels` | YOLO 형식 `<이미지 이름>.txt`, 라벨 폴더 바로 아래에 위치 |

- 클래스 번호 : 라벨의 클래스 번호가 **모델의 클래스 번호와 같아야 함** (다르면 mAP가 0 근처)
- 모델 클래스 확인 : `python -c "from ultralytics import YOLO; print(YOLO('model.pt').names)"`
- 자동 분할 : 이미지를 비율로 나눠 양자화와 정확도 측정에 서로 다른 사진 사용

| 분할 | 비율 | 1000장일 때 | 용도 |
|---|---|---|---|
| Calibration | 20%, 최대 200 | 200 | 양자화 (이 중 50장 사용) |
| Tuning | 20%, 최대 500 | 200 | 설정 비교 실험용 (`sweep`) |
| Evaluation | 나머지 | 600 | PT 대비 정확도 측정 |

- 같은 폴더·같은 설정이면 항상 같은 분할, 클래스 비율이 분할마다 같도록 나눔

## 처리 과정

1. 데이터 준비 : 이미지 자동 분할 (이미지가 없으면 기본 사진 사용)
2. 포맷 결정 : 장비 DX-RT 버전을 읽어 장비가 읽을 수 있는 가장 새 dxnn 포맷 선택 (3.5.0 이상 → v9)
3. 설정 결정 : 추천값 적용 (`configs/recommend.yaml`)
4. 변환 : pt → ONNX (Ultralytics export) → dxnn (DX-COM)
5. 설정 확인 : dxnn 안에 실제로 기록된 설정이 요청한 값과 같은지 확인
6. NPU 검증 : 장비에서 Evaluation 분할로 mAP와 FPS 측정, PT 대비 하락 폭 출력 (라벨이 없으면 FPS만)

## 결과 확인

```
runs/convert/<모델 이름>_<날짜_시각>/
├── <모델 이름>.dxnn    최종 결과물 (NPU에 올릴 파일)
├── <모델 이름>.onnx
├── report.json         입력, 버전, 설정, NPU 검증 결과, 경고, 소요 시간
├── resolved.yaml       실제로 적용된 설정
└── compile.log         DX-COM 출력
```

실행 화면 예 (이미지·라벨이 있을 때)

```
대상 DX-RT 3.5.0 → dxnn v9 (DX-COM 2.5.0)
▶ NPU 검증
  NPU mAP50-95 58.79  mAP50 89.67  FPS 205.8  (PT 60.49, Δ mAP50-95 -1.70%p, Δ mAP50 -0.78%p)
완료
```

- 성공 : 마지막 줄 `완료`
- 정확도 : PT 대비 하락(Δ)이 보통 mAP50-95 −0.5 ~ −2%p
- `[dxnn 설정 불일치]` 경고가 있으면 해당 dxnn 사용 금지
- 무시해도 되는 경고 : `No supported GPU/CUDA environment` (CPU로 양자화), `cosine similarity ... below` (후처리 출력 특성)
- 실패 시 : 오류 메시지 + `report.json`, `compile.log` 확인 (문제 보고 시 두 파일 전달)

## 장비 배포

```bash
scp -P <포트> runs/convert/<폴더>/<모델 이름>.dxnn <계정>@<장비 IP>:<장비 경로>
```

- 장비에서 동작 확인 : `run_model -m <모델 이름>.dxnn -b -t 10` (FPS)
- Windows로 꺼내기 : [`docs/windows_wsl2.md`](docs/windows_wsl2.md) 5절

## 추천값과 근거

| 설정 | 값 | 근거 (DX-M1 실측) |
|---|---|---|
| ONNX opset, simplify | 17, 사용 | opset 11~21 모두 dxnn의 NPU 데이터가 같음 |
| 최적화 수준 (opt_level) | 1 | 정확도 같고 FPS 약 20% 높음 |
| aggressive partitioning | 끔 | 정확도 같고 FPS 약 7% 높음 (YOLO26 n/s/m/l 모두) |
| 양자화 (이미지 있음) | minmax, 50장 | ALVIS Fire·Falldown, COCO YOLO26 n/s/m/l에서 가장 좋거나 비슷 |
| 양자화 (pt만) | ema, 100장 (기본 사진) | 일반 사진으로 양자화할 때 minmax보다 안정적 |

- 양자화 설정을 모델마다 여러 개 비교해 고르는 방식(`dxconv sweep`)도 시험했으나, 6개 모델에서 추천값 1회 변환과 차이가 0~0.5%p로 추천값이 같거나 좋아 기본 사용에서 제외 (추천값 재검증용 실험 기능으로만 유지)

## 제약 사항

- 변환 PC : Linux x86-64만 지원 (DX-COM 제약), Windows는 WSL2
- 대상 모델 : end2end 탐지 모델 기준으로 검증 (세그멘테이션, 포즈 등 다른 task는 미검증)
- 장비 접속 : SSH 키 인증만 지원
- DX-RT 3.5.0 미만 장비 : dxnn v8 (DX-COM 2.4.1) 필요, 별도 가상환경에 `DXCOM_VERSION=2.4.1 ./scripts/install_host.sh`
- DX-COM 관련 알려진 제약 : [`docs/user_guide.md`](docs/user_guide.md) 9절

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/user_guide.md`](docs/user_guide.md) | 사용설명서 (설치, 사용법, 결과 확인, 제약, 실측 결과, 문제 해결) |
| [`docs/commands.md`](docs/commands.md) | 명령어와 옵션 전체 |
| [`docs/paths.md`](docs/paths.md) | 경로·설정 값 입력 위치 (명령 인자, 장비 설정, 환경변수) |
| [`docs/windows_wsl2.md`](docs/windows_wsl2.md) | Windows PC 설치·실행, 파일 경로, 문제 해결 |

## 개발

```bash
pytest
pytest -m integration
ruff check .
```

- `pytest` : 실제 ultralytics / DX-COM 호출 테스트 제외
- `pytest -m integration` : 실제 ultralytics / DX-COM 호출 (DX-COM 설치 필요)

```
configs/        기본값, 추천값, 호환성 표, 장비 설정 예시
dxconv/         명령(cli.py), 설정(config.py), 실행 흐름(pipeline.py), 자동 분할(datasplit.py)
  env/          DX-COM / DX-RT 찾기, 버전 선택
  export/       pt 검사, ONNX 생성
  compile/      캘리브레이션, DX-COM 설정 생성, 컴파일, dxnn 실제 설정 읽기
  evaluate/     PT 정확도, mAP 계산, NPU 원격 평가
  remote/       SSH 장비 연결
  sweep/        설정 비교 실험 (sweep)
device_agent/   NPU 장비에서 실행하는 평가 스크립트
scripts/        설치 스크립트
tests/          pytest
```
