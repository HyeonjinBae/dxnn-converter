# dxconv 사용설명서

## 1. 개요

명령어와 옵션 전체: [`commands.md`](commands.md) / 경로·설정 값: [`paths.md`](paths.md) / Windows: [`windows_wsl2.md`](windows_wsl2.md)

Ultralytics YOLO 학습 결과(`.pt`) → DEEPX NPU 모델(`.dxnn`) 변환

```
model.pt ──(export)──▶ model.onnx ──(compile, DX-COM)──▶ model.dxnn ──▶ NPU 장비
```

- 변환 (`convert`): 실측 기반 추천 설정으로 1회 컴파일 + NPU 장비에서 정확도·FPS 확인. 약 2~3분. pt만으로도 동작
- 버전 맞춤: 배포 장비 DX-RT 버전에 맞는 dxnn 포맷(v9 / v8)으로 생성
- 설정 확인: 요청 설정이 dxnn에 실제로 들어갔는지 매번 확인, 실행 기록은 `report.json`에 저장

대상: end2end(one2one head) Ultralytics YOLO 탐지 모델 전부. 모델별 사전 설정 불필요

---

## 2. 준비물

| 항목 | 설명 |
|---|---|
| PC | Linux x86-64 또는 Windows 10/11 + WSL2 Ubuntu (DX-COM이 Linux x86-64 전용) |
| Python | 3.10 이상 |
| DX-COM | DEEPX 컴파일러 2.5.0. 설치 스크립트가 `pip install dx-com`으로 설치 |
| `.pt` 파일 | 변환할 모델 |
| 이미지 폴더 + YOLO 라벨 폴더 (선택) | 없으면 기본 사진으로 캘리브레이션, 정확도 검증 생략. 있으면 학습 데이터와 비슷한 장면 300장 이상 권장 |
| NPU 장비 (선택) | 정확도·FPS 검증용. SSH 키 접속 필요 |

라벨을 쓸 때 클래스 번호는 **모델 클래스 번호와 같아야 함.**
예) 모델이 `0 person, 1 sitting, 2 falldown`인데 라벨이 `0 = falldown`이면 라벨 번호를 2로 바꾼 사본을 사용

```bash
python -c "from ultralytics import YOLO; print(YOLO('model.pt').names)"   # 모델 클래스 확인
```

### dxnn 포맷과 장비 DX-RT

| 포맷 | DX-COM | 필요 DX-RT |
|---|---|---|
| v9 | 2.5.0 | 3.5.0 이상 |
| v8 | 2.4.1 | 3.1.0 이상 |

장비 DX-RT 확인: 장비에서 `dxrt-cli --status` (첫 줄 `DXRT v3.5.0`)

---

## 3. 설치

### 3.1 Linux (x86-64)

```bash
sudo apt update
sudo apt install -y python3-venv git libgl1 libglib2.0-0t64
git clone https://github.com/HyeonjinBae/dxnn-converter.git ~/dxconv
cd ~/dxconv
./scripts/install_host.sh
source .venv/bin/activate
dxconv env
```

- `install_host.sh`: 가상환경(.venv) + dxconv + DX-COM 2.5.0 (pip). Linux x86-64가 아니거나 Python이 3.10 미만이면 안내 후 중단
- `dxconv env`에 DX-COM 2.5.0 (출처 pip)이 보이면 성공
- Ubuntu 22.04는 `libglib2.0-0t64` 대신 `libglib2.0-0`
- 기본 python3가 3.10 미만이면 `PYTHON=python3.12 ./scripts/install_host.sh`
- v8 포맷용 DX-COM 2.4.1이 필요하면 별도 폴더에 `DXCOM_VERSION=2.4.1 ./scripts/install_host.sh` (가상환경 하나에 한 버전)

### 3.2 Windows (WSL2 Ubuntu)

DX-COM은 Linux 전용이라 Windows에서는 WSL2(Windows 내장 Linux)로 실행. 듀얼부팅 불필요. 자세한 절차·파일 경로·문제 해결: [`windows_wsl2.md`](windows_wsl2.md)

PowerShell(관리자)에서 실행 후 재부팅

```powershell
wsl --install -d Ubuntu-24.04
```

이후 시작 메뉴의 **Ubuntu**에서 3.1 Linux와 같은 절차. 경로 규칙(`/mnt/c/...`)은 [`paths.md`](paths.md) 1절

- 첫 `wsl --install`은 WSL 기능만 켜고 끝날 수 있음 → 재부팅 후 같은 명령을 한 번 더 실행하면 Ubuntu 설치
- 작업은 Ubuntu 홈(`cd ~`)에서. 처음 열린 위치가 `/mnt/c/WINDOWS/system32`일 수 있음
- `dxconv env`의 DX-COM 출처가 `pip`으로 표시되면 정상 (이 PC에서 직접 실행)

`dxconv env`에서 "DX-RT 없음 / NPU 없음"은 정상 (변환 PC에는 NPU 불필요)

### 3.3 NPU 장비 연결 (검증 시)

1. 장비 설정 파일: `hostname`(장비 IP), `username`(장비 계정), `port`(SSH 포트, 22가 아니면 반드시 수정)

```bash
cp configs/remote.example.yaml configs/remote.yaml
nano configs/remote.yaml
```

2. SSH 키 등록 (처음 한 번, 입력을 기다리는 명령이 있으니 한 줄씩)

```bash
ls ~/.ssh/id_ed25519.pub
ssh-keygen -t ed25519
ssh-copy-id -p <포트> <계정>@<장비 IP>
ssh -p <포트> <계정>@<장비 IP> echo ok
```

- 첫 줄에서 파일이 보이면 `ssh-keygen`은 건너뜀 (키가 이미 있음)
- `ssh-copy-id`: `yes` 입력 후 장비 비밀번호 1번. 마지막 줄이 비밀번호 없이 `ok`면 성공

3. 장비 준비, 점검

```bash
dxconv remote setup --host npu1
dxconv remote check --host npu1
```

- `setup`: 평가 스크립트 복사 / `check`: DX-RT·드라이버·펌웨어 버전, 읽을 수 있는 dxnn 포맷, 평가 python
- 접속 실패 시 원인별 안내 (시간 초과 → 주소·포트·네트워크, `Permission denied` → `ssh-copy-id`, 처음 접속 → 한 번 수동 접속)

- 각 항목에 넣을 값과 찾는 법: [`paths.md`](paths.md) 2절
- 장비 평가용 python(`dx_engine`, `numpy`, `opencv`)은 자동 탐지. 못 찾으면 `python:`에 경로 지정
- OpenCV는 PC와 같은 버전 권장 (다르면 경고, mAP 0.2%p 정도 변동 가능)
- `configs/remote.yaml`은 개인 설정이라 git 제외. 장비 정보는 이 파일에만 적음

---

## 4. 변환 (`convert`)

### 4.1 pt만 있을 때

```bash
dxconv convert --pt model.pt --remote npu1
```

1. 캘리브레이션: 기본 사진(일반 사진 128장, coco128)을 처음 한 번 받아 사용 (`~/.cache/dxconv/calib/`)
2. 포맷 결정: 장비 DX-RT를 읽어 포맷 선택 (3.5.0 이상이면 v9)
3. 설정 결정: 추천값 + 기본 사진용 값(`settings_no_data`, ema·100)으로 1회 컴파일
4. NPU 검증: FPS만 측정 (라벨이 없어 정확도는 못 잼)

```
NPU FPS 203.5  (라벨이 없어 mAP 미측정: --images + --labels 로 측정)
```

- 인터넷이 없으면 `--images` 또는 `--calib-dir`로 사진 폴더 지정
- 배포 현장 사진이 있으면 쓰는 쪽이 정확도가 높음 (실측 0.4~0.9%p 차이, 9절 실측 참고)

### 4.2 이미지/라벨이 있을 때

```bash
dxconv convert --pt model.pt --images ./data/images --labels ./data/labels --remote npu1
```

1. 데이터 분할: 이미지/라벨 폴더를 비율로 캘리브레이션 / 튜닝 / 평가로 나눔 (5절)
2. 포맷 결정: 장비 DX-RT를 읽어 포맷 선택
3. 설정 결정: 추천값(`configs/recommend.yaml`, minmax·50)으로 1회 컴파일
4. NPU 검증: 장비에서 평가 분할로 mAP와 FPS를 재고, PT 대비 하락 폭 출력

```
NPU mAP50-95 84.73  mAP50 97.67  FPS 205.9  (PT 84.88, Δ mAP50-95 -0.15%p, Δ mAP50 +0.01%p)
```

- 라벨 없이 `--images`만 주면 그 사진으로 캘리브레이션하고 FPS만 측정

장비가 없을 때

```bash
# 배포 장비 DX-RT 버전만 지정해 포맷 맞춤 (검증 없음)
dxconv convert --pt model.pt --images ./data/images --labels ./data/labels --target-dxrt 3.2.0
# 포맷 직접 지정
dxconv convert --pt model.pt --images ./data/images --labels ./data/labels --dxnn-format 8
```

설정 변경

```bash
dxconv convert --pt model.pt ... --calib-num 50 --calib-method minmax
dxconv convert --pt model.pt ... --set export.opset=19 --set compile.opt_level=0
dxconv convert --pt model.pt ... --config my.yaml
dxconv convert --pt model.pt ... --dry-run          # 컴파일 없이 설정과 DX-COM config만 확인
dxconv recommend show                               # 현재 추천값과 근거
```

우선순위: **명령행 옵션 > `--set` > `--config` 파일 > 모델 프리셋 > 추천값 > 기본값**

---

## 5. 데이터 자동 분할

| 분할 | 용도 | 기본 비율 | 1000장 기준 |
|---|---|---|---|
| 캘리브레이션 | 양자화 범위 계산 | 20% (최대 200), 후보 25/50/100% | 50/100/200 |
| 튜닝 | 설정 비교 실험(`sweep`)용 | 20% (최대 500) | 200 |
| 평가 | 최종 보고 (선택에 미사용) | 나머지 | 600 |

```bash
dxconv dataset split --images ./data/images --labels ./data/labels    # 분할 미리 보기
```

- 클래스 비율이 모든 분할에서 같도록 분할, 내용이 같은 이미지는 하나만 사용
- 같은 폴더·같은 설정이면 항상 같은 분할
- 비율 변경: `configs/defaults.yaml`의 `split`

---

## 6. 설정 비교 실험 (`sweep`, 실험용)

모델마다 양자화 설정(ema / minmax × 사진 묶음 3개)을 여러 번 변환·평가해 고르는 기능. **기본 사용에는 필요 없음**

- 시험 결과: ALVIS Fire·Falldown, COCO YOLO26 n/s/m/l 6개 모델에서 추천값 1회 변환과 차이 0~0.5%p, 추천값이 같거나 좋음 (9절 실측 참고)
- 남겨 둔 용도: 새 DX-COM 버전이나 새 종류의 모델에서 추천값이 여전히 맞는지 재검증
- 사용법과 옵션: [`commands.md`](commands.md) 2절

## 7. PT / ONNX 정확도 재기 (`baseline`)

NPU 없이 PC에서 원본 PT와 FP32 ONNX의 mAP 계산

```bash
dxconv baseline --pt model.pt --images ./data/images --labels ./data/labels
```

결과: `<out>/baseline/<model>_<날짜>/baseline_table.md` (mAP50, mAP50-95, P/R, 클래스별 AP)

---

## 8. 결과 확인

실행마다 폴더 1개 생성: `<out>/convert/<model>_<날짜_시각>/`

| 파일 | 내용 |
|---|---|
| `<model>.dxnn` | **최종 결과물** (NPU에 올릴 파일) |
| `<model>.onnx` | 중간 ONNX |
| `report.json` | 실행 기록 전체 (성공 여부, 입력 해시, 버전, 설정, dxnn 실제 설정, NPU 검증, 경고, 소요 시간) |
| `resolved.yaml` | 실제 적용된 최종 설정 |
| `compile.log` | DX-COM 출력 |
| `calib_subset/` | 실제 사용한 캘리브레이션 이미지 (letterbox 처리본) |

### 8.1 확인 항목

1. 마지막 줄 **"완료"** 여부. 실패 시 "오류: …" + 종료 코드 1
2. **NPU 검증** 줄의 PT 대비 하락(Δ). 보통 mAP50-95 −0.5~−1.5%p
3. **경고** 목록. `[dxnn 설정 불일치]`가 있으면 해당 dxnn 사용 금지

### 8.2 무시해도 되는 경고

| 경고 | 의미 |
|---|---|
| `No supported GPU/CUDA environment` | GPU 없이 CPU로 양자화 (GPU 없는 PC, WSL2) |
| `After surgeon, cosine similarity … below 0.90/0.99` | 후처리(TopK) 부분 그래프 변환 경고. NPU 실측 정확도로 판단 |

### 8.3 장비 배포

1. 장비 DX-RT가 dxnn 포맷을 읽을 수 있는지 확인 (2절 표)
2. 기존 모델 파일 백업 후 새 dxnn으로 교체
3. 장비에서 확인: `run_model -m model.dxnn -b -t 10` (FPS), `parse_model -m model.dxnn` (포맷, 메모리)

---

## 9. 알려진 제약

| 항목 | 내용 |
|---|---|
| DX-COM 2.4.1 CLI | config의 `calibration_method` / `calibration_num` 무시. dxconv는 2.4를 Python API로 컴파일해 우회 |
| `calibration_method: iqr` | DX-COM 2.5.0 일반 컴파일에서 minmax와 결과 동일 → 기본 탐색에서 제외 (지정은 가능) |
| DX-COM `resize` 가운데 패딩 | 미지원 → letterbox를 PC에서 미리 처리 (`compile.preprocess.letterbox_mode: host`) |
| `aggressive_partitioning` | dxnn·로그에 값이 남지 않음 → 명령행 플래그로만 적용 여부 확인 |
| Q-PRO (`--use_q_pro`) | 미지원. GPU 없이 CPU로 6시간 이상, 시험한 모델에서 정확도 개선 없음 (−0.15 → −0.64%p) |
| `--quant_diagnosis` | 미지원. 켜면 ema 캘리브 결과가 일반 컴파일과 달라지고(시험 모델 −1.01 → −1.76%p), 리포트의 cosine/QSNR은 모델 간 mAP 하락과 맞지 않음 |
| dxnn의 컴파일러 버전 표기 | `pip install dx-com==2.5.0`으로 만든 dxnn에는 `2.5.0-rc.1`로 기록됨. dxconv는 2.5.0으로 취급 (v9 포맷 동일) |
| `--checkpoint` + `--recalibration_method` | 2.5.0 CLI에서 방식 변경이 적용되지 않음 (결과·dxnn 기록 모두 원래 방식) |
| 장비 접속 | SSH 키 인증만 지원 |

### 실측 참고 (DX-M1 3코어, YOLO26)

| 항목 | 결과 |
|---|---|
| opt_level × aggressive | n/s/m/l 모두 opt1 + aggressive off가 최대 FPS (n 264, s 201, m 123, l 94) |
| ONNX opset | 11~21 모두 dxnn NPU 데이터 바이트 동일 (DX-COM이 내부에서 opset 21로 변환) |
| 캘리브 이미지 | 실제 사진이면 10장, 다른 데이터셋 사진도 차이 0.2%p 이내. 노이즈 −1%p, 단색 이미지는 붕괴 |
| pt만 (기본 사진 coco128) | 자기 데이터 최선값 대비 mAP50-95 0.4~0.9%p, mAP50 0.1~0.4%p 손해. ema가 minmax보다 일반 사진에 강함 |
| 캘리브 방식 | 자기 데이터: minmax·50이 ema·100보다 0.4~0.5%p 좋음 / 일반 사진: ema·100이 좋음 |
| 튀는 사진 20% 섞기 | 최종 결과 차이 0.05~0.16%p |
| 라벨 없는 검증 (PT 예측을 정답으로) | 붕괴(단색 캘리브)는 잡지만 0.5%p 수준 차이는 구분 못 함 → 미사용 |
| 캘리브 반복 | 같은 설정이라도 뽑힌 이미지에 따라 0.2~0.8%p 흔들림 → 1회 측정 1위는 신뢰 낮음 |
| 설정 비교 실험(`sweep`) vs 추천값 | 6개 모델(Fire, Falldown, COCO YOLO26 n/s/m/l)에서 차이 0~0.5%p, 추천값이 같거나 좋음 → 기본 흐름에서 제외 |
| 캘리브 결정성 | 같은 onnx·config·이미지면 결과 동일 |
| 입력 해상도 (640 학습 모델) | 480: 전체 −1%p 이내, FPS 1.7배 / 416: −2%p, 2.2배 / 320: −7%p, 3.1배 |

---

## 10. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `캘리브레이션 입력이 없습니다` | `--images`(권장) 또는 `--calib-dir` 지정 |
| 캘리브레이션 이미지 수 부족 | 이미지 수 ≥ `calibration_num` 필요. `--calib-num`으로 줄이거나 이미지 추가 |
| `DX-COM 실행 실패` / dxcom 없음 | `dxconv env`로 확인. 가상환경 활성화(`source .venv/bin/activate`) 확인, 없으면 `pip install dx-com==2.5.0` |
| `장비 ... 는 dxnn v9 을 읽을 수 없습니다` | 장비 DX-RT 3.5.0 미만. `--remote`로 자동 선택하거나 `--dxnn-format 8` |
| NPU mAP가 거의 0 | 라벨 클래스 번호가 모델과 다름 (2절) |
| `[dxnn 설정 불일치]` 경고 | 해당 dxnn 사용 금지. `report.json`의 `dxnn_settings.mismatches` 확인 |
| `설치된 DX-COM 버전이 목표 버전(...)과 다릅니다` | 장비가 v8(DX-RT 3.5.0 미만)이면 별도 폴더에 `DXCOM_VERSION=2.4.1 ./scripts/install_host.sh` |
| `장비 접속 실패` | 출력된 원인대로 조치 (포트·주소·키 등록). `ssh -p <포트> <계정>@<장비 IP> echo ok`가 비밀번호 없이 되는지 확인 |
| 원인 추가 확인 | `dxconv --debug convert ...`, run 폴더 `compile.log`, `report.json` |

문제 보고 시 **run 폴더의 `report.json`과 `compile.log`** 함께 전달

---

## 11. 용어

| 용어 | 뜻 |
|---|---|
| DX-COM | DEEPX 컴파일러 (ONNX → dxnn) |
| DX-RT | NPU 장비 런타임. dxnn 포맷별 최소 버전 존재 |
| 캘리브레이션 | INT8 양자화 범위를 정하기 위해 이미지를 모델에 통과시키는 과정 |
| ema / minmax | 캘리브레이션 방식 (값 범위 결정 방법) |
| opt_level | DX-COM 최적화 수준 (1이 기본, FPS 더 높음) |
| aggressive partitioning | NPU에 최대한 많은 연산을 올리는 분할 옵션 |
| letterbox | 비율 유지 리사이즈 후 남는 영역을 회색(114)으로 채우는 전처리. 학습과 같아야 함 |
| 튜닝 / 평가 분할 | 튜닝: 조합 선택용. 평가: 최종 보고용 (선택에 미사용) |
| 추천값 / 프리셋 | 추천값: 포맷별 공통 설정. 프리셋: 모델별 설정 (선택, `configs/presets/`) |
