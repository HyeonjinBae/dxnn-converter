# 경로와 설정 값 정리

dxconv 코드에는 장비 계정, 호스트 이름, IP, 설치 경로가 들어 있지 않음. 환경마다 다른 값은 아래 세 곳에서만 받음.

| 어디에 | 무엇을 | 예 |
|---|---|---|
| 명령행 인자 | 이 PC의 입력·출력 경로 (pt, 이미지, 라벨, 결과 폴더) | `--pt ~/models/fire.pt` |
| `configs/remote.yaml` | NPU 장비 접속 정보 (주소, 계정, 포트, 키, 장비 쪽 경로) | `hostname: 192.168.0.10` |
| 환경변수 (선택) | 설정 폴더, 캐시 폴더 위치 | `DXCONV_CACHE_DIR=/data/cache` |

`configs/remote.yaml`은 git에서 제외됨. 장비 정보는 이 파일에만 적고 문서·노트·커밋에 넣지 않음.

---

## 1. 명령에 넣는 경로 (이 PC 기준)

| 옵션 | 넣는 것 | 형식 | 예 |
|---|---|---|---|
| `--pt` | 학습된 YOLO 모델 파일 | 파일 경로 (`.pt`) | `~/models/fire.pt`, `./best.pt` |
| `--images` | 이미지 폴더 | 폴더 경로. **하위 폴더까지 모두 읽음** (jpg/jpeg/png/bmp) | `~/data/fire/images` |
| `--labels` | YOLO 라벨 폴더 | 폴더 경로. `<이미지 파일 이름>.txt`가 **이 폴더 바로 아래**에 있어야 함 | `~/data/fire/labels` |
| `--calib-dir` | 캘리브레이션에만 쓸 사진 폴더 | 폴더 경로 (하위 폴더 포함, 라벨 불필요) | `~/site_photos` |
| `--dataset` | 분할 목록을 직접 정한 데이터셋 설정 | yaml 경로 또는 `configs/datasets/<이름>.local.yaml`의 이름 | `./my_split.yaml` |
| `--config` | 설정 값을 모은 파일 | yaml 경로 | `./my_settings.yaml` |
| `--out` | 결과를 저장할 폴더 | 폴더 경로 (없으면 만듦). 기본 `runs` = 명령을 실행한 폴더 아래 | `~/dxconv_out` |

- 상대 경로는 **명령을 실행한 폴더** 기준. `~`는 홈 폴더
- 경로에 공백이 있으면 따옴표: `--images "/home/user/my data/images"`
- 이미지와 라벨은 같은 폴더여도 됨 (`--images ~/data --labels ~/data`)

### 이미지 / 라벨 폴더 예

```
~/data/fire/
├── images/
│   ├── cam1/0001.jpg        ← 하위 폴더 OK
│   ├── cam1/0002.jpg
│   └── 0003.png
└── labels/
    ├── 0001.txt             ← 이미지 이름(확장자 제외)과 같은 이름, 하위 폴더 없이
    ├── 0002.txt
    └── 0003.txt
```

```bash
dxconv convert --pt ~/models/fire.pt --images ~/data/fire/images --labels ~/data/fire/labels --remote npu1
```

- 라벨 한 줄: `<클래스 번호> <x중심> <y중심> <너비> <높이>` (0~1 비율, YOLO 형식)
- 클래스 번호는 **모델 클래스 번호와 같아야 함**. 모델 클래스 확인:
  `python -c "from ultralytics import YOLO; print(YOLO('model.pt').names)"`
- 하위 폴더가 달라도 파일 이름(확장자 제외)이 같은 이미지가 있으면 라벨을 구분할 수 없어 오류
- 라벨이 없는 이미지는 "물체 없음"으로 평가에 포함

### Windows (WSL2)에서 경로

| Windows 위치 | WSL2 Ubuntu에서 쓰는 경로 |
|---|---|
| `C:\Users\홍길동\Desktop\fire.pt` | `/mnt/c/Users/홍길동/Desktop/fire.pt` |
| 바탕화면이 OneDrive로 동기화된 경우 | `/mnt/c/Users/홍길동/OneDrive/바탕 화면/fire.pt` (경로에 공백 → 따옴표) |
| `D:\data\images` | `/mnt/d/data/images` |
| Ubuntu 홈 | `~/` (= `/home/<Ubuntu 사용자>/`) |

- 바탕화면 파일이 `Desktop`에 없으면 OneDrive 아래를 확인: `ls /mnt/c/Users/<사용자>/OneDrive/`
- 데이터가 zip이면 풀어서 사용: `unzip "/mnt/c/.../fire_validation.zip" -d ~/` (`sudo apt install unzip`)
- `/mnt/c/...`는 그대로 써도 되지만 많이 느림 → 이미지가 많으면 Ubuntu 홈으로 복사해서 사용
  `cp -r /mnt/c/Users/홍길동/Desktop/fire_data ~/fire_data`
- Windows 탐색기에서 Ubuntu 홈 열기: 주소창에 `\\wsl$\Ubuntu-24.04\home\<Ubuntu 사용자>`

---

## 2. NPU 장비 설정 (`configs/remote.yaml`)

```bash
cp configs/remote.example.yaml configs/remote.yaml
```

```yaml
hosts:
  npu1:                    # 장비 이름 (명령에서 --remote npu1)
    hostname: 192.168.0.10
    username: user
    # port: 22
    # key_filename: ~/.ssh/id_ed25519
    # workdir: dxconv_agent
    # python: auto
    # dxrt_bin: /usr/local/bin
    # lock_file: /tmp/dxconv_eval.lock
```

| 항목 | 필수 | 기본값 | 넣는 값 | 값 찾는 법 |
|---|---|---|---|---|
| (장비 이름) | ✓ | — | 아무 이름 (영문·숫자) | 명령에서 `--remote <이름>`으로 씀 |
| `hostname` | ✓ | — | 장비 IP 또는 호스트 이름 | 장비에서 `hostname -I` |
| `username` | ✓ | — | 장비 로그인 계정 | 장비에서 `whoami` |
| `port` | | 22 | SSH 포트 | 평소 `ssh -p <포트>`로 접속하면 그 번호 |
| `key_filename` | | ssh 기본 키 | **이 PC의** SSH 개인 키 경로 | `ls ~/.ssh/` → `id_ed25519` 등 (`.pub` 아닌 쪽) |
| `workdir` | | `dxconv_agent` | 장비 작업 폴더. 상대 경로면 장비 홈 기준 | 보통 그대로 둠. 디스크가 부족하면 큰 디스크 경로 |
| `python` | | `auto` | dx_engine이 있는 **장비의** python 경로 | 장비에서 `ls ~/dx-all-suite*/dx-runtime/venv-dx-runtime/bin/python` |
| `dxrt_bin` | | 없음 | **장비의** `dxrt-cli`, `run_model`이 있는 폴더 | 장비에서 `which dxrt-cli`가 안 나올 때만. `find / -name dxrt-cli 2>/dev/null` |
| `lock_file` | | `/tmp/dxconv_eval.lock` | 측정 동시 실행 방지용 파일 | 보통 그대로 둠 |

### `python: auto`가 찾는 순서

1. 장비의 `python3`
2. `~/dx-all-suite*/dx-runtime/venv-dx-runtime/bin/python`
3. `~/*/dx-runtime/venv-dx-runtime/bin/python`, `~/dx_rt/venv*/bin/python`, `~/*/venv-dx-runtime/bin/python`

이 중 `import dx_engine`이 되는 것을 사용. 여러 개면 **이 PC와 OpenCV 버전이 같은 것** (letterbox 결과가 같아야 mAP가 같음) → 없으면 dx_engine 버전이 가장 높은 것. `dxconv remote check` 결과에 실제로 고른 python과 OpenCV 버전이 나옴.

### 장비 연결 순서

```bash
ssh-keygen -t ed25519                                   # 이 PC에 키가 없을 때만 (엔터 3번)
ssh-copy-id -p <port> <username>@<hostname>             # 장비 비밀번호 1번 입력 (일반 터미널에서. 아래 주의)
ssh -p <port> <username>@<hostname> echo ok             # 비밀번호 없이 ok 가 나와야 함
dxconv remote setup --host npu1                         # 장비 작업 폴더에 평가 스크립트 복사
dxconv remote check --host npu1                         # DX-RT, 드라이버, 펌웨어, python, 디스크 점검
```

- `ssh-copy-id`는 비밀번호를 입력할 터미널이 필요함. Claude Code 같은 도구 안의 `!` 명령으로 실행하면 `ssh_askpass: No such file`로 실패 → 일반 터미널 창에서 실행

`remote check`에서 확인할 것: 자동으로 고른 평가 python 경로와 작업 폴더, `DX-RT 3.5.0`(v9) 또는 `3.1.0` 이상(v8), python의 `dx_engine` 버전, OpenCV 버전 (PC와 다르면 경고)

### 장비 쪽 폴더 (`workdir`, 자동 생성)

```
~/dxconv_agent/
├── agent/      평가 스크립트 (dxconv 실행 때마다 PC에서 복사)
├── models/     평가한 dxnn (<해시 16자>.dxnn)
├── data/       평가 이미지 (분할 목록에 있는 것만, 처음 한 번 업로드)
└── results/    평가 결과 JSON
```

정리할 때는 `models/`, `data/`를 지워도 됨 (다음 평가 때 다시 올림)

---

## 3. 환경변수 (선택)

| 환경변수 | 기본 | 바꿀 때 |
|---|---|---|
| `DXCONV_CONFIG_DIR` | 저장소의 `configs/` → 현재 폴더의 `configs/` | 설정 파일(`remote.yaml` 등)을 다른 곳에 둘 때. 폴더 안에 `defaults.yaml`이 있어야 함 |
| `DXCONV_CACHE_DIR` | `~/.cache/dxconv` | 기본 캘리브레이션 사진을 받아 둘 위치를 바꿀 때 (인터넷이 없는 PC는 미리 복사) |

```bash
export DXCONV_CONFIG_DIR=~/my_dxconv_configs
export DXCONV_CACHE_DIR=/data/dxconv_cache
```

---

## 4. 결과가 생기는 곳

| 무엇 | 위치 |
|---|---|
| 변환 결과 (`convert`) | `<--out>/convert/<모델 이름>_<날짜_시각>/<모델 이름>.dxnn` |
| 정밀 모드 (`sweep`) | `<--out>/sweep/<모델 이름>/` (`summary.md`, `sweep.db`, `combos/`) |
| 자동 분할 목록 | `<--out>/datasets/<모델 이름>_<지문>/` (같은 폴더·같은 설정이면 재사용) |
| 기본 캘리브레이션 사진 | `~/.cache/dxconv/calib/coco128/` |
| 정밀 모드 프리셋 (`--save-preset`) | `configs/presets/<모델 이름>.yaml` (git 제외, 이후 `convert --model <이름>`에 자동 적용) |

`<모델 이름>`은 `--model`로 정하고, 생략하면 pt 파일 이름 (`fire.pt` → `fire`)

---

## 5. 자주 하는 실수

| 증상 | 원인 | 해결 |
|---|---|---|
| `원격 장비 설정이 없습니다` | `configs/remote.yaml` 없음 | `cp configs/remote.example.yaml configs/remote.yaml` 후 수정 |
| `remote.yaml 에 'xxx' 호스트가 없습니다` | `--remote` 이름과 yaml의 장비 이름이 다름 | yaml의 `hosts:` 아래 이름 확인 |
| SSH 접속 실패 / `Permission denied (publickey)` | 키 미등록 또는 `key_filename`이 `.pub` 파일 | `ssh-copy-id` 다시, `key_filename`은 `.pub` 없는 개인 키 |
| `dx_engine 이 있는 python 을 찾지 못했습니다` | DX-RT python이 표준 위치에 없음 | 장비에서 경로를 찾아 `python:`에 적기 |
| `DX-RT(dxrt-cli)를 찾지 못했습니다` | SSH 접속 시 PATH에 DX-RT가 없음 | 장비에서 `dxrt-cli` 위치를 찾아 `dxrt_bin:`에 폴더 적기 |
| NPU mAP가 거의 0 | 라벨 클래스 번호가 모델과 다름 | 1절 클래스 번호 확인 |
| `/mnt/c/...` 경로에서 매우 느림 (WSL) | Windows 디스크를 거쳐 읽음 | Ubuntu 홈(`~/`)으로 복사 |
| `기본 캘리브레이션 사진을 받지 못했습니다` | 인터넷 없음 | `--images` 또는 `--calib-dir`로 사진 지정, 또는 `DXCONV_CACHE_DIR`에 미리 복사 |
