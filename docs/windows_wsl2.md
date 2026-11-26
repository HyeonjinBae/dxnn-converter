# Windows PC에서 사용 (WSL2)

DX-COM(DEEPX 컴파일러)은 Linux x86-64 전용. Windows에서는 Windows 내장 기능인 **WSL2**로 Ubuntu를 띄워 그 안에서 실행. 듀얼부팅, 별도 Linux PC, docker 불필요

```
Windows ── WSL2 Ubuntu ── dxconv + DX-COM (pip) ── SSH ──▶ NPU 장비
```

---

## 1. 검증 결과 (2026-11-26)

저장소 코드를 고치지 않고 `configs/remote.yaml`만 작성해서 설치부터 장비 검증까지 확인

| 항목 | 값 |
|---|---|
| Windows / WSL | 빌드 10.0.26300 / WSL 3.0.5 (커널 6.18) |
| Ubuntu / Python | 24.04.5 LTS / 3.12.3 |
| PC | Intel Core Ultra 9 285K (24 스레드), 메모리 31GB (WSL 할당), RTX 5060 |
| NPU 장비 | DEEPX M1 3코어, DX-RT 3.5.0 (aarch64) |
| 테스트 | 189개 통과 (검증 당시 커밋 d5cedcd) |

| 실행 | 입력 | 소요 | NPU FPS | PT 대비 mAP50-95 / mAP50 |
|---|---|---:|---:|---|
| 추천 모드 | pt만 | 1분 39초 | 203.4 | (라벨 없음, 측정 안 함) |
| 추천 모드 | pt + 이미지 + 라벨 (Fire 1000장) | 2분 58초 | 205.9 | −1.75%p / −0.96%p |
| 정밀 모드 | 같은 데이터 | 10분 51초 | 205.2 | −1.46%p / −0.35%p |

- 같은 명령을 macOS에서 실행한 결과: −1.70%p, 205.7 FPS → **PC 환경이 달라도 0.1%p 이내로 같음** (차이는 PC 쪽 PyTorch/ultralytics 버전 차이로 PT 정확도가 조금 다르게 계산된 것)
- x86 PC라 DX-COM이 에뮬레이션 없이 실행돼 macOS(docker) 대비 약 2배 빠름

---

## 2. 준비물

| 항목 | 확인 방법 |
|---|---|
| Windows 10 22H2 이상 또는 Windows 11 | `Win + R` → `winver` |
| 가상화 사용 | 작업 관리자 → 성능 → CPU → "가상화: 사용" (꺼져 있으면 BIOS에서 VT-x/AMD-V 켜기) |
| 관리자 권한 | WSL 설치 때만 필요 |
| 인터넷 | 설치(약 3GB), 첫 실행 때 기본 캘리브레이션 사진(7MB) |
| pt 파일, (선택) 이미지·라벨 | Windows 어디에 있어도 됨 (5절 경로) |
| NPU 장비 접속 정보 | 주소, 포트, 계정, 장비 비밀번호 (SSH 키 등록에 1번) |

---

## 3. 설치 (약 30분)

### 3.1 WSL2 + Ubuntu

PowerShell을 **관리자 권한**으로 열고:

```powershell
wsl --install -d Ubuntu-24.04
```

- 재부팅 → Ubuntu 창이 열리면 Ubuntu 사용자 이름·비밀번호 설정 (비밀번호는 입력해도 안 보이는 게 정상)
- 재부팅 후 `wsl -l -v`에 배포가 없다고 나오면 **같은 명령을 한 번 더** 실행 (첫 실행은 WSL 기능만 켜고 끝나는 경우가 있음)
- 확인: `wsl -l -v` → `Ubuntu-24.04 ... 2`

### 3.2 dxconv + DX-COM

시작 메뉴 → **Ubuntu** 실행 후:

```bash
cd ~                                       # 처음 위치가 /mnt/c/WINDOWS/system32 일 수 있음 → 반드시 홈으로
sudo apt update && sudo apt install -y python3-venv git libgl1 libglib2.0-0t64 unzip
git clone https://github.com/HyeonjinBae/dxnn-converter.git ~/dxconv
cd ~/dxconv
./scripts/install_host.sh                  # 가상환경 + dxconv + DX-COM 2.5.0 (pip install dx-com)
source .venv/bin/activate
dxconv env                                 # DX-COM 2.5.0, 출처 pip → 성공
```

- Ubuntu 22.04는 `libglib2.0-0t64` 대신 `libglib2.0-0`
- DX-COM은 DEEPX 포털 wheel 없이 `pip install dx-com==2.5.0`으로 설치됨
- `dxconv env`의 "DX-RT 없음 / NPU 없음"은 정상 (변환 PC에는 NPU 불필요)
- 새 Ubuntu 창을 열 때마다: `cd ~/dxconv && source .venv/bin/activate`

### 3.3 NPU 장비 연결

```bash
ssh-keygen -t ed25519                      # 엔터 3번 (키가 없을 때만)
cp configs/remote.example.yaml configs/remote.yaml
nano configs/remote.yaml                   # 아래처럼 작성 → Ctrl+O 저장, Ctrl+X 종료
```

```yaml
hosts:
  npu1:                    # 장비 이름 (명령에서 --remote npu1)
    hostname: <장비 IP>
    port: <SSH 포트>       # 22 면 생략
    username: <장비 계정>
```

장비 작업 폴더와 평가 python은 자동으로 잡힘 (값을 적어야 하는 경우: [`paths.md`](paths.md) 2절)

**SSH 키 등록은 일반 Ubuntu 터미널에서** (Claude Code 등 도구 안의 `!` 명령으로 하면 비밀번호 입력 창이 없어 `ssh_askpass: No such file`로 실패)

```bash
ssh-copy-id -p <포트> <계정>@<장비 IP>       # yes → 장비 비밀번호 1번
ssh -p <포트> <계정>@<장비 IP> echo ok       # 비밀번호 없이 ok → 성공
dxconv remote setup --host npu1
dxconv remote check --host npu1             # DX-RT 3.5.0, dxnn v9, 평가 python(자동 탐지), OpenCV 버전 확인
```

---

## 4. 실행

```bash
cd ~/dxconv && source .venv/bin/activate

# pt만
dxconv convert --pt ~/fire.pt --remote npu1

# 이미지 + 라벨 (정확도까지)
dxconv convert --pt ~/fire.pt --images ~/fire_validation/images --labels ~/fire_validation/labels --remote npu1

# 정밀 모드 (약 11분)
dxconv sweep --pt ~/fire.pt --images ~/fire_validation/images --labels ~/fire_validation/labels --remote npu1
```

결과는 `~/dxconv/runs/` 아래 (`--out`으로 변경). 명령 설명: [`commands.md`](commands.md)

---

## 5. Windows 파일 쓰기

| Windows 위치 | Ubuntu에서 경로 |
|---|---|
| `C:\Users\<사용자>\Desktop\fire.pt` | `/mnt/c/Users/<사용자>/Desktop/fire.pt` |
| 바탕화면이 OneDrive로 동기화된 경우 | `"/mnt/c/Users/<사용자>/OneDrive/바탕 화면/fire.pt"` (공백 → 따옴표) |
| `D:\data` | `/mnt/d/data` |

```bash
cp "/mnt/c/Users/<사용자>/OneDrive/바탕 화면/fire.pt" ~/
unzip "/mnt/c/Users/<사용자>/OneDrive/바탕 화면/fire_validation.zip" -d ~/      # zip 이면
```

- 이미지는 **Ubuntu 홈으로 복사해서** 사용 (`/mnt/c/...`를 바로 읽으면 많이 느림)
- 결과 dxnn을 Windows로 꺼내기: `cp ~/dxconv/runs/convert/<폴더>/<모델>.dxnn /mnt/c/Users/<사용자>/Desktop/`
- Windows 탐색기에서 Ubuntu 홈 열기: 주소창에 `\\wsl$\Ubuntu-24.04\home\<Ubuntu 사용자>`

---

## 6. 문제 해결 (검증 중 실제로 겪은 것)

| 증상 | 원인 | 해결 |
|---|---|---|
| 재부팅 후 `wsl -l -v`에 "설치된 배포가 없습니다" | 첫 `wsl --install`은 WSL 기능만 켬 | 같은 명령 한 번 더 |
| `libglib2.0-0` 설치 실패 | Ubuntu 24.04에서 이름이 `libglib2.0-0t64`로 바뀜 | `libglib2.0-0t64` |
| `claude: command not found` 등 설치한 명령이 안 잡힘 | `~/.local/bin`이 PATH에 없음 | `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc` |
| `ssh-copy-id`가 `ssh_askpass: No such file` | 비밀번호 입력 창(터미널)이 없는 곳에서 실행 | 일반 Ubuntu 터미널에서 실행 |
| 바탕화면 파일이 `Desktop`에 없음 | OneDrive 바탕화면 리디렉션 | `ls /mnt/c/Users/<사용자>/OneDrive/` |
| 데이터가 zip | — | `unzip ... -d ~/` |
| 컴파일 중 `cosine similarity ... below 0.99` 경고 | end2end 출력(TopK)에서 나오는 DX-COM 경고 | 무시 (실제 mAP 하락으로 판단) |
| dxnn에 컴파일러 버전이 `2.5.0-rc.1`로 기록 | pip 패키지 dx-com 2.5.0의 내부 표기 | 정상 (v9 포맷 동일) |

---

## 7. 지켜야 할 것

- 장비 접속 정보는 `configs/remote.yaml`에만 (git 제외 파일). 문서·노트·커밋에 쓰지 않음
- SSH 개인 키(`~/.ssh/id_ed25519`, `.pub` 아닌 쪽)는 복사·공유하지 않음. 장비에는 공개 키만 등록됨
- 정밀 모드 `--save-preset` 결과(`configs/presets/<모델>.yaml`)도 git 제외 (PC마다 다름)
- 같은 NPU 장비를 여러 PC가 동시에 써도 됨: 장비 측정은 잠금 파일로 한 번에 하나씩 (다른 쪽은 대기)
