# Windows PC에서 사용 (WSL2)

DX-COM(DEEPX 컴파일러)은 Linux x86-64 전용. Windows에서는 Windows 내장 기능인 **WSL2**로 Ubuntu를 띄워 그 안에서 실행. 듀얼부팅, 별도 Linux PC 불필요

```
Windows ── WSL2 Ubuntu ── dxconv + DX-COM (pip) ── SSH ──▶ NPU 장비
```

- 확인 : Windows 11 + WSL2 Ubuntu 24.04에서 코드 수정 없이 `configs/remote.yaml`만 작성해 설치 → 변환 → NPU 검증까지 동작 (변환 + 검증 약 2~3분)

---

## 1. 준비물

| 항목 | 확인 방법 |
|---|---|
| Windows 10 22H2 이상 또는 Windows 11 | `Win + R` → `winver` |
| 가상화 사용 | 작업 관리자 → 성능 → CPU → "가상화: 사용" (꺼져 있으면 BIOS에서 VT-x/AMD-V 켜기) |
| 관리자 권한 | WSL 설치 때만 필요 |
| 인터넷 | 설치(약 3GB), 처음 실행 때 기본 캘리브레이션 사진(7MB) |
| pt 파일, (선택) 이미지·라벨 | Windows 어디에 있어도 됨 (4절 경로) |
| NPU 장비 접속 정보 | IP 주소, SSH 포트, 계정, 장비 비밀번호 (SSH 키 등록에 1번) |

---

## 2. 설치 (약 30분)

### 2.1 WSL2 + Ubuntu

PowerShell을 **관리자 권한**으로 열고 실행

```powershell
wsl --install -d Ubuntu-24.04
```

- 재부팅 → Ubuntu 창이 열리면 Ubuntu 사용자 이름·비밀번호 설정 (비밀번호는 입력해도 안 보이는 게 정상)
- 재부팅 후 `wsl -l -v`에 배포가 없다고 나오면 **같은 명령을 한 번 더** 실행 (첫 실행은 WSL 기능만 켜고 끝나는 경우가 있음)
- 확인 : `wsl -l -v` → `Ubuntu-24.04 ... 2`

### 2.2 dxconv + DX-COM

시작 메뉴 → **Ubuntu** 실행 후 한 줄씩 실행

```bash
cd ~
sudo apt update
sudo apt install -y python3-venv git libgl1 libglib2.0-0t64 unzip
git clone https://github.com/HyeonjinBae/dxnn-converter.git ~/dxconv
cd ~/dxconv
./scripts/install_host.sh
source .venv/bin/activate
dxconv env
```

- `cd ~` : 처음 위치가 `/mnt/c/WINDOWS/system32`일 수 있어 반드시 Ubuntu 홈으로 이동
- `install_host.sh` : 가상환경 + dxconv + DX-COM 2.5.0 (`pip install dx-com`)
- `dxconv env` : DX-COM 2.5.0, 출처 pip이면 성공. "DX-RT 없음 / NPU 없음"은 정상 (변환 PC에는 NPU 불필요)
- Ubuntu 22.04는 `libglib2.0-0t64` 대신 `libglib2.0-0`
- 새 Ubuntu 창을 열 때마다 : `cd ~/dxconv && source .venv/bin/activate`

### 2.3 NPU 장비 연결

```bash
cp configs/remote.example.yaml configs/remote.yaml
nano configs/remote.yaml
```

```yaml
hosts:
  npu1:
    hostname: <장비 IP>
    username: <장비 계정>
    port: <SSH 포트>
```

- `npu1` : 장비 이름 (명령에서 `--remote npu1`)
- `port` : 22가 아니면 반드시 수정 (틀리면 접속 시간 초과)
- 저장 : `Ctrl+O` → 엔터 → `Ctrl+X`
- 장비 작업 폴더와 평가 python은 자동으로 잡힘 (직접 적어야 하는 경우 : [`paths.md`](paths.md) 2절)

SSH 키 등록은 **일반 Ubuntu 터미널에서 한 줄씩** 실행 (Claude Code 등 도구 안의 `!` 명령으로 하면 비밀번호 입력 창이 없어 `ssh_askpass: No such file`로 실패)

```bash
ls ~/.ssh/id_ed25519.pub
ssh-keygen -t ed25519
ssh-copy-id -p <포트> <계정>@<장비 IP>
ssh -p <포트> <계정>@<장비 IP> echo ok
dxconv remote setup --host npu1
dxconv remote check --host npu1
```

- `ls` : 파일이 보이면 키가 이미 있으니 `ssh-keygen`은 건너뜀 (없을 때만 실행, 엔터 3번)
- `ssh-copy-id` : `yes` 입력 후 장비 비밀번호 1번
- `ssh ... echo ok` : 비밀번호 없이 `ok`가 나오면 성공
- `remote check` : DX-RT 3.5.0, dxnn v9, 평가 python(자동 탐지), OpenCV 버전 확인 → 결과 `OK`

---

## 3. 실행

```bash
cd ~/dxconv
source .venv/bin/activate
dxconv convert --pt ~/fire.pt --remote npu1
dxconv convert --pt ~/fire.pt --images ~/fire_validation/images --labels ~/fire_validation/labels --remote npu1
```

- 첫 번째 : pt만 (기본 사진으로 양자화, NPU FPS 측정)
- 두 번째 : 이미지 + 라벨 (내 사진으로 양자화, PT 대비 정확도와 FPS 측정)
- 결과는 `~/dxconv/runs/` 아래 (`--out`으로 변경). 명령 설명 : [`commands.md`](commands.md)

---

## 4. Windows 파일 쓰기

| Windows 위치 | Ubuntu에서 경로 |
|---|---|
| `C:\Users\<사용자>\Desktop\fire.pt` | `/mnt/c/Users/<사용자>/Desktop/fire.pt` |
| 바탕화면이 OneDrive로 동기화된 경우 | `"/mnt/c/Users/<사용자>/OneDrive/바탕 화면/fire.pt"` (공백 → 따옴표) |
| `D:\data` | `/mnt/d/data` |

```bash
cp "/mnt/c/Users/<사용자>/OneDrive/바탕 화면/fire.pt" ~/
unzip "/mnt/c/Users/<사용자>/OneDrive/바탕 화면/fire_validation.zip" -d ~/
```

- 이미지는 **Ubuntu 홈으로 복사해서** 사용 (`/mnt/c/...`를 바로 읽으면 많이 느림)
- 데이터가 zip이면 두 번째 줄처럼 풀어서 사용

## 5. 결과 dxnn을 Windows로 꺼내기

```bash
cp ~/dxconv/runs/convert/<폴더>/<모델>.dxnn /mnt/c/Users/<사용자>/Desktop/
```

- Windows 탐색기에서 Ubuntu 홈 열기 : 주소창에 `\\wsl$\Ubuntu-24.04\home\<Ubuntu 사용자>`

---

## 6. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| 재부팅 후 `wsl -l -v`에 "설치된 배포가 없습니다" | 첫 `wsl --install`은 WSL 기능만 켬 | 같은 명령 한 번 더 |
| `libglib2.0-0` 설치 실패 | Ubuntu 24.04에서 이름이 `libglib2.0-0t64`로 바뀜 | `libglib2.0-0t64` |
| 설치한 명령이 `command not found` | 가상환경 미활성화, 또는 `~/.local/bin`이 PATH에 없음 | `source ~/dxconv/.venv/bin/activate`, 그래도 없으면 `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc` |
| `장비 접속 실패` + `Operation timed out` | `port` 미입력·오류, 주소 오류, 다른 네트워크 | `remote.yaml`의 `port`, `hostname` 확인 |
| `장비 접속 실패` + `Permission denied (publickey)` | 이 PC의 SSH 키가 장비에 미등록 | `ssh-copy-id -p <포트> <계정>@<장비 IP>` |
| `ssh-copy-id`가 `ssh_askpass: No such file` | 비밀번호 입력 창(터미널)이 없는 곳에서 실행 | 일반 Ubuntu 터미널에서 실행 |
| 바탕화면 파일이 `Desktop`에 없음 | OneDrive 바탕화면 리디렉션 | `ls /mnt/c/Users/<사용자>/OneDrive/` |
| 컴파일 중 `cosine similarity ... below 0.90` 경고 | end2end 출력(TopK)에서 나오는 DX-COM 경고 | 무시 (실제 mAP 하락으로 판단) |
| dxnn에 컴파일러 버전이 `2.5.0-rc.1`로 기록 | pip 패키지 dx-com 2.5.0의 내부 표기 | 정상 (v9 포맷 동일) |

---

## 7. 지켜야 할 것

- 장비 접속 정보는 `configs/remote.yaml`에만 (git 제외 파일). 문서·노트·커밋에 쓰지 않음
- SSH 개인 키(`~/.ssh/id_ed25519`, `.pub` 아닌 쪽)는 복사·공유하지 않음. 장비에는 공개 키만 등록됨
- 같은 NPU 장비를 여러 PC가 동시에 써도 됨 : 장비 측정은 잠금 파일로 한 번에 하나씩 (다른 쪽은 대기)
