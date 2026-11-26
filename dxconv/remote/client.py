"""원격 NPU 장비 클라이언트 (시스템 OpenSSH: ssh / scp, 키 인증).

- 설정: configs/remote.yaml 의 hosts.<이름>. 필수는 hostname, username 뿐이고 나머지는 기본값/자동 탐지:
    port 22, key_filename (ssh 기본 키), workdir "dxconv_agent" (장비 홈 기준 상대 경로),
    python "auto" (dx_engine 을 import 할 수 있는 python 을 장비에서 찾음), dxrt_bin (DX-RT 실행 파일 폴더,
    PATH 에 없을 때만), lock_file /tmp/dxconv_eval.lock
  → 장비 계정·경로를 코드에 두지 않는다. 장비마다 다른 값은 remote.yaml 에만 둔다.
- 장비 쪽 구조 (workdir):
    agent/            device_agent/*.py (매번 동기화)
    models/<sha16>.dxnn
    data/<키>/images/ 평가 이미지 (분할 목록에 있는 것만, 없는 파일만 업로드)
    data/<키>/<split>.txt
    results/          eval_map / bench_fps 결과 JSON
- 측정은 `flock <lock_file>` 안에서 실행해 동시 측정을 막는다 (프로세스가 끝나면 자동 해제).
- 비밀번호 인증은 지원하지 않는다 (BatchMode). `ssh-copy-id` 로 키를 등록한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from dxconv import DxconvError
from dxconv.config import ConfigError, load_yaml

AGENT_DIR = Path(__file__).resolve().parents[2] / "device_agent"
AGENT_FILES = ("eval_map.py", "bench_fps.py", "check_env.py")


class HostCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hostname: str                      # 장비 IP 또는 호스트 이름
    port: int = 22                     # SSH 포트
    username: str                      # 장비 로그인 계정
    key_filename: str | None = None    # SSH 개인 키 (없으면 ssh 기본 키 / ~/.ssh/config)
    workdir: str = "dxconv_agent"      # 장비 작업 폴더. 상대 경로면 장비 계정 홈 기준
    python: str = "auto"               # dx_engine 이 있는 python. auto: 장비에서 자동 탐지
    dxrt_bin: str | None = None        # dxrt-cli / run_model 이 있는 폴더 (PATH 에 없을 때만)
    lock_file: str = "/tmp/dxconv_eval.lock"


# python: auto 일 때 장비에서 차례로 확인하는 후보 (dx_engine import 가 되는 첫 번째)
PYTHON_CANDIDATES = (
    "python3",
    "$HOME/dx-all-suite*/dx-runtime/venv-dx-runtime/bin/python",
    "$HOME/*/dx-runtime/venv-dx-runtime/bin/python",
    "$HOME/dx_rt/venv*/bin/python",
    "$HOME/*/venv-dx-runtime/bin/python",
)


def pick_python(rows: list[list[str]]) -> str:
    """후보 [(경로, OpenCV, dx_engine)] 중: PC 와 OpenCV 가 같은 것 (letterbox 가 같아야 mAP 가 같음)
    → 없으면 dx_engine 버전이 가장 높은 것 → 같으면 먼저 찾은 것."""
    try:
        import cv2

        host_cv = cv2.__version__
    except ImportError:
        host_cv = None
    same = [r for r in rows if r[1] == host_cv]
    if same:
        return same[0][0]

    def ver(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", v)[:3])

    return max(rows, key=lambda r: (ver(r[2]), -rows.index(r)))[0]


def load_host(name: str, config_dir: Path) -> HostCfg:
    path = config_dir / "remote.yaml"
    if not path.is_file():
        raise ConfigError(f"원격 장비 설정이 없습니다: {path} (configs/remote.example.yaml 을 복사해 채우세요)")
    hosts = (load_yaml(path) or {}).get("hosts") or {}
    if name not in hosts:
        raise ConfigError(f"remote.yaml 에 '{name}' 호스트가 없습니다. 사용 가능: {', '.join(hosts) or '(없음)'}")
    try:
        return HostCfg.model_validate(hosts[name])
    except ValidationError as e:
        raise ConfigError(f"remote.yaml hosts.{name} 오류:\n{e}") from e


def _vtuple(v: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3]) or (0,)


# DX-RT 버전 → 읽을 수 있는 dxnn 포맷 (DX-RT RELEASE_NOTES: v8 은 3.1.0, v9 는 3.5.0 부터)
DXRT_FORMATS = ((3, 5, 0), 9), ((3, 1, 0), 8), ((0,), 7)


def supported_formats(dxrt: str | None) -> list[int]:
    v = _vtuple(dxrt)
    top = next((f for minv, f in DXRT_FORMATS if v >= minv), 7)
    return list(range(6, top + 1))


@dataclass
class RemoteClient:
    name: str
    cfg: HostCfg
    timeout: float = 3600.0
    _uploaded: set[str] = field(default_factory=set)
    _python: str | None = None

    # ------------------------------------------------------------ 기본
    @property
    def target(self) -> str:
        return f"{self.cfg.username}@{self.cfg.hostname}"

    def _ssh_base(self) -> list[str]:
        cmd = ["ssh", "-p", str(self.cfg.port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if self.cfg.key_filename:
            cmd += ["-i", str(Path(self.cfg.key_filename).expanduser())]
        return cmd + [self.target]

    def run(self, script: str, *, timeout: float | None = None, check: bool = True,
            stdin: bytes | None = None) -> str:
        """장비에서 bash 스크립트 실행 → stdout."""
        prefix = "export PATH=" + (f"{shlex.quote(self.cfg.dxrt_bin)}:" if self.cfg.dxrt_bin else "") \
            + "$HOME/.local/bin:/usr/local/bin:$PATH; "
        try:
            p = subprocess.run([*self._ssh_base(), "bash -c " + shlex.quote(prefix + script)], input=stdin,
                               capture_output=True, timeout=timeout or self.timeout)
        except subprocess.TimeoutExpired as e:
            raise DxconvError(f"[{self.name}] 원격 명령 타임아웃: {script[:80]}") from e
        out = p.stdout.decode("utf-8", "replace")
        if check and p.returncode != 0:
            err = p.stderr.decode("utf-8", "replace").strip().splitlines()
            err = [ln for ln in err if "unable to resolve host" not in ln][-8:]
            raise DxconvError(f"[{self.name}] 원격 명령 실패 (코드 {p.returncode}): {script[:120]}\n"
                              + "\n".join(err))
        return out

    def put(self, local: Path, remote: str) -> None:
        cmd = ["scp", "-q", "-P", str(self.cfg.port), "-o", "BatchMode=yes"]
        if self.cfg.key_filename:
            cmd += ["-i", str(Path(self.cfg.key_filename).expanduser())]
        p = subprocess.run([*cmd, str(local), f"{self.target}:{remote}"], capture_output=True,
                           timeout=self.timeout)
        if p.returncode != 0:
            raise DxconvError(f"[{self.name}] 업로드 실패: {local} → {remote}\n"
                              f"{p.stderr.decode('utf-8', 'replace')[-500:]}")

    def read_json(self, remote: str) -> Any:
        return json.loads(self.run(f"cat {shlex.quote(remote)}"))

    @property
    def wd(self) -> str:
        return self.cfg.workdir.rstrip("/") or "."

    @property
    def python(self) -> str:
        """평가 python. 설정이 auto 면 장비에서 dx_engine 을 import 할 수 있는 python 을 찾아 기억한다."""
        if self.cfg.python != "auto":
            return self.cfg.python
        if self._python is None:
            # dx_engine 을 import 할 수 있는 후보를 모두 찾고 (경로, OpenCV, dx_engine 버전)
            probe = ("seen=; for p in " + " ".join(PYTHON_CANDIDATES) + "; do "
                     "q=$(command -v \"$p\" 2>/dev/null) || continue; "
                     "case \" $seen \" in *\" $q \"*) continue;; esac; seen=\"$seen $q\"; "
                     "v=$(\"$q\" -c 'import dx_engine,cv2;print(cv2.__version__, dx_engine.__version__)' "
                     "2>/dev/null) && echo \"$q $v\"; done; true")
            rows = [ln.split() for ln in self.run(probe, check=False, timeout=60).splitlines()
                    if len(ln.split()) == 3]
            if not rows:
                raise DxconvError(
                    f"[{self.name}] 장비에서 dx_engine 이 있는 python 을 찾지 못했습니다. "
                    "configs/remote.yaml 의 python 에 경로를 적으세요 "
                    "(장비에서 확인: ls ~/dx-all-suite*/dx-runtime/venv-dx-runtime/bin/python).")
            self._python = pick_python(rows)
        return self._python

    # ------------------------------------------------------------ 점검
    def check(self) -> dict[str, Any]:
        try:
            py_path = self.python
        except DxconvError as e:  # 자동 탐지 실패 → 점검 결과의 problems 로 보고
            py_path, self._py_error = "python3", str(e)
        py = shlex.quote(py_path)
        out = self.run(
            "echo __HOST__; hostname; uname -m; "
            "echo __STATUS__; dxrt-cli --status 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g'; "
            f"echo __PY__; {py} -c 'import dx_engine,cv2,numpy,sys;"
            "print(dx_engine.__version__, cv2.__version__, numpy.__version__, sys.version.split()[0])' 2>&1 | tail -1; "
            f"echo __DISK__; df -Pk {shlex.quote(self.wd)} 2>/dev/null | tail -1 || df -Pk ~ | tail -1; "
            f"echo __LOCK__; (flock -n {shlex.quote(self.cfg.lock_file)} true && echo free) || echo busy",
            check=False, timeout=60)
        sec: dict[str, list[str]] = {}
        cur = None
        for ln in out.splitlines():
            if ln.startswith("__") and ln.endswith("__"):
                cur = ln.strip("_").lower()
                sec[cur] = []
            elif cur:
                sec[cur].append(ln)
        status = "\n".join(sec.get("status", []))

        def grab(pat: str) -> str | None:
            m = re.search(pat, status)
            return m.group(1) if m else None

        py_parts = (sec.get("py") or [""])[-1].split()
        ok_py = len(py_parts) == 4 and not py_parts[0].endswith(":")
        disk = (sec.get("disk") or [""])[-1].split()
        info = {
            "host": self.name, "target": f"{self.target}:{self.cfg.port}",
            "hostname": (sec.get("host") or [None])[0],
            "arch": (sec.get("host") or [None, None])[1],
            "dxrt": grab(r"DXRT v([\d.]+)"),
            "rt_driver": grab(r"RT Driver version\s*:\s*v([\d.]+)"),
            "pcie_driver": grab(r"PCIe Driver version\s*:\s*v([\d.]+)"),
            "firmware": grab(r"FW version\s*:\s*v([\d.]+)"),
            "device": grab(r"Device 0:\s*([^\n]+)"),
            "npu_cores": len(re.findall(r"^NPU \d+:", status, re.M)),
            "dx_engine": py_parts[0] if ok_py else None,
            "opencv": py_parts[1] if ok_py else None,
            "numpy": py_parts[2] if ok_py else None,
            "python": py_parts[3] if ok_py else None,
            "python_error": None if ok_py else " ".join(py_parts)[:200],
            "python_path": py_path,
            "python_source": "remote.yaml" if self.cfg.python != "auto" else "자동 탐지",
            "workdir": self.wd,
            "disk_free_gb": round(int(disk[3]) / 1024 / 1024, 1) if len(disk) >= 4 and disk[3].isdigit() else None,
            "lock": (sec.get("lock") or ["?"])[-1],
        }
        info["dxnn_formats"] = supported_formats(info["dxrt"]) if info["dxrt"] else []
        problems = []
        if not info["dxrt"]:
            problems.append("DX-RT(dxrt-cli)를 찾지 못했습니다.")
        if getattr(self, "_py_error", None):
            problems.append(self._py_error)
        elif not ok_py:
            problems.append(f"평가 python({py_path})에서 dx_engine/cv2/numpy import 실패: "
                            f"{info['python_error']}")
        info["problems"] = problems
        info["ok"] = not problems
        return info

    def setup(self) -> None:
        """작업 폴더와 device_agent 동기화."""
        self.run(f"mkdir -p {self.wd}/agent {self.wd}/models {self.wd}/data {self.wd}/results")
        for f in AGENT_FILES:
            if (AGENT_DIR / f).is_file():
                self.put(AGENT_DIR / f, f"{self.wd}/agent/{f}")

    # ------------------------------------------------------------ 파일
    def upload_dxnn(self, dxnn: Path, sha256: str | None = None) -> str:
        sha = sha256 or hashlib.sha256(dxnn.read_bytes()).hexdigest()
        remote = f"{self.wd}/models/{sha[:16]}.dxnn"
        if remote in self._uploaded:
            return remote
        exists = self.run(f"test -f {remote} && stat -c %s {remote} || echo none", check=False).strip()
        if exists != str(dxnn.stat().st_size):
            self.put(dxnn, remote + ".part")
            self.run(f"mv {remote}.part {remote}")
        self._uploaded.add(remote)
        return remote

    def sync_images(self, key: str, images_dir: Path, names: list[str], split: str) -> tuple[str, str]:
        """분할 이미지를 장비 data/<key>/images/ 로 (없거나 크기가 다른 파일만). (images_dir, list) 반환."""
        rdir = f"{self.wd}/data/{key}"
        listing = self.run(f"mkdir -p {rdir}/images && cd {rdir}/images && "
                           "find . -type f -printf '%P\\t%s\\n'", check=False)
        have = dict(ln.split("\t", 1) for ln in listing.splitlines() if "\t" in ln)
        missing = [n for n in names if have.get(n) != str((images_dir / n).stat().st_size)]
        if missing:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w", dereference=True) as tar:  # 심볼릭 링크는 실제 파일로
                for n in missing:
                    tar.add(images_dir / n, arcname=n)
            self.run(f"tar xf - -C {rdir}/images", stdin=buf.getvalue(), timeout=self.timeout)
        lst = f"{rdir}/{split}.txt"
        self.run(f"cat > {lst}", stdin=("\n".join(names) + "\n").encode())
        return f"{rdir}/images", lst

    # ------------------------------------------------------------ 측정
    def _locked(self, cmd: str) -> str:
        return f"flock -w {int(self.timeout)} {shlex.quote(self.cfg.lock_file)} {cmd}"

    def eval_map(self, dxnn_remote: str, images: str, lst: str, out_name: str,
                 conf: float = 0.001) -> dict[str, Any]:
        out = f"{self.wd}/results/{out_name}"
        py = shlex.quote(self.python)
        self.run(self._locked(f"{py} {self.wd}/agent/eval_map.py --dxnn {dxnn_remote} --images-dir {images} "
                              f"--list {lst} --conf {conf} --out {out}"))
        return self.read_json(out)

    def bench_fps(self, dxnn_remote: str, out_name: str, seconds: float = 10.0) -> dict[str, Any]:
        out = f"{self.wd}/results/{out_name}"
        py = shlex.quote(self.python)
        self.run(self._locked(f"{py} {self.wd}/agent/bench_fps.py --dxnn {dxnn_remote} "
                              f"--seconds {seconds} --out {out}"))
        return self.read_json(out)


def fps_value(bench: dict[str, Any]) -> float | None:
    """선택 기준 FPS: run_model 최대 처리량(코어 모두 사용) 우선, 없으면 동기 추론."""
    rm = bench.get("run_model") or {}
    if rm.get("fps"):
        return float(rm["fps"])
    sync = bench.get("sync") or {}
    return float(sync["fps"]) if sync.get("fps") else None
