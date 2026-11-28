"""DX-COM / DX-RT 설치 탐지.

모든 탐지 단계는 실패해도 예외를 던지지 않고 다음 단계로 진행한다.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: 버전 문자열 패턴 (예: 2.5.0, 2.4.1-rc.1, v3.0.0)
VERSION_RE = re.compile(r"v?(\d+\.\d+\.\d+(?:[-_.]?(?:rc|a|b|alpha|beta)[-_.]?\d+)?)", re.I)

DXCOM_PIP_NAMES = ("dx-com", "dx_com")
DXCOM_EXECUTABLES = ("dxcom", "dx_com")
DXCOM_VERSION_FLAGS = ("-v", "--version")  # dxcom --help(2.5.0)로 확인: "DX-COM (DEEPX Compiler) 2.5.0"
DXRT_CLI = "dxrt-cli"
DXRT_STATUS_ARGS = ("--status",)

_SUBPROCESS_TIMEOUT = 15


@dataclass
class EnvInfo:
    """탐지 결과.

    dxcom: [{source, version, path}] (source: pip | path)
    dxrt:  {version, driver, firmware, raw} 또는 None (NPU 없음)
    """

    dxcom: list[dict[str, Any]] = field(default_factory=list)
    dxrt: dict[str, Any] | None = None
    npu_present: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_version(text: str) -> str | None:
    m = VERSION_RE.search(text or "")
    return m.group(1) if m else None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("명령 실행 실패 %s: %s", cmd, e)
        return None


def detect_dxcom_pip() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for name in DXCOM_PIP_NAMES:
        try:
            ver = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as e:  # noqa: BLE001 - 메타데이터 손상 등
            log.debug("pip 메타데이터 조회 실패(%s): %s", name, e)
            continue
        path = None
        try:
            dist = importlib.metadata.distribution(name)
            path = str(dist.locate_file(""))
        except Exception:  # noqa: BLE001
            pass
        found.append({"source": "pip", "version": ver, "path": path})
        break  # dx-com / dx_com 은 같은 패키지
    return found


def detect_dxcom_path() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for exe in DXCOM_EXECUTABLES:
        path = shutil.which(exe)
        if not path or path in seen:
            continue
        seen.add(path)
        version = None
        for flag in DXCOM_VERSION_FLAGS:
            proc = _run([path, flag])
            if proc is None:
                continue
            version = parse_version((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if version:
                break
        found.append({"source": "path", "version": version, "path": path})
    return found


def parse_dxrt_status(text: str) -> dict[str, Any]:
    """dxrt-cli --status 출력에서 버전 정보를 뽑는다.

    이 PC 에 DX-RT 가 있을 때만 쓰임 (참고 정보). 'DXRT v3.5.0', 'RT Driver version : v2.7.0',
    'FW version : v2.7.6' 형태의 줄에서 버전을 뽑는다. NPU 장비 점검은 remote.client.check 가 따로 한다.
    """
    info: dict[str, Any] = {"version": None, "driver": None, "firmware": None}
    for line in text.splitlines():
        low = line.lower()
        ver = parse_version(line)
        if not ver:
            continue
        if "firmware" in low or "fw" in low.split():
            info["firmware"] = info["firmware"] or ver
        elif "driver" in low:
            info["driver"] = info["driver"] or ver
        elif "dxrt" in low or "runtime" in low or "rt version" in low:
            info["version"] = info["version"] or ver
    return info


def detect_dxrt() -> tuple[dict[str, Any] | None, bool]:
    path = shutil.which(DXRT_CLI)
    if path is None:
        return None, False
    proc = _run([path, *DXRT_STATUS_ARGS])
    if proc is None:
        return None, False
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    info = parse_dxrt_status(text)
    info["path"] = path
    info["raw"] = text.strip()[:2000]
    low = text.lower()
    # 장치 존재: 정상 종료 + "no device" 등 오류 문구 없음 (참고 정보, 변환에는 쓰지 않음)
    npu = proc.returncode == 0 and not any(
        k in low for k in ("no device", "not found", "failed", "error")
    )
    return info, npu


def detect_env() -> EnvInfo:
    """DX-COM / DX-RT 를 모든 방법으로 탐지한다. 예외를 던지지 않는다."""
    env = EnvInfo()
    for fn in (detect_dxcom_pip, detect_dxcom_path):
        try:
            env.dxcom.extend(fn())
        except Exception as e:  # noqa: BLE001 - 탐지는 절대 실패하지 않음
            env.notes.append(f"{fn.__name__} 실패: {e}")
    try:
        env.dxrt, env.npu_present = detect_dxrt()
    except Exception as e:  # noqa: BLE001
        env.notes.append(f"detect_dxrt 실패: {e}")
    if not env.dxcom:
        env.notes.append(
            "DX-COM 미설치: `./scripts/install_host.sh` 또는 `pip install dx-com==2.5.0` "
            "(Linux x86-64 / Windows WSL2 전용, README 설치)."
        )
    if env.dxrt is None:
        env.notes.append("DX-RT 없음: 이 PC에는 NPU 런타임이 없습니다 (평가는 원격 장비 사용).")
    return env
