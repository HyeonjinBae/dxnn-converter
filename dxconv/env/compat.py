"""DX-COM 버전 선택 및 호환성 검사."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from dxconv.config import load_yaml
from dxconv.env.detect import EnvInfo

#: 20채널 실험 기준 DX-COM 버전
REFERENCE_DXCOM_VERSION = "2.5.0"


def load_compat(config_dir: Path) -> dict[str, Any]:
    return load_yaml(config_dir / "compat.yaml")


def to_version(v: str | None) -> Version | None:
    if not v:
        return None
    try:
        return Version(str(v).lstrip("vV"))
    except InvalidVersion:
        return None


def same_version(a: str | None, b: str | None) -> bool:
    va, vb = to_version(a), to_version(b)
    return va is not None and va == vb


@dataclass
class CompatResult:
    version: str
    source: str  # cli | detected | config
    warnings: list[str] = field(default_factory=list)
    compat_entry: dict[str, Any] | None = None


def select_dxcom_version(
    requested: str | None, env: EnvInfo, config_version: str | None
) -> tuple[str, str]:
    """사용할 DX-COM 버전을 결정한다.

    --dxcom-version 지정 > 설정 파일의 compile.dxcom_version (목표 버전) > 탐지된 버전 중 최신.
    여러 버전(예: docker 이미지 dxcom:2.5.0, dxcom:2.4.1)이 있을 수 있으므로 설정의 목표 버전을 우선한다.
    """
    if requested:
        return requested, "cli"
    if config_version:
        return config_version, "config"
    detected = [d["version"] for d in env.dxcom if to_version(d.get("version"))]
    if detected:
        return max(detected, key=lambda v: to_version(v)), "detected"  # type: ignore[arg-type,return-value]
    return REFERENCE_DXCOM_VERSION, "default"


def version_for_format(fmt: int, compat: dict[str, Any]) -> str:
    """--dxnn-format 8|9 → 대표 DX-COM 버전 (compat.yaml formats)."""
    from dxconv import DxconvError

    table = {int(k): str(v) for k, v in (compat.get("formats") or {}).items()}
    if fmt not in table:
        raise DxconvError(f"지원하지 않는 dxnn 포맷: v{fmt} (가능: {', '.join(f'v{k}' for k in sorted(table))})")
    return table[fmt]


def version_for_dxrt(dxrt: str, compat: dict[str, Any]) -> tuple[str, int]:
    """대상 장비 DX-RT 버전 → 읽을 수 있는 가장 새 dxnn 포맷과 그 대표 DX-COM 버전.

    compat.yaml 의 formats(포맷 → 대표 DX-COM 버전)와 dxcom.<버전>.min_dxrt 로 판단한다.
    """
    from dxconv import DxconvError

    have = to_version(dxrt)
    if have is None:
        raise DxconvError(f"DX-RT 버전을 해석할 수 없습니다: {dxrt!r}")
    ok = []
    for fmt, ver in (compat.get("formats") or {}).items():
        entry = find_compat_entry(str(ver), compat) or {}
        need = to_version(entry.get("min_dxrt"))
        if need is not None and have >= need:
            ok.append((int(fmt), str(ver)))
    if not ok:
        raise DxconvError(f"DX-RT {dxrt} 가 읽을 수 있는 dxnn 포맷을 만드는 DX-COM 이 compat.yaml 에 없습니다.")
    fmt, ver = max(ok)
    return ver, fmt


def find_compat_entry(version: str, compat: dict[str, Any]) -> dict[str, Any] | None:
    for key, entry in (compat.get("dxcom") or {}).items():
        if same_version(str(key), version):
            return entry or {}
    return None


def check_compat(
    requested: str | None,
    env: EnvInfo,
    compat: dict[str, Any],
    config_version: str | None = None,
) -> CompatResult:
    version, source = select_dxcom_version(requested, env, config_version)
    res = CompatResult(version=version, source=source)

    if to_version(version) is None:
        res.warnings.append(f"DX-COM 버전 문자열을 해석할 수 없습니다: {version!r}")
    if not same_version(version, REFERENCE_DXCOM_VERSION):
        res.warnings.append(
            f"DX-COM {version} 사용: 20채널 실험 기준 버전은 {REFERENCE_DXCOM_VERSION} 입니다. "
            "결과(정확도/FPS)가 다를 수 있습니다."
        )
    if not env.dxcom:
        res.warnings.append(
            f"DX-COM 이 탐지되지 않았습니다. 설정값({version})을 기준으로 진행합니다."
        )
    if source in ("cli", "config") and env.dxcom and not any(
        same_version(d.get("version"), version) for d in env.dxcom
    ):
        found = ", ".join(str(d.get("version")) for d in env.dxcom)
        res.warnings.append(
            f"요청한 DX-COM {version} 이(가) 탐지된 버전({found})에 없습니다."
        )

    entry = find_compat_entry(version, compat)
    res.compat_entry = entry
    if entry is None:
        res.warnings.append(f"compat.yaml 에 DX-COM {version} 정보가 없습니다.")
    else:
        min_dxrt = entry.get("min_dxrt")
        fmt = entry.get("dxnn_format")
        if min_dxrt is None:
            res.warnings.append(f"DX-COM {version} 의 최소 DX-RT 버전이 compat.yaml 에 없습니다.")
        if env.dxrt and min_dxrt:
            local = to_version(env.dxrt.get("version"))
            need = to_version(min_dxrt)
            if local is None:
                res.warnings.append("로컬 DX-RT 버전을 확인할 수 없어 호환성 검사를 건너뜁니다.")
            elif need is not None and local < need:
                res.warnings.append(
                    f"로컬 DX-RT {local} < {need}: DX-COM {version} 이 만든 dxnn"
                    f"(format v{fmt}) 을 로드할 수 없습니다. DX-RT 를 업그레이드하세요."
                )
    return res
