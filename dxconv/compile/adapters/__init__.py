"""버전 문자열 → DX-COM 어댑터 선택."""

from __future__ import annotations

import logging

from packaging.version import InvalidVersion, Version

from dxconv import DxconvError
from dxconv.compile.adapters.base import DxcomAdapter
from dxconv.compile.adapters.v2_2 import DxcomV22Adapter
from dxconv.compile.adapters.v2_4 import DxcomV24Adapter
from dxconv.compile.adapters.v2_5 import DxcomV25Adapter

log = logging.getLogger(__name__)

ADAPTERS: dict[tuple[int, int], type[DxcomAdapter]] = {
    (2, 2): DxcomV22Adapter,
    (2, 4): DxcomV24Adapter,
    (2, 5): DxcomV25Adapter,
}


def get_adapter(version: str) -> DxcomAdapter:
    """`2.5.x` → v2_5, `2.4.x` → v2_4, `2.2.x` → v2_2.

    알려진 계열보다 새로운 버전은 가장 최신 어댑터로 처리하되 경고한다.
    """
    try:
        v = Version(str(version).lstrip("vV"))
    except InvalidVersion as e:
        raise DxconvError(f"DX-COM 버전 형식이 올바르지 않습니다: {version!r}") from e
    key = (v.major, v.minor)
    if key in ADAPTERS:
        return ADAPTERS[key](str(version))
    newest = max(ADAPTERS)
    if key > newest:
        log.warning("DX-COM %s 전용 어댑터가 없어 %d.%d 어댑터를 사용합니다.", version, *newest)
        return ADAPTERS[newest](str(version))
    supported = ", ".join(f"{a}.{b}.x" for a, b in sorted(ADAPTERS))
    raise DxconvError(f"지원하지 않는 DX-COM 버전입니다: {version} (지원: {supported})")


__all__ = ["DxcomAdapter", "get_adapter"]
