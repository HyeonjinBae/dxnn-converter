"""sweep 캐시 키.

키 = sha256( 정렬된 JSON( 컴파일 결과를 바꾸는 모든 입력 ) ) 앞 16자.
입력: export/compile 설정 전체, pt sha256, 캘리브레이션 입력 서명(목록 sha256 또는 폴더·시드·샘플링),
DX-COM 버전, CACHE_VERSION. 단계(A/B/C)는 넣지 않는다 → 단계가 달라도 같은 조합이면 재사용.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: 결과 형식이 바뀌면 올려서 기존 캐시를 무효화한다.
CACHE_VERSION = 1


def cache_key(params: dict[str, Any]) -> str:
    blob = json.dumps({"v": CACHE_VERSION, **params}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
