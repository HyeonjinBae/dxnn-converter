"""DX-COM 2.2.x 어댑터.

2.2.0-rc.2 는 dxnn v8 을 만든다 (업체 배포 모델 parse_model 로 확인). 컴파일 자체는 미검증이라
2.5 어댑터를 그대로 쓴다. v8 이 필요하면 검증된 2.4.1 사용 권장.
"""

from __future__ import annotations

from dxconv.compile.adapters.v2_5 import DxcomV25Adapter


class DxcomV22Adapter(DxcomV25Adapter):
    series = "2.2"
