"""dxconv: DEEPX NPU용 Ultralytics YOLO 모델 변환 자동화 CLI."""

from __future__ import annotations

__version__ = "0.1.0"


class DxconvError(Exception):
    """사용자에게 그대로 보여줄 수 있는 dxconv 오류 (CLI에서 종료 코드 1로 처리)."""

