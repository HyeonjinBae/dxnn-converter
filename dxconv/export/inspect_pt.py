"""Ultralytics .pt 파일 정보 추출."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dxconv import DxconvError


@dataclass
class PtInfo:
    path: str
    task: str | None
    num_classes: int | None
    names: dict[int, str] = field(default_factory=dict)
    train_imgsz: int | list[int] | None = None
    head_type: str | None = None
    end2end_supported: bool = False
    ultralytics_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def head_supports_end2end(head: Any) -> bool:
    """one2one(end2end) 분기를 가진 head 인지 판별.

    YOLOv10 / YOLO26 계열 Detect head 는 `end2end` 속성 또는 `one2one_cv2` 모듈을 가진다.
    """
    if getattr(head, "end2end", False):
        return True
    return hasattr(head, "one2one_cv2") or hasattr(head, "one2one")


def inspect_pt(pt_path: str | Path) -> PtInfo:
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        raise DxconvError(f"pt 파일이 없습니다: {pt_path}")
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as e:
        raise DxconvError(
            "ultralytics 가 설치되어 있지 않습니다: pip install ultralytics"
        ) from e
    try:
        yolo = YOLO(str(pt_path))
    except Exception as e:  # noqa: BLE001 - ultralytics 내부 예외 다양
        raise DxconvError(f"pt 파일을 로드할 수 없습니다: {pt_path} ({e})") from e

    net = getattr(yolo, "model", None)
    names = dict(getattr(yolo, "names", {}) or {})
    head = None
    try:
        head = net.model[-1]  # type: ignore[union-attr,index]
    except Exception:  # noqa: BLE001
        head = None
    args = getattr(net, "args", {}) or {}
    if not isinstance(args, dict):
        args = vars(args) if hasattr(args, "__dict__") else {}
    nc = getattr(head, "nc", None) if head is not None else None
    return PtInfo(
        path=str(pt_path),
        task=getattr(yolo, "task", None),
        num_classes=int(nc) if nc is not None else (len(names) or None),
        names={int(k): str(v) for k, v in names.items()},
        train_imgsz=args.get("imgsz"),
        head_type=type(head).__name__ if head is not None else None,
        end2end_supported=head_supports_end2end(head) if head is not None else False,
        ultralytics_version=getattr(ultralytics, "__version__", None),
    )
