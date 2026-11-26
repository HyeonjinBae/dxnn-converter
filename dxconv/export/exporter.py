"""Ultralytics YOLO → ONNX 내보내기."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from dxconv import DxconvError
from dxconv.config import ExportCfg

log = logging.getLogger(__name__)


def export_kwargs(cfg: ExportCfg, cfg_keys: dict[str, Any] | None = None) -> dict[str, Any]:
    """설치된 ultralytics 의 인자 체계에 맞춰 export 인자를 만든다.

    - 구버전(8.3.x): `end2end`, `nms`, `half` 가 각각 독립 인자.
    - 신버전(8.4.x~): `end2end` 는 deprecated 이고 `nms` 가 3상태로 통합됨
      (False: NMS-free end2end head, None: 외부 NMS(raw 출력), True: NMS 내장).
      `half` 도 deprecated(→ `quantize`) 이므로 False 일 때는 넘기지 않는다.
    cfg_keys 를 생략하면 설치된 ultralytics 의 DEFAULT_CFG_DICT 를 사용한다.
    """
    if cfg_keys is None:
        cfg_keys = _ultralytics_cfg_keys()[1]
    kw: dict[str, Any] = {
        "format": "onnx",
        "opset": cfg.opset,
        "simplify": cfg.simplify,
        "dynamic": cfg.dynamic,
        "max_det": cfg.max_det,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
    }
    if "end2end" in cfg_keys:
        kw["end2end"] = cfg.end2end
        kw["nms"] = cfg.nms
    elif "nms" in cfg_keys and cfg_keys["nms"] is None:
        kw["nms"] = False if cfg.end2end else (True if cfg.nms else None)
    else:
        raise DxconvError(
            "설치된 ultralytics 는 end2end export 를 지원하지 않습니다 "
            "('end2end' 인자 또는 3상태 'nms' 인자 없음). "
            "pip install -U ultralytics 로 업그레이드하세요."
        )
    if "half" in cfg_keys:
        kw["half"] = cfg.half
    elif cfg.half:
        kw["quantize"] = 16  # rules.py 에서 이미 막지만 의미 보존
    return kw


def _ultralytics_cfg_keys() -> tuple[str, dict[str, Any]]:
    try:
        import ultralytics
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except ImportError as e:
        raise DxconvError("ultralytics 가 설치되어 있지 않습니다: pip install -U ultralytics") from e
    return getattr(ultralytics, "__version__", "unknown"), dict(DEFAULT_CFG_DICT)


def check_ultralytics_args() -> str:
    """설치된 ultralytics 가 end2end export 를 지원하는지 확인. 버전 문자열 반환."""
    ver, keys = _ultralytics_cfg_keys()
    try:
        export_kwargs(ExportCfg(), keys)
    except DxconvError as e:
        raise DxconvError(f"ultralytics {ver}: {e}") from e
    return ver


def export_onnx(pt_path: str | Path, cfg: ExportCfg, dest: str | Path) -> Path:
    """pt 를 ONNX 로 내보내고 dest 경로로 이동한다.

    ultralytics 는 ONNX 를 pt 와 같은 폴더에 쓴다. 원본 pt 옆의 같은 이름 ONNX 를 덮어쓰거나
    (읽기 전용 폴더에서) 실패하지 않도록, dest 옆 임시 폴더에 pt 를 복사해서 export 한다.
    """
    pt_path = Path(pt_path)
    dest = Path(dest)
    check_ultralytics_args()
    from ultralytics import YOLO

    work = dest.parent / "_export_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        local_pt = work / pt_path.name
        shutil.copy2(pt_path, local_pt)
        try:
            out = YOLO(str(local_pt)).export(**export_kwargs(cfg))
        except Exception as e:  # noqa: BLE001
            raise DxconvError(f"ONNX export 실패: {e}") from e
        if not out or not Path(out).is_file():
            raise DxconvError(f"ONNX export 결과 파일을 찾을 수 없습니다: {out}")
        shutil.move(str(out), dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    log.info("ONNX 생성: %s", dest)
    return dest
