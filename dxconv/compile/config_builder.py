"""DX-COM config.json 생성.

키 이름/구조는 버전별 어댑터(compile/adapters)에 위임하고, 여기서는
버전 독립적인 전처리 단계 목록을 만든다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dxconv import DxconvError
from dxconv.compile.onnx_check import OnnxInfo, TensorInfo
from dxconv.config import PreprocessCfg, RunCfg

if TYPE_CHECKING:
    from dxconv.compile.adapters.base import DxcomAdapter


def input_hw(t: TensorInfo, layout: str) -> tuple[int, int]:
    shape = t.shape
    if len(shape) != 4 or not t.is_static:
        raise DxconvError(f"입력 shape 이 4차원 정적 shape 가 아닙니다: {t.name} {shape}")
    if layout == "NCHW":
        return int(shape[2]), int(shape[3])  # type: ignore[arg-type]
    return int(shape[1]), int(shape[2])  # type: ignore[arg-type]


def preprocess_steps(pre: PreprocessCfg, height: int, width: int) -> list[dict[str, Any]]:
    """버전 독립적 전처리 단계 목록.

    이미지 로더는 OpenCV(BGR, HWC, uint8) 기준으로 가정한다.
    Ultralytics 추론 전처리: letterbox(114, 중앙 정렬) → BGR→RGB → /255 → HWC→CHW → batch 축.
    letterbox_mode=host 이면 캘리브레이션 이미지가 이미 letterbox 된 상태라 resize 단계를 넣지 않는다.
    """
    steps: list[dict[str, Any]] = []
    if pre.letterbox_mode == "dxcom":
        steps.append({"op": "letterbox", "height": height, "width": width,
                      "pad_value": pre.pad_value})
    elif pre.letterbox_mode == "none":
        steps.append({"op": "resize", "height": height, "width": width})
    if pre.color == "RGB":
        steps.append({"op": "bgr2rgb"})
    if pre.scale != 1.0:
        steps.append({"op": "div", "value": pre.scale})
    if pre.layout == "NCHW":
        steps.append({"op": "transpose", "axis": [2, 0, 1]})
    steps.append({"op": "expand_dim", "axis": 0})
    return steps


def build_dxcom_config(
    cfg: RunCfg, onnx_info: OnnxInfo, calib_dir: str | Path, adapter: DxcomAdapter
) -> dict[str, Any]:
    return adapter.build_config(cfg, onnx_info, Path(calib_dir))


def write_config(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def onnx_info_from_cfg(cfg: RunCfg, input_name: str = "images") -> OnnxInfo:
    """ONNX 없이(dry-run) 설정값으로 입력 정보를 만든다. Ultralytics 기본 입력 이름은 'images'."""
    e = cfg.export
    if cfg.compile.preprocess.layout == "NCHW":
        shape: list[int | str | None] = [e.batch, 3, e.imgsz, e.imgsz]
    else:
        shape = [e.batch, e.imgsz, e.imgsz, 3]
    return OnnxInfo(
        path="(dry-run: 미생성)",
        opset=e.opset,
        inputs=[TensorInfo(name=input_name, shape=shape, dtype="float32")],
    )
