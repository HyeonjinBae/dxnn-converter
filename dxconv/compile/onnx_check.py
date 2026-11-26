"""ONNX 모델 검사: checker, opset, 입출력 정보, 정적 shape 확인."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dxconv import DxconvError


@dataclass
class TensorInfo:
    name: str
    shape: list[int | str | None]
    dtype: str

    @property
    def is_static(self) -> bool:
        return all(isinstance(d, int) and d > 0 for d in self.shape)


@dataclass
class OnnxInfo:
    path: str
    opset: int | None
    inputs: list[TensorInfo] = field(default_factory=list)
    outputs: list[TensorInfo] = field(default_factory=list)
    producer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def primary_input(self) -> TensorInfo:
        return self.inputs[0]


def _tensor_info(vi: Any) -> TensorInfo:
    import onnx

    t = vi.type.tensor_type
    dims: list[int | str | None] = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append(None)
    try:
        dtype = onnx.helper.tensor_dtype_to_np_dtype(t.elem_type).name
    except Exception:  # noqa: BLE001
        dtype = str(t.elem_type)
    return TensorInfo(name=vi.name, shape=dims, dtype=dtype)


def inspect_onnx(path: str | Path, check: bool = True) -> OnnxInfo:
    """ONNX 를 로드해 정보를 추출한다 (shape 검증은 하지 않음)."""
    import onnx

    path = Path(path)
    if not path.is_file():
        raise DxconvError(f"ONNX 파일이 없습니다: {path}")
    if check:
        try:
            onnx.checker.check_model(str(path))
        except Exception as e:  # noqa: BLE001
            raise DxconvError(f"ONNX 검사 실패 (onnx.checker): {e}") from e
    model = onnx.load(str(path), load_external_data=False)
    initializers = {i.name for i in model.graph.initializer}
    opset = None
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            opset = int(imp.version)
    return OnnxInfo(
        path=str(path),
        opset=opset,
        inputs=[_tensor_info(i) for i in model.graph.input if i.name not in initializers],
        outputs=[_tensor_info(o) for o in model.graph.output],
        producer=f"{model.producer_name} {model.producer_version}".strip() or None,
    )


def check_onnx(
    path: str | Path,
    opset_range: tuple[int, int] | list[int] | None = None,
    expected_opset: int | None = None,
) -> tuple[OnnxInfo, list[str]]:
    """ONNX 검사. 동적 차원/opset 범위 위반 시 DxconvError. (info, warnings) 반환."""
    info = inspect_onnx(path)
    warnings: list[str] = []
    errors: list[str] = []
    if not info.inputs:
        errors.append("ONNX 입력이 없습니다.")
    for t in info.inputs + info.outputs:
        if not t.is_static:
            errors.append(f"동적 차원이 있습니다: {t.name} shape={t.shape} (NPU 는 정적 shape 필요)")
    if len(info.inputs) > 1:
        warnings.append(f"입력이 {len(info.inputs)} 개입니다. 첫 번째 입력만 캘리브레이션에 사용합니다.")
    if opset_range and info.opset is not None:
        lo, hi = opset_range
        if not lo <= info.opset <= hi:
            errors.append(f"ONNX opset {info.opset} 이(가) 지원 범위 [{lo}, {hi}] 밖입니다.")
    if expected_opset is not None and info.opset is not None and info.opset != expected_opset:
        warnings.append(f"ONNX opset({info.opset}) 이 설정값({expected_opset}) 과 다릅니다.")
    if errors:
        raise DxconvError("ONNX 검사 실패:\n" + "\n".join(f"  - {e}" for e in errors))
    return info, warnings
