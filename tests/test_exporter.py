from __future__ import annotations

from pathlib import Path

import pytest

from dxconv import DxconvError
from dxconv.compile.onnx_check import OnnxInfo, TensorInfo
from dxconv.config import ExportCfg, RunCfg
from dxconv.export.exporter import export_kwargs
from dxconv.pipeline import check_end2end_output

OLD_KEYS = {"end2end": None, "nms": False, "half": False}  # ultralytics 8.3.x
NEW_KEYS = {"nms": None, "quantize": None}  # ultralytics 8.4.x (end2end/half deprecated)


def test_old_api_passes_end2end():
    kw = export_kwargs(ExportCfg(), OLD_KEYS)
    assert kw["end2end"] is True and kw["nms"] is False and kw["half"] is False
    assert kw["format"] == "onnx" and kw["opset"] == 17


@pytest.mark.parametrize(
    ("end2end", "nms", "expected"),
    [(True, False, False), (False, False, None), (False, True, True)],
)
def test_new_api_maps_to_tristate_nms(end2end: bool, nms: bool, expected: bool | None):
    kw = export_kwargs(ExportCfg(end2end=end2end, nms=nms), NEW_KEYS)
    assert "end2end" not in kw and "half" not in kw
    assert kw["nms"] is expected


def test_unsupported_ultralytics():
    with pytest.raises(DxconvError, match="end2end"):
        export_kwargs(ExportCfg(), {"nms": False})


def _info(shape: list[int]) -> OnnxInfo:
    return OnnxInfo(path="x", opset=17, inputs=[], outputs=[TensorInfo("o", shape, "float32")])


def test_end2end_output_check():
    cfg = RunCfg(model="fire")
    assert check_end2end_output(_info([1, 300, 6]), cfg) == []
    assert check_end2end_output(_info([1, 84, 8400]), cfg)


@pytest.mark.integration
def test_real_export(tmp_path):
    """ultralytics 로 yolo26n.yaml 모델을 만들어 실제 export (가중치 다운로드 없음)."""
    from ultralytics import YOLO

    from dxconv.compile.onnx_check import check_onnx
    from dxconv.export.exporter import export_onnx
    from dxconv.export.inspect_pt import inspect_pt

    pt = tmp_path / "m.pt"
    YOLO("yolo26n.yaml").save(str(pt))
    assert inspect_pt(pt).end2end_supported
    onnx = export_onnx(pt, ExportCfg(), tmp_path / "out" / "m.onnx")
    info, _ = check_onnx(onnx, (11, 21))
    assert info.outputs[0].shape == [1, 300, 6]


def test_export_does_not_touch_pt_folder(tmp_path, monkeypatch):
    """ultralytics 는 pt 옆에 ONNX 를 쓴다 → 원본 폴더의 같은 이름 ONNX 를 건드리지 않아야 함."""
    import ultralytics

    from dxconv.export import exporter

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            self.path = path

        def export(self, **kw):
            out = Path(self.path).with_suffix(".onnx")
            out.write_bytes(b"new-onnx")
            return str(out)

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(exporter, "check_ultralytics_args", lambda: "x")
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.pt").write_bytes(b"pt")
    (src / "m.onnx").write_bytes(b"user-onnx")  # 사용자가 따로 둔 ONNX
    dest = exporter.export_onnx(src / "m.pt", ExportCfg(), tmp_path / "run" / "m.onnx")
    assert dest.read_bytes() == b"new-onnx"
    assert (src / "m.onnx").read_bytes() == b"user-onnx"
    assert sorted(p.name for p in src.iterdir()) == ["m.onnx", "m.pt"]
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["m.onnx"]  # 임시 폴더 정리
