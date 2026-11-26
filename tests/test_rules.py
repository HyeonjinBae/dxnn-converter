from __future__ import annotations

import pytest

from dxconv.config import ExportCfg, PostCfg
from dxconv.export.inspect_pt import PtInfo
from dxconv.export.rules import RuleError, check_export_rules

COMPAT = {"opset_range": [11, 21]}


def _errors(**kw: object) -> list[str]:
    return check_export_rules(ExportCfg(**kw), PostCfg(), COMPAT).errors


def test_defaults_pass():
    r = check_export_rules(ExportCfg(), PostCfg(), COMPAT)
    assert r.ok and r.errors == []


def test_half_is_error():
    assert any("half" in e for e in _errors(half=True))


@pytest.mark.parametrize("opset", [10, 22])
def test_opset_out_of_range(opset: int):
    assert any("opset" in e for e in _errors(opset=opset))


def test_opset_range_from_compat():
    r = check_export_rules(ExportCfg(opset=17), PostCfg(), {"opset_range": [11, 16]})
    assert any("opset" in e for e in r.errors)


def test_dynamic_is_error():
    assert any("dynamic" in e for e in _errors(dynamic=True))


def test_batch_is_error():
    assert any("batch" in e for e in _errors(batch=4))


def test_nms_and_end2end_is_error():
    assert any("nms" in e for e in _errors(nms=True, end2end=True))
    assert _errors(nms=True, end2end=False) == []


def test_imgsz_multiple_of_32():
    assert any("imgsz" in e for e in _errors(imgsz=650))
    assert _errors(imgsz=672) == []


def test_max_det_mismatch_is_warning_only():
    r = check_export_rules(ExportCfg(max_det=300), PostCfg(max_det=100), COMPAT)
    assert r.ok
    assert any("max_det" in w for w in r.warnings)


def test_end2end_nondefault_max_det_warns():
    r = check_export_rules(ExportCfg(max_det=100), PostCfg(), COMPAT)
    assert r.ok
    assert any("max_det" in w for w in r.warnings)


def test_end2end_unsupported_head_is_error():
    info = PtInfo(path="x.pt", task="detect", num_classes=2, head_type="Detect",
                  end2end_supported=False)
    r = check_export_rules(ExportCfg(end2end=True), PostCfg(), COMPAT, info)
    assert any("end2end" in e for e in r.errors)
    r2 = check_export_rules(ExportCfg(end2end=False), PostCfg(), COMPAT, info)
    assert r2.ok


def test_raise_if_error_collects_all():
    r = check_export_rules(ExportCfg(half=True, dynamic=True), PostCfg(), COMPAT)
    with pytest.raises(RuleError) as ei:
        r.raise_if_error()
    assert len(ei.value.errors) == 2
