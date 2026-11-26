from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from conftest import make_onnx

from dxconv import DxconvError
from dxconv.compile.adapters import get_adapter
from dxconv.compile.adapters.v2_2 import DxcomV22Adapter
from dxconv.compile.adapters.v2_4 import DxcomV24Adapter
from dxconv.compile.adapters.v2_5 import DxcomV25Adapter
from dxconv.compile.config_builder import build_dxcom_config, onnx_info_from_cfg, preprocess_steps
from dxconv.compile.onnx_check import check_onnx
from dxconv.config import PreprocessCfg, resolve_config

SNAPSHOT = Path(__file__).parent / "fixtures" / "dxcom_config_v2_5.json"
PLACEHOLDER = "<CALIB_DIR>"


def _build(config_dir: Path, tmp_path: Path, model: str = "fire", **kw: object) -> dict:
    cfg = resolve_config(model, config_dir=config_dir, **kw)
    onnx_path = make_onnx(tmp_path / "m.onnx")
    info, _ = check_onnx(onnx_path, (11, 21))
    calib = tmp_path / "calib_subset"
    calib.mkdir()
    d = build_dxcom_config(cfg, info, calib, get_adapter("2.5.0"))
    loader = d["default_loader"]
    assert loader["dataset_path"] == str(calib.resolve())
    loader["dataset_path"] = PLACEHOLDER
    return d


def test_v25_snapshot(config_dir: Path, tmp_path: Path):
    d = _build(config_dir, tmp_path)
    if os.environ.get("UPDATE_SNAPSHOTS") or not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert d == json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_v25_expected_keys(config_dir: Path, tmp_path: Path):
    d = _build(config_dir, tmp_path)
    assert set(d) >= {"inputs", "calibration_method", "calibration_num", "default_loader"}
    assert d["inputs"] == {"images": [1, 3, 640, 640]}
    assert d["calibration_num"] == 50  # fire 프리셋
    assert set(d["default_loader"]) >= {"dataset_path", "file_extensions", "preprocessings"}
    assert "jpg" in d["default_loader"]["file_extensions"]


def test_cli_values_reach_config(config_dir: Path, tmp_path: Path):
    d = _build(config_dir, tmp_path, set_items=["compile.calibration_method=MinMax",
                                                "compile.calibration_num=10"])
    assert d["calibration_method"] == "minmax"
    assert d["calibration_num"] == 10


def test_preprocess_steps_order():
    # 기본 host: 캘리브레이션 이미지가 이미 letterbox 되어 있으므로 resize/pad 없음
    steps = preprocess_steps(PreprocessCfg(), 640, 640)
    assert [s["op"] for s in steps] == ["bgr2rgb", "div", "transpose", "expand_dim"]
    assert steps[1]["value"] == 255.0


def test_preprocess_steps_dxcom_letterbox():
    steps = preprocess_steps(PreprocessCfg(letterbox_mode="dxcom"), 640, 640)
    assert [s["op"] for s in steps] == ["letterbox", "bgr2rgb", "div", "transpose", "expand_dim"]
    assert steps[0]["pad_value"] == 114


def test_host_config_has_no_resize(config_dir: Path, tmp_path: Path):
    d = _build(config_dir, tmp_path)
    keys = [next(iter(x)) for x in d["default_loader"]["preprocessings"]]
    assert keys == ["convertColor", "div", "transpose", "expandDim"]


@pytest.mark.parametrize("raw, expected", [
    ({"letterbox": True}, "host"),
    ({"letterbox": False}, "none"),
    ({"letterbox": False, "letterbox_mode": "host"}, "none"),
    ({"letterbox": True, "letterbox_mode": "dxcom"}, "dxcom"),
    ({"letterbox_mode": "DXCOM"}, "dxcom"),
])
def test_letterbox_mode_legacy(raw: dict, expected: str):
    assert PreprocessCfg.model_validate(raw).letterbox_mode == expected


def test_preprocess_variants():
    steps = preprocess_steps(
        PreprocessCfg(letterbox_mode="none", color="BGR", scale=1.0, layout="NHWC"), 320, 320
    )
    assert [s["op"] for s in steps] == ["resize", "expand_dim"]


def test_command_v25(config_dir: Path, tmp_path: Path):
    cfg = resolve_config("fire", config_dir=config_dir)
    cmd = get_adapter("2.5.0").build_command(
        tmp_path / "m.onnx", tmp_path / "c.json", tmp_path / "out", cfg
    )
    assert cmd[:7] == ["dxcom", "-m", str(tmp_path / "m.onnx"), "-c", str(tmp_path / "c.json"),
                       "-o", str(tmp_path / "out")]
    assert "--opt_level" in cmd and cmd[cmd.index("--opt_level") + 1] == "1"
    cfg0 = resolve_config("fire", config_dir=config_dir,
                          set_items=["compile.aggressive_partitioning=true"])
    cmd0 = get_adapter("2.5.0").build_command(tmp_path / "m.onnx", tmp_path / "c.json",
                                              tmp_path / "out", cfg0)
    assert len(cmd0) == len(cmd) + 1  # 추천값은 aggressive off → 켜면 플래그 1개 추가


@pytest.mark.parametrize(
    ("version", "cls"),
    [("2.5.0", DxcomV25Adapter), ("2.5.3", DxcomV25Adapter), ("2.4.1-rc.1", DxcomV24Adapter),
     ("2.2.0-rc.2", DxcomV22Adapter), ("2.9.0", DxcomV25Adapter)],
)
def test_get_adapter(version: str, cls: type):
    assert type(get_adapter(version)) is cls


@pytest.mark.parametrize("version", ["1.0.0", "garbage"])
def test_get_adapter_unsupported(version: str):
    with pytest.raises(DxconvError):
        get_adapter(version)


def test_onnx_info_from_cfg(config_dir: Path):
    cfg = resolve_config("fire", config_dir=config_dir, set_items=["export.imgsz=320"])
    info = onnx_info_from_cfg(cfg)
    assert info.primary_input.shape == [1, 3, 320, 320]


def test_dynamic_onnx_rejected(tmp_path: Path):
    p = make_onnx(tmp_path / "d.onnx", shape=["batch", 3, 640, 640])
    with pytest.raises(DxconvError, match="동적"):
        check_onnx(p, (11, 21))


def test_onnx_opset_range(tmp_path: Path):
    p = make_onnx(tmp_path / "o.onnx", opset=17)
    with pytest.raises(DxconvError, match="opset"):
        check_onnx(p, (11, 16))


def test_parse_log():
    text = "INFO npu subgraphs: 3\nWARNING: op X fallback\ncpu partitions = 2\n"
    s = get_adapter("2.5.0").parse_log(text)
    assert s["npu_partitions"] == 3
    assert s["cpu_partitions"] == 2
    assert len(s["warnings"]) == 1


def test_v25_parse_log_final_result():
    from dxconv.compile.adapters.v2_5 import DxcomV25Adapter

    log = (
        "\x1b[A[INFO] - Final result: 3 NPU groups, 2 CPU groups\n"
        "[WARNING] - No supported GPU/CUDA environment was detected.\n"
    )
    s = DxcomV25Adapter("2.5.0").parse_log(log)
    assert s["npu_partitions"] == 3 and s["cpu_partitions"] == 2
    assert s["warnings"] == ["[WARNING] - No supported GPU/CUDA environment was detected."]


@pytest.mark.parametrize("line", [
    "[WARNING] - After surgeon, cosine similarity for output index 0 is 0.8078 (below 0.90). "
    "This output flows through TopK/NMS, which amplifies minor floating-point differences "
    "from graph rewrites. Skipping error.",
    "WARNING: failed to fuse node X, skipping",
    "2026-11-24 04:23:23,025 - WARNING - Skipping error.",
])
def test_parse_log_warning_lines_are_not_errors(line: str):
    s = get_adapter("2.5.0").parse_log(line + "\n")
    assert s["warnings"] == [line]
    assert s["errors"] == []


def test_parse_log_real_errors_still_detected():
    text = "[ERROR] - Failed to compile model\nTraceback (most recent call last):\n"
    s = get_adapter("2.5.0").parse_log(text)
    assert len(s["errors"]) == 2
