"""dxnn 실제 설정 읽기 / 요청 설정 비교 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dxconv.compile.dxnn_info import (
    DxnnFormatError,
    compare_settings,
    opt_level_from_log,
    read_dxnn,
)

PRE = [{"convertColor": {"form": "BGR2RGB"}}, {"div": {"x": 255}},
       {"transpose": {"axis": [2, 0, 1]}}, {"expandDim": {"axis": 0}}]
GRAPHS = {"graphs": [
    {"name": "npu_0", "device": "NPU", "inputs": [{"name": "images"}]},
    {"name": "cpu_0", "device": "CPU",
     "inputs": [{"name": "/model.23/Split_output_1"}, {"name": "/model.23/Split_output_0"}]},
]}


def make_dxnn(path: Path, compile_config: dict, version: int = 9) -> Path:
    """실측 구조를 따른 최소 dxnn. v9: payload_offset 기준, v8: 헤더 영역 크기(size) 기준."""
    cc, gi = json.dumps(compile_config).encode(), json.dumps(GRAPHS).encode()
    data = {"compile_config": {"type": "str", "offset": 0, "size": len(cc)},
            "graph_info": {"type": "str", "offset": len(cc), "size": len(gi)},
            "merged_model": {"type": "bytes", "offset": 0, "size": 0}}
    if version == 9:
        hdr = {"schema_version": 1, "version": 9, "payload_offset": 0, "data": data}
        while True:
            head = b"DXNN\x09\x00\x00\x00" + json.dumps(hdr).encode()
            if hdr["payload_offset"] == len(head):
                break
            hdr["payload_offset"] = len(head)
    else:
        hdr = {"version": 8, "signature": "DXNN", "size": 1024, "data": data}
        head = (b"DXNN\x08\x00\x00\x00" + json.dumps(hdr).encode()).ljust(1024, b"\x00")
    path.write_bytes(head + cc + gi + b"\x00" * 16)
    return path


V9_CC = {"input_shapes": {"images": [1, 3, 640, 640]}, "dataset_path": "/x/calib_subset",
         "preprocessings": PRE, "compile_version": "2.5.0-rc.1",
         "calibration": {"num": 50, "method": "minmax", "seed": None}}
# 기존 2.4.1 C05 와 같은 모양: 요청 minmax/50 이었지만 기록은 ema/100
V8_CC = {"input_shapes": {"images": [1, 3, 640, 640]}, "dataset_path": "/tmp/cal_50",
         "calibration_num": 100, "calibration_method": "ema", "preprocessings": PRE,
         "compile_version": "2.4.1-rc.1", "seed": 42}


def test_read_v9(tmp_path: Path):
    info = read_dxnn(make_dxnn(tmp_path / "a.dxnn", V9_CC))
    assert (info.format_version, info.compile_version) == (9, "2.5.0-rc.1")
    assert (info.calibration_method, info.calibration_num) == ("minmax", 50)
    assert info.preprocessings == ["convertColor", "div", "transpose", "expandDim"]
    assert (info.npu_graphs, info.cpu_graphs) == (1, 1)
    assert info.cpu_inputs == ["/model.23/Split_output_0", "/model.23/Split_output_1"]
    assert info.to_dict()["cpu_graphs"] == 1


def test_read_v8(tmp_path: Path):
    info = read_dxnn(make_dxnn(tmp_path / "a.dxnn", V8_CC, version=8))
    assert (info.format_version, info.calibration_method, info.calibration_num) == (8, "ema", 100)


@pytest.mark.parametrize("content", [b"", b"NOTDXNN", b"DXNN\x09\x00\x00\x00{broken"])
def test_read_invalid(tmp_path: Path, content: bytes):
    p = tmp_path / "bad.dxnn"
    p.write_bytes(content)
    with pytest.raises(DxnnFormatError):
        read_dxnn(p)


def test_opt_level_from_log():
    log = "\x1b[A[INFO] -   Optimization Level : 0\nfoo\n"
    assert opt_level_from_log(log) == 0
    assert opt_level_from_log("no level") is None


REQ = {"calibration_method": "minmax", "calibration_num": 50, "opt_level": 1,
       "aggressive_partitioning": True,
       "preprocessings": ["convertColor", "div", "transpose", "expandDim"],
       "input_shapes": {"images": [1, 3, 640, 640]}}


def test_compare_match(tmp_path: Path):
    info = read_dxnn(make_dxnn(tmp_path / "a.dxnn", V9_CC))
    res = compare_settings(REQ, info, "Optimization Level : 1", calib_images=50)
    assert res["mismatches"] == []
    assert res["actual"]["partition"]["cpu_graphs"] == 1
    assert any("aggressive_partitioning" in n for n in res["notes"])  # 확인 불가 안내


def test_compare_legacy_241_mismatch(tmp_path: Path):
    """기존 2.4.1 C05: 요청 minmax/50 → 기록 ema/100, 폴더 이미지 50 장."""
    info = read_dxnn(make_dxnn(tmp_path / "a.dxnn", V8_CC, version=8))
    res = compare_settings(REQ, info, "Optimization Level : 1", calib_images=50)
    joined = "\n".join(res["mismatches"])
    assert "calibration_method: 요청 'minmax' ≠ 실제 'ema'" in joined
    assert "calibration_num: 요청 50 ≠ 실제 100" in joined
    assert "이미지는 50 장뿐" in joined


def test_compare_without_dxnn():
    res = compare_settings(REQ, None, "")
    assert res["mismatches"] == []
    assert any("확인할 수 없음" in n for n in res["notes"])


def test_header_with_brace_byte_before_json(tmp_path: Path):
    """실제 v9 처럼 시그니처 뒤 고정 필드(체크섬 등)에 '{'(0x7B) 바이트가 우연히 있어도 헤더를 읽는다."""
    cc, gi = json.dumps(V9_CC).encode(), json.dumps(GRAPHS).encode()
    fixed = b"DXNN" + b"\x09\x00\x00\x00" + b"\xb9\x02\x00\x00\x00\x00\x00\x00" + b"\x2c\x55\x7b\x9e" + b"\x00" * 12
    hdr = {"schema_version": 1, "version": 9, "payload_offset": 0,
           "data": {"compile_config": {"type": "str", "offset": 0, "size": len(cc)},
                    "graph_info": {"type": "str", "offset": len(cc), "size": len(gi)}}}
    while True:
        head = fixed + json.dumps(hdr).encode()
        if hdr["payload_offset"] == len(head):
            break
        hdr["payload_offset"] = len(head)
    p = tmp_path / "brace.dxnn"
    p.write_bytes(head + cc + gi)
    assert fixed.index(b"{") < len(fixed)  # 체크섬 안에 '{'
    info = read_dxnn(p)
    assert (info.calibration_method, info.calibration_num, info.format_version) == ("minmax", 50, 9)
