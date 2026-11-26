"""CLI / 파이프라인 동작 테스트 (DX-COM 은 가짜 실행 파일로 대체)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from conftest import make_images, make_onnx
from typer.testing import CliRunner

from dxconv.cli import app

runner = CliRunner()

FAKE_DXCOM = """#!/usr/bin/env python3
import json, os, sys, pathlib
args = sys.argv[1:]
if args and args[0] in ("-v", "--version"):
    print("DX-COM v2.5.0"); sys.exit(0)
out = pathlib.Path(args[args.index("-o") + 1])
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads(pathlib.Path(args[args.index("-c") + 1]).read_text())
if os.environ.get("FAKE_DXCOM_LOG"):  # 호출 기록 (sweep 캐시 테스트용)
    with open(os.environ["FAKE_DXCOM_LOG"], "a") as f:
        f.write(json.dumps({"method": cfg["calibration_method"], "num": cfg["calibration_num"],
                            "args": args}) + "\\n")
opt = args[args.index("--opt_level") + 1] if "--opt_level" in args else "1"
print("compiling...")
print("[INFO] -   Optimization Level : " + os.environ.get("FAKE_OPT", opt))
print("npu subgraphs: 1")
print("WARNING: fake warning")
# 실제 dxnn(v9) 형식: 헤더 JSON + payload (compile_config, graph_info 섹션)
cc = {"input_shapes": cfg["inputs"], "dataset_path": cfg["default_loader"]["dataset_path"],
      "preprocessings": cfg["default_loader"]["preprocessings"], "compile_version": "2.5.0-rc.1",
      "calibration": {"num": int(os.environ.get("FAKE_NUM", cfg["calibration_num"])),
                      "method": os.environ.get("FAKE_METHOD", cfg["calibration_method"])}}
gi = {"graphs": [{"name": "npu_0", "device": "NPU", "inputs": [{"name": "images"}]},
                 {"name": "cpu_0", "device": "CPU", "inputs": [{"name": "t0"}]}]}
a, g = json.dumps(cc).encode(), json.dumps(gi).encode()
hdr = {"schema_version": 1, "version": 9, "payload_offset": 0,
       "data": {"compile_config": {"type": "str", "offset": 0, "size": len(a)},
                "graph_info": {"type": "str", "offset": len(a), "size": len(g)}}}
while True:  # payload_offset 를 헤더 길이에 맞춘다 (자릿수가 바뀌면 다시)
    head = b"DXNN" + b"\\x09\\x00\\x00\\x00" + json.dumps(hdr).encode()
    if hdr["payload_offset"] == len(head):
        break
    hdr["payload_offset"] = len(head)
(out / "m.dxnn").write_bytes(head + a + g)
if "--export_html" in args:
    (out / "m_summary.html").write_text("<html></html>")
"""

# docker run ... <image> dxcom <args> 를 흉내 낸다: 인자를 기록하고 가짜 dxcom 처럼 동작
FAKE_DOCKER = """#!/usr/bin/env python3
import sys, pathlib, json, os
args = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_DOCKER_LOG"])
with log.open("a") as f:
    f.write(json.dumps(args) + "\\n")
if args[:1] != ["run"]:
    sys.exit(0)
i = args.index("dxcom")
dx = args[i + 1:]
out = pathlib.Path(dx[dx.index("-o") + 1])
out.mkdir(parents=True, exist_ok=True)
print("npu subgraphs: 2")
(out / "m.dxnn").write_bytes(b"DOCKER")
"""


@pytest.fixture
def fake_dxcom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "dxcom"
    exe.write_text(FAKE_DXCOM)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return exe


@pytest.fixture
def no_dxcom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    # pip 으로 설치된 DX-COM(Linux)은 PATH 와 무관하게 검출되므로 함께 숨긴다
    monkeypatch.setattr("dxconv.env.detect.detect_dxcom_pip", lambda: [])


def test_env_without_dxcom(no_dxcom: None):
    res = runner.invoke(app, ["env", "--json"])
    assert res.exit_code == 0, res.output
    assert '"dxcom": []' in res.output


def test_convert_dry_run(tmp_path: Path, no_dxcom: None):
    make_images(tmp_path / "calib", 60)
    pt = tmp_path / "fire.pt"
    pt.write_bytes(b"fake")
    res = runner.invoke(app, ["convert", "--pt", str(pt), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"),
                              "--out", str(tmp_path / "runs"), "--dry-run"])
    assert res.exit_code == 0, res.output
    run = next((tmp_path / "runs" / "convert").iterdir())
    for name in ("resolved.yaml", "dxcom_config.json", "report.json", "env.json"):
        assert (run / name).is_file(), name
    rep = json.loads((run / "report.json").read_text())
    assert rep["success"] is True and rep["dry_run"] is True
    assert rep["inputs"]["pt"]["sha256"]
    assert rep["inputs"]["calib"]["sampled"] == 50
    assert len(list((run / "calib_subset").iterdir())) == 50


def test_compile_with_fake_dxcom(tmp_path: Path, fake_dxcom: Path):
    make_images(tmp_path / "calib", 60)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"), "--calib-num", "20",
                              "--out", str(tmp_path / "runs")])
    assert res.exit_code == 0, res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    assert (run / "fire.dxnn").read_bytes().startswith(b"DXNN")
    assert (run / "fire.onnx").is_file()
    assert "npu subgraphs" in (run / "compile.log").read_text()
    rep = json.loads((run / "report.json").read_text())
    assert rep["success"] is True
    assert rep["versions"]["dxcom"]["selected"] == "2.5.0"
    pre = rep["preprocess"]
    assert pre["letterbox_mode"] == "host" and pre["target_hw"] == [640, 640]
    assert pre["steps"] == ["bgr2rgb", "div", "transpose", "expand_dim"]
    assert pre["host_letterbox"]["impl"] == "ultralytics.data.augment.LetterBox"
    assert all(f.suffix == ".png" for f in (run / "calib_subset").iterdir())
    assert rep["compile_summary"]["npu_partitions"] == 1
    assert "compile" in rep["timings_sec"]
    assert "ultralytics_version" in rep["inputs"]["onnx"]  # 테스트 ONNX 는 메타데이터 없음 → None
    ds = rep["dxnn_settings"]
    assert ds["mismatches"] == []
    assert ds["requested"]["calibration_num"] == 20 and ds["actual"]["calibration_num"] == 20
    assert ds["actual"]["calibration_method"] == "minmax" and ds["actual"]["opt_level"] == 1
    assert ds["actual"]["aggressive_flag_in_command"] is False  # 추천값 aggressive off
    assert ds["actual"]["preprocessings"] == ["convertColor", "div", "transpose", "expandDim"]
    assert not any("dxnn 설정" in w for w in rep["warnings"])


def test_compile_export_html(tmp_path: Path, fake_dxcom: Path):
    make_images(tmp_path / "calib", 30)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"), "--calib-num", "20",
                              "--export-html", "--out", str(tmp_path / "runs")])
    assert res.exit_code == 0, res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    assert (run / "m_summary.html").is_file()
    rep = json.loads((run / "report.json").read_text())
    assert "--export_html" in rep["dxcom_command"]
    assert rep["outputs"]["html_reports"] == [str(run / "m_summary.html")]


@pytest.mark.parametrize("mode", ["host", "none"])
def test_compile_via_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "docker"
    exe.write_text(FAKE_DOCKER)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    pybin = Path(os.path.realpath(sys.executable)).parent
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{pybin}")
    dlog = tmp_path / "docker.log"
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(dlog))

    make_images(tmp_path / "calib", 30)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"), "--calib-num", "20",
                              "--set", f"compile.preprocess.letterbox_mode={mode}",
                              "--out", str(tmp_path / "runs")])
    assert res.exit_code == 0, res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    assert (run / "fire.dxnn").read_bytes() == b"DOCKER"
    rep = json.loads((run / "report.json").read_text())
    assert rep["dxcom_runner"] == "docker:dxcom:2.5.0"
    calls = [json.loads(ln) for ln in dlog.read_text().splitlines()]
    run_call = next(c for c in calls if c[:1] == ["run"])
    assert "dxcom:2.5.0" in run_call
    mounts = {run_call[i + 1] for i, a in enumerate(run_call) if a == "-v"}
    calib_mount = f"{(tmp_path / 'calib').resolve()}:{(tmp_path / 'calib').resolve()}"
    if mode == "host":
        # letterbox 된 PNG 가 run 폴더에 있으므로 원본 폴더는 마운트하지 않는다
        assert calib_mount not in mounts
    else:
        # 심볼릭 링크 대상(캘리브레이션 원본) 폴더가 같은 경로로 마운트돼야 함
        assert calib_mount in mounts
    assert any(m.startswith(str(run.resolve()) + ":") or str(run.resolve()).startswith(m.split(":")[0])
               for m in mounts)


def test_compile_report_class_counts(tmp_path: Path, fake_dxcom: Path):
    make_images(tmp_path / "calib" / "fire", 30)
    make_images(tmp_path / "calib" / "normal", 30)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"),
                              "--out", str(tmp_path / "runs"), "--dry-run"])
    assert res.exit_code == 0, res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    calib = json.loads((run / "report.json").read_text())["inputs"]["calib"]
    assert calib["sampling"] == "stratified"  # 기본값
    assert calib["class_totals"] == {"fire": 30, "normal": 30}
    assert calib["class_counts"] == {"fire": 25, "normal": 25}  # fire 프리셋 calibration_num 50


def test_compile_report_sampling_fallback_warns(tmp_path: Path, fake_dxcom: Path):
    make_images(tmp_path / "calib", 60)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"),
                              "--out", str(tmp_path / "runs"), "--dry-run"])
    assert res.exit_code == 0, res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    rep = json.loads((run / "report.json").read_text())
    assert rep["inputs"]["calib"]["sampling"] == "random"
    assert rep["inputs"]["calib"]["sampling_requested"] == "stratified"
    assert any("하위 폴더가 없어" in w for w in rep["warnings"])


@pytest.mark.parametrize("env, expect", [
    ({"FAKE_METHOD": "ema", "FAKE_NUM": "100"}, "calibration_num"),   # 2.4.1 처럼 num 이 바뀐 경우
    ({"FAKE_METHOD": "ema"}, "calibration_method"),                    # minmax 요청 → ema 기록
    ({"FAKE_OPT": "0"}, "opt_level"),                                  # 로그의 Optimization Level
])
def test_compile_warns_on_dxnn_setting_mismatch(tmp_path: Path, fake_dxcom: Path,
                                                monkeypatch: pytest.MonkeyPatch,
                                                env: dict, expect: str):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    make_images(tmp_path / "calib", 30)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"), "--calib-num", "20",
                              "--calib-method", "minmax", "--out", str(tmp_path / "runs")])
    assert res.exit_code == 0, res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    rep = json.loads((run / "report.json").read_text())
    ds = rep["dxnn_settings"]
    assert any(m.startswith(expect) for m in ds["mismatches"]), ds
    assert any("[dxnn 설정 불일치]" in w and expect in w for w in rep["warnings"])
    assert ds["requested"]["calibration_method"] == "minmax"


def test_compile_failure_writes_report(tmp_path: Path, no_dxcom: None):
    make_images(tmp_path / "calib", 60)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"),
                              "--out", str(tmp_path / "runs")])
    assert res.exit_code == 1
    assert "Traceback" not in res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    rep = json.loads((run / "report.json").read_text())
    assert rep["success"] is False
    assert "DX-COM" in rep["error"]


def test_bad_model_name_exit_1(tmp_path: Path):
    res = runner.invoke(app, ["export", "--pt", "x.pt", "--model", "bad name/x"])
    assert res.exit_code == 1
    assert "모델 이름" in res.output


@pytest.mark.integration
def test_real_convert(tmp_path: Path):
    """실제 ultralytics + DX-COM 으로 변환 (DXCONV_TEST_PT, DXCONV_TEST_CALIB 필요)."""
    pt = os.environ.get("DXCONV_TEST_PT")
    calib = os.environ.get("DXCONV_TEST_CALIB")
    if not pt or not calib:
        pytest.skip("DXCONV_TEST_PT / DXCONV_TEST_CALIB 환경변수가 필요합니다.")
    res = runner.invoke(app, ["convert", "--pt", pt, "--model", "fire", "--calib-dir", calib,
                              "--out", str(tmp_path / "runs")])
    assert res.exit_code == 0, res.output
