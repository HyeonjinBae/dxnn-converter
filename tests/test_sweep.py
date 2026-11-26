"""sweep 테스트 (가짜 dxcom / 고정 출력 ONNX / 가짜 PT 러너)."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import REPO
from test_eval import _const_onnx, _noise
from test_pipeline import FAKE_DXCOM
from typer.testing import CliRunner

from dxconv import DxconvError
from dxconv.cli import app
from dxconv.sweep.objective import parse_objective, select_stage
from dxconv.sweep.space import iter_combinations, load_space, to_nested

runner = CliRunner()

DETS = np.zeros((1, 300, 6), np.float32)
DETS[0, 0] = [240, 240, 400, 400, 0.9, 0]
DETS[0, 1] = [10, 10, 60, 60, 0.4, 1]

SPACE = {
    "stages": {
        "A": {"export.opset": [17, 18]},
        "B": {"compile.calibration_method": ["ema", "minmax"], "compile.calibration_num": [2, 3]},
        "C": {"compile.opt_level": [0, 1], "compile.aggressive_partitioning": [True, False]},
    },
    "objective": "PT 대비 mAP 하락 <= 1%p 중 FPS 최대",
}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    # configs 사본 (개인 *.local.yaml 제외) + 데이터셋 설정
    cfgdir = tmp_path / "configs"
    shutil.copytree(REPO / "configs", cfgdir, ignore=shutil.ignore_patterns("*.local.yaml"))
    monkeypatch.setenv("DXCONV_CONFIG_DIR", str(cfgdir))
    data = tmp_path / "data"
    names = []
    for i in range(8):
        _noise(data / "images" / f"im{i}.png", 640, 640 - i)
        (data / "labels").mkdir(parents=True, exist_ok=True)
        (data / "labels" / f"im{i}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        names.append(f"im{i}.png")
    (data / "cal3.txt").write_text("\n".join(names[:3]))
    (data / "cal2.txt").write_text("\n".join(names[:2]))
    (data / "tuning.txt").write_text("\n".join(names[3:6]))
    (data / "evaluation.txt").write_text("\n".join(names[6:]))
    (cfgdir / "datasets").mkdir(exist_ok=True)
    (cfgdir / "datasets" / "fire.local.yaml").write_text(yaml.safe_dump({
        "name": "fire", "images_dir": str(data / "images"), "labels_dir": str(data / "labels"),
        "calibration": {2: str(data / "cal2.txt"), 3: str(data / "cal3.txt")},
        "tuning": str(data / "tuning.txt"), "evaluation": str(data / "evaluation.txt")}))
    space = tmp_path / "space.yaml"
    space.write_text(yaml.safe_dump(SPACE, allow_unicode=True))

    # 가짜 dxcom (호출 기록)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "dxcom"
    exe.write_text(FAKE_DXCOM)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    calls = tmp_path / "dxcom_calls.jsonl"
    monkeypatch.setenv("FAKE_DXCOM_LOG", str(calls))

    # export: 고정 출력 ONNX 복사 (실제 ultralytics 없이)
    onnx_src = _const_onnx(tmp_path / "const.onnx", DETS)

    def fake_export_stage(cfg, pt, run_dir, report, config_dir, dry_run):  # noqa: ANN001, ANN202
        dest = run_dir / f"{cfg.model}.onnx"
        shutil.copy2(onnx_src, dest)
        return dest

    monkeypatch.setattr("dxconv.pipeline.export_stage", fake_export_stage)

    state = {"pt_dets": DETS[0]}

    class FakePt:
        hw = (640, 640)
        net = type("Net", (), {"names": {0: "fire", 1: "smoke"}})()

        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return state["pt_dets"]

    monkeypatch.setattr("dxconv.evaluate.baseline.PtRunner", FakePt)
    pt = tmp_path / "fire.pt"
    pt.write_bytes(b"fake pt")
    return {"tmp": tmp_path, "space": space, "pt": pt, "calls": calls, "out": tmp_path / "runs",
            "state": state, "root": tmp_path / "runs" / "sweep" / "fire"}


def _sweep(e: dict, *extra: str):  # noqa: ANN202
    return runner.invoke(app, ["sweep", "--pt", str(e["pt"]), "--model", "fire",
                               "--space", str(e["space"]), "--out", str(e["out"]),
                               "--check-images", "2", "--set", "compile.calibration_num=2", *extra])


def _ncalls(e: dict) -> int:
    return len(e["calls"].read_text().splitlines()) if e["calls"].exists() else 0


def _db(e: dict) -> sqlite3.Connection:
    con = sqlite3.connect(e["root"] / "sweep.db")
    con.row_factory = sqlite3.Row
    return con


# --------------------------------------------------------------------------- #
def test_full_run_selects_A_then_pending_B(env: dict):
    res = _sweep(env, "--compile-only")
    assert res.exit_code == 0, res.output
    # A 2개 + B 4개 중 (ema, num 2) 는 A 의 (opset 17, 기본 ema/2) 와 같은 설정 → 캐시 재사용
    assert _ncalls(env) == 2 + 3
    assert "캐시" in res.output
    con = _db(env)
    sel = {r["stage"]: dict(r) for r in con.execute("SELECT * FROM selections")}
    assert sel["A"]["status"] == "selected" and "기본 설정" in sel["A"]["reason"]
    assert sel["B"]["status"] == "pending"
    assert "C" not in sel  # B 에서 멈춤
    # A 는 FP32 확인 통과, B 는 A 선택값(opset 17) 고정
    a_rows = con.execute("SELECT c.fp32_ok, m.params FROM members m JOIN combos c USING(hash) "
                         "WHERE stage='A'").fetchall()
    assert all(r["fp32_ok"] == 1 for r in a_rows)
    b_fixed = {json.loads(r["fixed"])["export.opset"] for r in
               con.execute("SELECT fixed FROM members WHERE stage='B'")}
    assert b_fixed == {17}
    # 설정 검증 결과 기록
    for r in con.execute("SELECT settings, settings_ok, status FROM combos"):
        s = json.loads(r["settings"])
        assert r["settings_ok"] == 1 and r["status"] == "compiled" and s["mismatches"] == []
        assert s["requested"]["calibration_method"] == s["actual"]["calibration_method"]
    pending = json.loads((env["root"] / "pending_eval.json").read_text())
    assert pending["stage"] == "B" and len(pending["results"]) == 4
    assert all(it["dxnn_sha256"] and it["map50_95"] is None for it in pending["results"])
    summary = (env["root"] / "summary.md").read_text()
    assert "Tuning mAP50-95" in summary and "pending" in summary and "✅ 선택" in summary


def test_cache_and_resume(env: dict):
    assert _sweep(env, "--compile-only").exit_code == 0
    n = _ncalls(env)
    # 같은 설정으로 다시 실행 → 컴파일 없음
    res = _sweep(env, "--compile-only")
    assert res.exit_code == 0 and _ncalls(env) == n and "캐시" in res.output
    # 중단된 조합(running)은 다시 컴파일
    con = _db(env)
    h = con.execute("SELECT hash FROM members WHERE stage='B' ORDER BY ord LIMIT 1").fetchone()[0]
    con.execute("UPDATE combos SET status='running' WHERE hash=?", (h,))
    con.commit()
    assert _sweep(env, "--compile-only").exit_code == 0
    assert _ncalls(env) == n + 1


def test_setting_mismatch_is_excluded(env: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_METHOD", "ema")  # minmax 요청도 ema 로 기록되는 상황 (2.4.1 처럼)
    res = _sweep(env, "--stage", "B", "--compile-only")
    assert res.exit_code == 0, res.output
    assert "제외 (설정 검증 실패)" in res.output
    con = _db(env)
    rows = con.execute("SELECT m.params, c.status, c.error FROM members m JOIN combos c USING(hash) "
                       "WHERE stage='B'").fetchall()
    for r in rows:
        method = json.loads(r["params"])["compile.calibration_method"]
        assert r["status"] == ("compiled" if method == "ema" else "excluded")
        if method == "minmax":
            assert "calibration_method" in r["error"]
    pending = json.loads((env["root"] / "pending_eval.json").read_text())
    assert len(pending["results"]) == 2  # 제외 조합은 평가 대상 아님
    assert "제외/실패" in (env["root"] / "summary.md").read_text()


def test_stage_B_without_A_uses_base(env: dict):
    res = _sweep(env, "--stage", "B", "--compile-only")
    assert res.exit_code == 0, res.output
    assert "기본 설정 값" in res.output
    fixed = json.loads(_db(env).execute("SELECT fixed FROM members WHERE stage='B'").fetchone()[0])
    assert fixed["export.opset"] == 17 and fixed["_source"]["A"] == "base"


def test_fp32_mismatch_stops_at_A(env: dict):
    env["state"]["pt_dets"] = DETS[0] + np.array([5, 0, 0, 0, 0, 0], np.float32)  # 좌표 5px 차이
    res = _sweep(env, "--compile-only")
    assert res.exit_code == 0, res.output
    sel = dict(_db(env).execute("SELECT * FROM selections WHERE stage='A'").fetchone())
    assert sel["status"] == "none"
    assert _ncalls(env) == 2  # B 로 진행하지 않음


def test_import_eval_selects_B_then_C(env: dict):
    assert _sweep(env, "--compile-only").exit_code == 0
    pend = json.loads((env["root"] / "pending_eval.json").read_text())
    for i, it in enumerate(pend["results"]):
        it["map50_95"] = 0.50 + i * 0.01  # 마지막 조합이 최대
        it["map50"] = 0.8
    best_b = pend["results"][-1]["hash"]
    pend["pt"] = {"map50": 0.85, "map50_95": 0.535}
    f = env["tmp"] / "eval_b.json"
    f.write_text(json.dumps(pend))
    res = runner.invoke(app, ["sweep", "import-eval", str(f), "--model", "fire",
                              "--space", str(env["space"]), "--out", str(env["out"])])
    assert res.exit_code == 0, res.output
    con = _db(env)
    sel_b = dict(con.execute("SELECT * FROM selections WHERE stage='B'").fetchone())
    assert sel_b["status"] == "selected" and sel_b["hash"] == best_b
    delta = con.execute("SELECT delta_pp FROM combos WHERE hash=?", (best_b,)).fetchone()[0]
    assert delta == pytest.approx((0.53 - 0.535) * 100)

    # C 단계 → 컴파일 후 pending (FPS 없음)
    assert _sweep(env, "--compile-only").exit_code == 0
    sel_c = dict(_db(env).execute("SELECT * FROM selections WHERE stage='C'").fetchone())
    assert sel_c["status"] == "pending"
    pend_c = json.loads((env["root"] / "pending_eval.json").read_text())
    assert pend_c["stage"] == "C"
    # PT 0.535, 허용 하락 1%p → mAP50-95 0.525 이상만 후보.
    # 0.515(FPS 300, -2%p), 0.52(FPS 260, -1.5%p) 는 탈락 → 남은 것 중 FPS 250 선택
    vals = [(0.515, 300.0), (0.53, 250.0), (0.53, 240.0), (0.52, 260.0)]
    for it, (m, fps) in zip(pend_c["results"], vals, strict=False):
        it["map50_95"], it["fps"] = m, fps
        if it["hash"] == best_b:  # B 선택 조합과 같은 dxnn (캐시 공유) → 같은 mAP
            it["map50_95"], it["fps"] = 0.54, 200.0
    pend_c.pop("pt")
    f2 = env["tmp"] / "eval_c.json"
    f2.write_text(json.dumps(pend_c))
    res = runner.invoke(app, ["sweep", "import-eval", str(f2), "--model", "fire",
                              "--space", str(env["space"]), "--out", str(env["out"])])
    assert res.exit_code == 0, res.output
    sel_c = dict(_db(env).execute("SELECT * FROM selections WHERE stage='C'").fetchone())
    assert sel_c["status"] == "selected"
    assert sel_c["hash"] == pend_c["results"][1]["hash"]
    # B 선택이 바뀌면 C 는 stale
    pend["results"][0]["map50_95"] = 0.9
    f.write_text(json.dumps(pend))
    res = runner.invoke(app, ["sweep", "import-eval", str(f), "--model", "fire",
                              "--space", str(env["space"]), "--out", str(env["out"])])
    assert "덮어씀" in res.output  # 기존 평가값을 바꾸면 경고
    sel_c = dict(_db(env).execute("SELECT * FROM selections WHERE stage='C'").fetchone())
    assert sel_c["status"] == "stale"


def test_import_eval_baseline_report_and_errors(env: dict):
    assert _sweep(env, "--compile-only").exit_code == 0
    rep = env["tmp"] / "baseline_report.json"
    rep.write_text(json.dumps({"baseline": {"tuning": {"pt": {"coco": {"map50": 0.9, "map50_95": 0.6}}}}}))
    bad = env["tmp"] / "bad.json"
    bad.write_text(json.dumps({"results": [{"hash": "zz", "map50_95": 0.5},
                                           {"hash": "0000000000000000", "map50_95": 0.5}]}))
    res = runner.invoke(app, ["sweep", "import-eval", str(bad), "--model", "fire",
                              "--baseline-report", str(rep), "--space", str(env["space"]),
                              "--out", str(env["out"])])
    assert res.exit_code == 0, res.output
    assert "8자 이상" in res.output and "일치하는 조합 0개" in res.output
    con = _db(env)
    pt = json.loads(con.execute("SELECT value FROM meta WHERE key='pt_baseline'").fetchone()[0])
    assert pt["map50_95"] == 0.6
    wrong = env["tmp"] / "wrong.json"
    wrong.write_text(json.dumps({"split": "evaluation", "results": []}))
    res = runner.invoke(app, ["sweep", "import-eval", str(wrong), "--model", "fire",
                              "--out", str(env["out"])])
    assert res.exit_code == 1 and "tuning" in res.output


def test_status_and_remote(env: dict):
    res = runner.invoke(app, ["sweep", "status", "--model", "fire", "--out", str(env["out"])])
    assert res.exit_code == 1 and "sweep DB" in res.output
    assert _sweep(env, "--stage", "A").exit_code == 0
    res = runner.invoke(app, ["sweep", "status", "--model", "fire", "--out", str(env["out"])])
    assert res.exit_code == 0 and "A 단계" in res.output
    res = _sweep(env, "--remote", "npu")
    assert res.exit_code == 1 and "remote.yaml" in res.output  # 원격 설정 없음


FAKE_DX_ENGINE = """
import json, os
import numpy as np
__version__ = "3.5.0"
DETS = np.array(json.loads(os.environ["FAKE_NPU_DETS"]), dtype=np.float32)[None]

class InferenceEngine:
    def __init__(self, path):
        self.path = path
    def get_input_tensors_info(self):
        return [{"name": "images", "shape": [1, 640, 640, 3], "dtype": np.uint8}]
    def get_output_tensors_info(self):
        return [{"name": "output0", "shape": [1, 300, 6], "dtype": np.float32}]
    def get_model_version(self):
        return "v9"
    def run(self, xs):
        assert xs[0].shape == (1, 640, 640, 3) and xs[0].dtype == np.uint8
        return [DETS.copy()]
"""

SHIMS = {
    # ssh ... user@host "bash -c <script>" → 로컬 실행
    "ssh": '#!/bin/bash\nlast="${@: -1}"\nexec bash -c "$last"\n',
    # scp -q -P port ... src user@host:path → cp
    "scp": '#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"; cp "$src" "${dst#*:}"\n',
    # flock -n file cmd | flock -w N file cmd → cmd
    "flock": '#!/bin/bash\nwhile [[ "$1" == -* ]]; do [[ "$1" == "-w" ]] && shift; shift; done\nshift\nexec "$@"\n',
    "dxrt-cli": ('#!/bin/bash\necho "DXRT v3.5.0"\necho " * Device 0: M1, Accelerator type"\n'
                 'echo " * RT Driver version   : v2.7.0"\necho " * PCIe Driver version : v2.7.0"\n'
                 'echo " * FW version          : v2.7.6"\necho "NPU 0: x"\n'),
}


def _setup_remote(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """가짜 ssh/scp/flock/dxrt-cli + 가짜 dx_engine 장비."""
    import sys

    tmp = env["tmp"]
    bindir = tmp / "bin"
    for name, body in SHIMS.items():
        f = bindir / name
        f.write_text(body)
        f.chmod(0o755)
    eng = tmp / "fake_engine" / "dx_engine"
    eng.mkdir(parents=True)
    (eng / "__init__.py").write_text(FAKE_DX_ENGINE)
    py = bindir / "fakepy"
    py.write_text(f'#!/bin/bash\nPYTHONPATH={tmp / "fake_engine"} exec {sys.executable} "$@"\n')
    py.chmod(0o755)
    monkeypatch.setenv("FAKE_NPU_DETS", json.dumps(DETS[0].tolist()))  # NPU 출력 = PT 출력
    (tmp / "configs" / "remote.yaml").write_text(yaml.safe_dump({"hosts": {"npu": {
        "hostname": "localhost", "username": "me", "workdir": str(tmp / "device"),
        "python": str(py), "lock_file": str(tmp / "lock")}}}))


def test_remote_sweep_end_to_end(env: dict, monkeypatch: pytest.MonkeyPatch):
    """가짜 ssh/scp/flock/dxrt-cli + 가짜 dx_engine 으로 원격 평가 전체 흐름."""
    tmp = env["tmp"]
    _setup_remote(env, monkeypatch)
    res = runner.invoke(app, ["remote", "check", "--host", "npu", "--out", str(env["out"])])
    assert res.exit_code == 0, res.output
    assert "v9" in res.output and "3.5.0" in res.output

    res = _sweep(env, "--remote", "npu", "--set", "evaluate.fps_seconds=0.2")
    assert res.exit_code == 0, res.output
    con = _db(env)
    sel = {r["stage"]: dict(r) for r in con.execute("SELECT * FROM selections")}
    assert {k: v["status"] for k, v in sel.items()} == {"A": "selected", "B": "selected", "C": "selected"}
    rows = con.execute("SELECT * FROM combos WHERE eval_source='remote:npu'").fetchall()
    assert rows and all(r["tuning_map50_95"] is not None and r["fps"] for r in rows)
    assert all(abs(r["delta_pp"]) < 1e-6 for r in rows)  # NPU 출력 = PT 출력
    meta = {r["key"]: json.loads(r["value"]) for r in con.execute("SELECT * FROM meta")}
    assert meta["device"]["dxrt"] == "3.5.0" and meta["pt_baseline"]["map50_95"] > 0
    fin = meta["final_eval"]
    assert fin["images"] == 2 and fin["delta_map50_95_pp"] == pytest.approx(0.0)
    summary = (env["root"] / "summary.md").read_text()
    assert "최종 평가" in summary and "NPU 장비" in summary
    # 장비 쪽: 모델·이미지·결과
    dev = tmp / "device"
    assert list((dev / "models").glob("*.dxnn")) and list((dev / "results").glob("*_tuning.json"))
    # 다시 실행하면 평가를 반복하지 않는다
    n = len(list((dev / "results").glob("*.json")))
    assert _sweep(env, "--remote", "npu", "--set", "evaluate.fps_seconds=0.2").exit_code == 0
    assert len(list((dev / "results").glob("*.json"))) == n


# --------------------------------------------------------------------------- #
# 단위
# --------------------------------------------------------------------------- #
def test_space_and_combinations():
    sp = load_space(REPO / "configs/sweep/default_space.yaml")
    assert list(sp["stages"]) == ["B"]  # 기본: 캘리브레이션만 (A/C 는 추천값 고정)
    assert sp["stages"]["B"]["compile.calibration_method"] == ["ema", "minmax"]  # iqr 제외
    combos = iter_combinations(sp["stages"]["B"])
    assert len(combos) == 6 and combos[0] == {"compile.calibration_method": "ema",
                                              "compile.calibration_num": 100, "compile.calib_repeat": 1}
    full = load_space(REPO / "configs/sweep/full_space.yaml")
    assert list(full["stages"]) == ["A", "B", "C"]
    assert to_nested({"a.b": 1, "a.c": 2, "d.e": 3}) == {"a": {"b": 1, "c": 2}, "d": {"e": 3}}


def test_objective_parse_and_select():
    o = parse_objective("PT 대비 mAP 하락 <= 1.5%p 중 FPS 최대")
    assert o.max_drop_pp == 1.5 and o.maximize == "fps"
    rows = [{"hash": "a", "status": "compiled", "params": {}, "tuning_map50_95": 0.5, "fps": 100},
            {"hash": "b", "status": "excluded", "params": {}, "tuning_map50_95": 0.9, "fps": 900}]
    assert select_stage("B", rows, o, {}, None).hash == "a"  # 제외 조합은 선택 안 함
    assert select_stage("C", rows, o, {}, None).status == "pending"  # PT 기준선 없음
    rows[0]["fps"] = None
    sel = select_stage("C", rows, o, {}, 0.5)
    assert sel.status == "pending" and sel.missing == ["a"]


def test_objective_forms():
    o = parse_objective("FPS >= 20*15 중 mAP 최대")
    assert o.min_fps == 300 and o.max_drop_pp is None and o.maximize == "tuning_map50_95"
    o = parse_objective("PT 대비 mAP50 하락 <= 2%p, FPS ≥ 300 중 mAP50 최대")
    assert (o.max_drop_pp, o.drop_metric, o.min_fps, o.maximize) == (2.0, "tuning_map50", 300, "tuning_map50")
    for bad in ("FPS 최대", "아무거나", "mAP 하락 <= 1%p"):
        with pytest.raises(DxconvError):
            parse_objective(bad)
    rows = [{"hash": "a", "status": "compiled", "params": {}, "tuning_map50": 0.80,
             "tuning_map50_95": 0.50, "fps": 250},
            {"hash": "b", "status": "compiled", "params": {}, "tuning_map50": 0.79,
             "tuning_map50_95": 0.49, "fps": 320},
            {"hash": "c", "status": "compiled", "params": {}, "tuning_map50": 0.70,
             "tuning_map50_95": 0.40, "fps": 400}]
    pt = {"map50": 0.80, "map50_95": 0.50}
    # 하락 <= 1%p 중 FPS 최대 → b (c 는 10%p 하락)
    assert select_stage("C", rows, parse_objective(None), {}, pt).hash == "b"
    # FPS >= 300 중 mAP 최대 → b (a 는 FPS 미달)
    assert select_stage("C", rows, parse_objective("FPS >= 300 중 mAP 최대"), {}, pt).hash == "b"
    # FPS 조건만 있으면 PT 기준선이 없어도 선택 가능
    assert select_stage("C", rows, parse_objective("FPS >= 20×15 중 mAP 최대"), {}, None).hash == "b"
    sel = select_stage("C", rows, parse_objective("FPS >= 1000 중 mAP 최대", strict=True), {}, pt)
    assert sel.status == "none"
    # 기본: 조건 미충족이면 하락이 가장 작은(mAP 최대) 조합으로 대체 선택
    sel = select_stage("C", rows, parse_objective("FPS >= 1000 중 mAP 최대"), {}, pt)
    assert sel.status == "selected" and sel.fallback and sel.hash == "a"
    sel = select_stage("C", rows, parse_objective("PT 대비 mAP 하락 <= 0.1%p 중 FPS 최대"),
                       {}, {"map50": 0.9, "map50_95": 0.7})
    assert sel.fallback and sel.hash == "a" and "대체 선택" in sel.reason
    # 최고 mAP 와 0.1%p 이내면 같은 정확도로 보고 FPS 최대 (b: 0.49 vs a 0.50 → 1%p 차이라 제외)
    near = [dict(r) for r in rows]
    near[1]["tuning_map50_95"] = 0.4995  # a 보다 0.05%p 낮고 FPS 320
    sel = select_stage("C", near, parse_objective("PT 대비 mAP 하락 <= 0.1%p 중 FPS 최대"),
                       {}, {"map50": 0.9, "map50_95": 0.7})
    assert sel.fallback and sel.hash == "b"


def test_save_preset_keeps_comments(tmp_path: Path):
    from dxconv.sweep.store import save_preset

    (tmp_path / "presets").mkdir()
    p = tmp_path / "presets" / "fire.yaml"
    p.write_text("# fire 프리셋\n# 기존 calibration_num 320\ncompile:\n  calibration_num: 50\n")
    save_preset("fire", {"compile.calibration_method": "minmax", "export.opset": 18}, tmp_path,
                note="sweep DB: x")
    text = p.read_text()
    assert "# 기존 calibration_num 320" in text and "# sweep DB: x" in text
    data = yaml.safe_load(text)
    assert data == {"compile": {"calibration_num": 50, "calibration_method": "minmax"},
                    "export": {"opset": 18}}
    assert (tmp_path / "presets" / "fire.yaml.bak").is_file()


def test_recommend_update_and_layer(tmp_path: Path):
    """sweep 선택 결과 → recommend.yaml 갱신 → 설정 병합에 반영."""
    from dxconv.config import resolve_config
    from dxconv.recommend import update
    from dxconv.sweep.store import SweepStore

    cfgdir = tmp_path / "configs"
    shutil.copytree(REPO / "configs", cfgdir, ignore=shutil.ignore_patterns("*.local.yaml", "remote.yaml"))
    roots = []
    for name, num in (("s1", 60), ("s2", 60), ("s3", 30)):
        st = SweepStore(tmp_path / name)
        for stage, params in (("B", {"compile.calibration_method": "minmax", "compile.calibration_num": num}),
                              ("C", {"compile.opt_level": 1, "compile.aggressive_partitioning": True})):
            h = f"{name}{stage}"
            st.upsert(h, config={"compile": {"dxcom_version": "2.5.0"}}, status="compiled")
            st.add_member(stage, h, params, {}, 1)
            st.set_selection(stage, "selected", h, "x")
        st.close()
        roots.append(tmp_path / name)
    changed = update(roots, cfgdir)
    assert changed[9]["compile.calibration_num"] == (50, 60)  # 다수결
    assert changed[9]["compile.aggressive_partitioning"] == (False, True)
    cfg = resolve_config("anything", config_dir=cfgdir)
    assert cfg.compile.calibration_method == "minmax" and cfg.compile.aggressive_partitioning is True
    from dxconv.config import load_yaml

    v8_before = load_yaml(REPO / "configs/recommend.yaml")["formats"][8]["settings"]["compile"]
    cfg8 = resolve_config("anything", set_items=["compile.dxcom_version=2.4.1"], config_dir=cfgdir)
    assert cfg8.compile.calibration_method == v8_before["calibration_method"]  # v8 추천값은 그대로
    assert (cfgdir / "recommend.yaml.bak").is_file()


def test_select_calibration_by_mean():
    """B 단계에 사진 묶음이 있으면 방식별 평균으로 고르고, 그 방식 안에서 Tuning 최대 묶음을 쓴다."""
    def row(h: str, method: str, rep: int, m: float) -> dict:
        return {"hash": h, "status": "compiled", "tuning_map50_95": m,
                "params": {"compile.calibration_method": method, "compile.calib_repeat": rep}}

    # ema 는 한 번 운 좋게 최고(0.80)지만 평균은 minmax 가 높다
    rows = [row("e1", "ema", 1, 0.80), row("e2", "ema", 2, 0.70), row("e3", "ema", 3, 0.70),
            row("m1", "minmax", 1, 0.76), row("m2", "minmax", 2, 0.77), row("m3", "minmax", 3, 0.75)]
    sel = select_stage("B", rows, parse_objective(None), {}, {"map50_95": 0.8})
    assert sel.status == "selected" and sel.hash == "m2"
    assert "minmax" in sel.reason and "3묶음" in sel.reason
    # 묶음이 없으면 기존처럼 Tuning 최대
    plain = [{**r, "params": {"compile.calibration_method": r["params"]["compile.calibration_method"]}}
             for r in rows]
    assert select_stage("B", plain, parse_objective(None), {}, {}).hash == "e1"


def test_calibration_sample(env: dict):
    from dxconv.config import find_config_dir
    from dxconv.dataset import load_dataset

    ds = load_dataset("fire", find_config_dir())
    assert ds.calibration_sample(2, 0) == ds.calibration_images(2)
    a, b = ds.calibration_sample(2, 1), ds.calibration_sample(2, 1)
    assert a == b and len(a) == 2 and set(a) <= set(ds.calibration_images(3))  # 풀(3장)에서 재현 가능 추출
    assert {tuple(ds.calibration_sample(2, r)) for r in range(1, 8)} != {tuple(a)}  # 묶음마다 다름
    assert ds.calibration_sample(3, 2) == ds.calibration_images(3)  # 풀 = 장수 → 분할 목록 그대로


def test_remote_sweep_calibration_repeats(env: dict, monkeypatch: pytest.MonkeyPatch):
    """기본 공간 형태(B 만, 방식 × 묶음): 컴파일·평가 후 방식별 평균으로 선택, 최종 평가까지."""
    space = env["tmp"] / "space_b.yaml"
    space.write_text(yaml.safe_dump({"stages": {"B": {
        "compile.calibration_method": ["ema", "minmax"], "compile.calibration_num": [2],
        "compile.calib_repeat": [1, 2, 3]}}, "objective": "PT 대비 mAP 하락 <= 1%p 중 FPS 최대"}))
    _setup_remote(env, monkeypatch)
    res = runner.invoke(app, ["sweep", "--pt", str(env["pt"]), "--model", "fire", "--space", str(space),
                              "--out", str(env["out"]), "--remote", "npu",
                              "--set", "evaluate.fps_seconds=0.2"])
    assert res.exit_code == 0, res.output
    con = _db(env)
    sel = {r["stage"]: dict(r) for r in con.execute("SELECT * FROM selections")}
    assert list(sel) == ["B"] and sel["B"]["status"] == "selected"
    assert "묶음" in sel["B"]["reason"]
    members = con.execute("SELECT * FROM members WHERE stage='B'").fetchall()
    assert 2 <= len(members) <= 6  # 풀(3장)에서 2장 → 겹치는 묶음은 같은 조합으로 합쳐짐
    meta = {r["key"]: json.loads(r["value"]) for r in con.execute("SELECT * FROM meta")}
    assert meta["final_eval"]["images"] == 2
    # 고정값: 추천값 (opt1, aggressive off)
    cfg = json.loads(con.execute("SELECT config FROM combos LIMIT 1").fetchone()[0])
    assert cfg["compile"]["opt_level"] == 1 and cfg["compile"]["aggressive_partitioning"] is False


def test_host_cfg_defaults_and_python_pick(monkeypatch: pytest.MonkeyPatch):
    """장비 설정은 hostname·username 만 필수, 나머지는 기본값 (장비 계정·경로를 코드에 두지 않음)."""
    from dxconv.config import load_yaml
    from dxconv.remote.client import HostCfg, pick_python

    cfg = HostCfg(hostname="h", username="u")
    assert (cfg.port, cfg.workdir, cfg.python, cfg.dxrt_bin) == (22, "dxconv_agent", "auto", None)
    ex = load_yaml(REPO / "configs/remote.example.yaml")["hosts"]
    assert all(HostCfg.model_validate(v) for v in ex.values())
    # OpenCV 가 PC 와 같은 python 우선, 없으면 dx_engine 버전이 높은 것
    import cv2

    rows = [["/a/python", "0.0.1", "3.5.0"], ["/b/python", cv2.__version__, "3.4.0"]]
    assert pick_python(rows) == "/b/python"
    rows = [["/a/python", "0.0.1", "3.4.0"], ["/c/python", "0.0.2", "3.5.0"]]
    assert pick_python(rows) == "/c/python"
