"""평가 지표 / 기준선(baseline) 테스트."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import REPO
from typer.testing import CliRunner

from dxconv import DxconvError
from dxconv.cli import app
from dxconv.evaluate.baseline import letterbox_input, read_yolo_labels, unletterbox
from dxconv.evaluate.metrics import (
    ImageDets,
    box_iou,
    coco_metrics,
    pr_at_conf,
    ultralytics_metrics,
)

runner = CliRunner()


def _im(gt: list[list[float]], gt_cls: list[int], dets: list[list[float]]) -> ImageDets:
    """dets: [x1, y1, x2, y2, score, cls]."""
    d = np.array(dets, dtype=float).reshape(-1, 6)
    return ImageDets(np.array(gt, dtype=float).reshape(-1, 4), np.array(gt_cls, dtype=int),
                     d[:, :4], d[:, 4], d[:, 5].astype(int))


BOX = [10, 10, 50, 50]
FAR = [200, 200, 240, 240]


def test_box_iou():
    iou = box_iou(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]))
    assert np.allclose(iou, [[1.0, 50 / 150, 0.0]])


def test_perfect_predictions():
    ims = [_im([BOX, FAR], [0, 1], [[*BOX, 0.9, 0], [*FAR, 0.8, 1]])]
    c = coco_metrics(ims, 2)
    assert c["map50"] == pytest.approx(1.0) and c["map50_95"] == pytest.approx(1.0)
    assert c["ap50_95_per_class"] == [pytest.approx(1.0), pytest.approx(1.0)]
    u = ultralytics_metrics(ims, 2)
    # ultralytics AP: 정밀도 곡선 끝 sentinel 때문에 완벽한 예측도 0.995 였으나, 8.4.x 일부 버전은 1.0
    assert u["map50"] == pytest.approx(0.995, abs=1e-3) or u["map50"] == pytest.approx(1.0)
    pr = pr_at_conf(ims)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (2, 0, 0)


def test_coco_ranking_matters():
    # TP 가 먼저면 AP 1, FP 가 먼저면 recall 1 에서 precision 0.5 → AP 0.5
    good = [_im([BOX], [0], [[*BOX, 0.9, 0], [*FAR, 0.8, 0]])]
    bad = [_im([BOX], [0], [[*FAR, 0.9, 0], [*BOX, 0.8, 0]])]
    assert coco_metrics(good, 1)["map50"] == pytest.approx(1.0)
    assert coco_metrics(bad, 1)["map50"] == pytest.approx(0.5)


def test_coco_max_dets_100():
    # 점수가 더 높은 FP 100 개 뒤의 TP 는 COCO(maxDets 100)에서 잘리고, ultralytics 에서는 남는다
    fps = [[300 + i, 300, 310 + i, 310, 0.9, 0] for i in range(100)]
    ims = [_im([BOX], [0], [*fps, [*BOX, 0.5, 0]])]
    assert coco_metrics(ims, 1)["map50"] == pytest.approx(0.0)
    assert coco_metrics(ims, 1)["ar50_per_class"] == [0.0]
    assert ultralytics_metrics(ims, 1)["map50"] > 0


def test_class_without_gt_is_excluded():
    ims = [_im([BOX], [0], [[*BOX, 0.9, 0], [*FAR, 0.9, 1]])]
    c = coco_metrics(ims, 2)
    assert c["ap50_95_per_class"][1] is None
    assert c["map50"] == pytest.approx(1.0)


def test_iou_threshold_levels():
    # IoU 0.64 (가로 40 중 겹침 32): 0.5~0.6 에서만 맞음 → AP50 1, AP50-95 = 3/10
    shifted = [18, 10, 58, 50]
    ims = [_im([BOX], [0], [[*shifted, 0.9, 0]])]
    assert box_iou(np.array([BOX]), np.array([shifted]))[0, 0] == pytest.approx(32 / 48)
    c = coco_metrics(ims, 1)
    assert c["map50"] == pytest.approx(1.0)
    assert c["map50_95"] == pytest.approx(0.4)  # 0.50, 0.55, 0.60, 0.65 통과 (IoU 0.667)


def test_pr_at_conf_threshold_and_duplicates():
    ims = [_im([BOX], [0], [[*BOX, 0.9, 0], [*BOX, 0.8, 0], [*FAR, 0.1, 0]])]
    pr = pr_at_conf(ims, conf=0.25)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (1, 1, 0)  # 중복 검출은 FP, conf 0.1 은 제외
    assert pr["precision"] == pytest.approx(0.5) and pr["recall"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 전처리 / 라벨
# --------------------------------------------------------------------------- #
def _noise(path: Path, w: int, h: int) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(w * h).integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
    return path


@pytest.mark.parametrize("w, h", [(1280, 720), (480, 800), (333, 251), (640, 640)])
def test_letterbox_input_roundtrip(tmp_path: Path, w: int, h: int):
    lb = letterbox_input(_noise(tmp_path / "a.png", w, h), 640, 640)
    assert lb.tensor.shape == (1, 3, 640, 640) and lb.tensor.dtype == np.float32
    assert 0.0 <= lb.tensor.min() and lb.tensor.max() <= 1.0
    # 원본 좌표 → 입력 좌표 → 원본 좌표
    orig = np.array([[0, 0, w, h], [w * 0.25, h * 0.5, w * 0.75, h * 0.9]])
    inp = orig * lb.ratio + np.array([lb.left, lb.top, lb.left, lb.top])
    assert np.allclose(unletterbox(inp, lb), orig, atol=1e-6)
    # 패딩 영역은 114/255
    if lb.top > 0:
        assert np.allclose(lb.tensor[0, :, 0, :], 114 / 255)


def test_read_yolo_labels(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.2 0.2\n")
    boxes, cls = read_yolo_labels(p, (100, 200))
    assert np.allclose(boxes[0], [80, 30, 120, 70]) and list(cls) == [0, 1]
    (tmp_path / "empty.txt").write_text("")
    assert len(read_yolo_labels(tmp_path / "empty.txt", (10, 10))[0]) == 0
    assert len(read_yolo_labels(tmp_path / "missing.txt", (10, 10))[0]) == 0


# --------------------------------------------------------------------------- #
# baseline CLI (고정 출력 ONNX + 가짜 PT 러너)
# --------------------------------------------------------------------------- #
def _const_onnx(path: Path, dets: np.ndarray) -> Path:
    """입력과 무관하게 dets [1, N, 6] 을 내는 end2end 형태 ONNX."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(dets.shape))
    const = numpy_helper.from_array(dets.astype(np.float32), "dets")
    zero = numpy_helper.from_array(np.zeros(1, np.float32), "zero")
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["dets", "z"], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "g", [inp], [out], [const, zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


@pytest.fixture
def baseline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """640x640 이미지 6장(tuning 3 / evaluation 3), 각 이미지에 GT 1개 (가운데 상자)."""
    cfgdir = tmp_path / "configs"
    shutil.copytree(REPO / "configs", cfgdir, ignore=shutil.ignore_patterns("*.local.yaml"))
    monkeypatch.setenv("DXCONV_CONFIG_DIR", str(cfgdir))
    root = tmp_path / "data"
    names = []
    for i in range(6):
        _noise(root / "images" / f"im{i}.png", 640, 640 - i)  # 내용이 서로 다르게
        (root / "labels").mkdir(parents=True, exist_ok=True)
        (root / "labels" / f"im{i}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        names.append(f"im{i}.png")
    (root / "tuning.txt").write_text("\n".join(names[:3]))
    (root / "evaluation.txt").write_text("\n".join(names[3:]))
    (root / "cal.txt").write_text("\n".join(names[:1]))
    ds = root / "ds.yaml"
    ds.write_text(yaml.safe_dump({"name": "fire", "images_dir": "images", "labels_dir": "labels",
                                  "calibration": {1: "cal.txt"}, "tuning": "tuning.txt",
                                  "evaluation": "evaluation.txt"}))
    # 가운데 상자(입력 좌표) + 낮은 점수의 엉뚱한 상자 + conf 미만 상자
    dets = np.zeros((1, 300, 6), np.float32)
    dets[0, 0] = [240, 240, 400, 400, 0.9, 0]
    dets[0, 1] = [0, 0, 20, 20, 0.3, 1]
    dets[0, 2] = [100, 100, 120, 120, 0.0005, 0]
    onnx = _const_onnx(tmp_path / "m.onnx", dets)

    class FakePt:
        kind = "pt"
        hw = (640, 640)
        net = type("N", (), {"names": {0: "fire", 1: "smoke"}})()

        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return dets[0]

    monkeypatch.setattr("dxconv.evaluate.baseline.PtRunner", FakePt)
    pt = tmp_path / "fire.pt"
    pt.write_bytes(b"fake")
    return {"ds": ds, "onnx": onnx, "pt": pt, "out": tmp_path / "runs"}


def test_baseline_cli(baseline_env: dict):
    e = baseline_env
    res = runner.invoke(app, ["baseline", "--pt", str(e["pt"]), "--model", "fire",
                              "--onnx", str(e["onnx"]), "--dataset", str(e["ds"]),
                              "--out", str(e["out"])])
    assert res.exit_code == 0, res.output
    run = next((e["out"] / "baseline").iterdir())
    rep = json.loads((run / "report.json").read_text())
    assert rep["success"] is True
    assert rep["eval_settings"]["conf"] == 0.001 and rep["eval_settings"]["pr_conf"] == 0.25
    for split in ("tuning", "evaluation"):
        for model in ("pt", "onnx"):
            r = rep["baseline"][split][model]
            assert r["images"] == 3 and r["gt_objects"] == 3
            assert r["predictions"] == 6  # conf 0.001 미만 상자는 제외
            assert r["coco"]["map50_95"] == pytest.approx(1.0)
            assert r["coco"]["ap50_95_per_class"][1] is None  # smoke GT 없음
            assert (r["pr_at_conf"]["tp"], r["pr_at_conf"]["fp"]) == (3, 3)
            assert Path(r["predictions_file"]).is_file()
    table = (run / "baseline_table.md").read_text()
    assert "DXNN mAP50-95" in table and "Δ DXNN vs PT" in table
    assert rep["inputs"]["onnx"]["onnx"]["sha256"]


def test_baseline_split_option_and_errors(baseline_env: dict):
    e = baseline_env
    res = runner.invoke(app, ["baseline", "--pt", str(e["pt"]), "--model", "fire",
                              "--onnx", str(e["onnx"]), "--dataset", str(e["ds"]),
                              "--split", "evaluation", "--out", str(e["out"])])
    assert res.exit_code == 0, res.output
    run = next((e["out"] / "baseline").iterdir())
    assert list(json.loads((run / "report.json").read_text())["baseline"]) == ["evaluation"]
    bad = runner.invoke(app, ["baseline", "--pt", str(e["pt"]), "--model", "fire",
                              "--onnx", str(e["onnx"]), "--dataset", str(e["ds"]),
                              "--split", "test", "--out", str(e["out"])])
    assert bad.exit_code == 1 and "알 수 없는 --split" in bad.output


def test_device_predictions_roundtrip(tmp_path):
    """eval_map.py 형식(입력 좌표 + letterbox 정보) → 원본 좌표로 되돌려 GT 와 완전 일치."""
    from dxconv.evaluate.npu import summarize_device

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    # 원본 320x640(h,w) → 640 입력: r=1, top=160 패딩
    h, w, r, left, top = 320, 640, 1.0, 0, 160
    x1, y1, x2, y2 = 0.4 * w, 0.3 * h, 0.6 * w, 0.7 * h
    doc = {"results": [{"file": "a.jpg", "orig_hw": [h, w], "ratio": r, "left": left, "top": top,
                        "dets": [[x1 * r + left, y1 * r + top, x2 * r + left, y2 * r + top, 0.9, 0]]}],
           "missing": []}
    s = summarize_device(doc, labels, 2, expected=["a.jpg"])
    assert s["coco"]["map50_95"] == pytest.approx(1.0)
    with pytest.raises(DxconvError, match="분할 불일치"):
        summarize_device(doc, labels, 2, expected=["a.jpg", "b.jpg"])
