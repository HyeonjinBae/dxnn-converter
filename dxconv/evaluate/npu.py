"""NPU 장비 추론 결과(device_agent/eval_map.py 출력) → 호스트에서 mAP 계산.

장비는 추론만 하고 입력 좌표 예측과 letterbox 정보를 돌려준다. 좌표 역변환과 지표 계산은
기준선 평가(baseline.py)와 같은 코드로 한다 → PT / ONNX / DXNN 수치가 같은 방식으로 계산된다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dxconv import DxconvError
from dxconv.evaluate.baseline import read_yolo_labels, summarize
from dxconv.evaluate.metrics import ImageDets


def device_dets(doc: dict[str, Any], labels_dir: Path, max_det: int = 300) -> list[ImageDets]:
    out: list[ImageDets] = []
    for r in doc.get("results", []):
        d = np.asarray(r["dets"], dtype=np.float64).reshape(-1, 6)
        d = d[np.argsort(-d[:, 4], kind="stable")][:max_det]
        h, w = r["orig_hw"]
        boxes = d[:, :4].copy()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - r["left"]) / r["ratio"]).clip(0, w)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - r["top"]) / r["ratio"]).clip(0, h)
        gt_boxes, gt_cls = read_yolo_labels(labels_dir / f"{Path(r['file']).stem}.txt", (h, w))
        out.append(ImageDets(gt_boxes, gt_cls, boxes, d[:, 4], d[:, 5].astype(np.int64)))
    return out


def summarize_device(path_or_doc: Path | dict[str, Any], labels_dir: Path, num_classes: int, *,
                     pr_conf: float = 0.25, max_det: int = 300,
                     expected: list[str] | None = None) -> dict[str, Any]:
    """eval_map.py JSON → summarize() 형식 (+ 장비 정보). expected: 분할의 이미지 이름 목록."""
    doc = (json.loads(Path(path_or_doc).read_text(encoding="utf-8"))
           if not isinstance(path_or_doc, dict) else path_or_doc)
    if doc.get("missing"):
        raise DxconvError(f"장비에서 읽지 못한 이미지 {len(doc['missing'])}개: {doc['missing'][:5]}")
    if expected is not None:
        got = {r["file"] for r in doc.get("results", [])}
        lack = sorted(set(expected) - got)
        if lack:
            raise DxconvError(f"장비 결과에 없는 이미지 {len(lack)}개 (분할 불일치): {lack[:5]}")
    s = summarize(device_dets(doc, labels_dir, max_det), num_classes, pr_conf)
    s["device"] = {k: doc.get(k) for k in ("dxnn", "model_version", "dx_engine", "opencv", "host", "conf")}
    try:
        import cv2

        if doc.get("opencv") and doc["opencv"] != cv2.__version__:
            s["warnings"] = [f"장비 OpenCV {doc['opencv']} ≠ 호스트 {cv2.__version__}: letterbox 결과가 달라 "
                             "mAP 가 조금 달라질 수 있습니다 (캘리브레이션은 호스트 버전으로 처리)."]
    except ImportError:
        pass
    s["mean_infer_ms"] = round((doc.get("timing_ms") or {}).get("run_mean", 0.0), 2)
    return s
