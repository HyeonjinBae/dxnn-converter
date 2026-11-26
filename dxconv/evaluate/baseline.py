"""PT / FP32 ONNX 기준선(baseline) mAP 평가.

- 입력: 데이터셋 설정(configs/datasets/<model>.local.yaml)의 tuning / evaluation 목록 + YOLO 라벨
- 전처리: Ultralytics LetterBox (정적 입력 크기, 가운데 패딩 114) → BGR→RGB → /255 → NCHW.
  캘리브레이션 host letterbox 와 같은 함수(dxconv.compile.calib)를 쓴다.
- 추론:
  - ONNX: onnxruntime (CPU). end2end 출력 [1, max_det, 6] = x1,y1,x2,y2,score,cls (입력 좌표)
  - PT: Ultralytics 모델 forward. export.end2end=True 이면 Detect head 의 end2end 를 켜서
        ONNX 와 같은 one2one 출력을 쓴다 (`YOLO.predict` 는 이 모델에서 NMS 경로를 타서 결과가 다르다).
- mAP 용 conf 는 evaluate.conf (기본 0.001). 배포용 postprocess.conf (0.25) 는 고정 conf P/R 에만 쓴다.
- 지표: COCO 방식 / Ultralytics 방식 / conf 0.25 P·R (dxconv.evaluate.metrics)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dxconv import DxconvError
from dxconv.compile.calib import LetterboxSpec, read_bgr, ultralytics_letterbox
from dxconv.evaluate.metrics import ImageDets, coco_metrics, pr_at_conf, ultralytics_metrics


@dataclass
class Letterboxed:
    tensor: np.ndarray  # (1, 3, H, W) float32 RGB 0~1
    ratio: float
    left: int
    top: int
    orig_hw: tuple[int, int]


def letterbox_input(path: Path, height: int, width: int, pad_value: int = 114) -> Letterboxed:
    img = read_bgr(path)
    h, w = img.shape[:2]
    lb = ultralytics_letterbox(LetterboxSpec(height, width, pad_value))(image=img)
    r = min(height / h, width / w)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    left, top = round((width - nw) / 2 - 0.1), round((height - nh) / 2 - 0.1)
    x = lb[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return Letterboxed(np.ascontiguousarray(x), r, left, top, (h, w))


def unletterbox(boxes: np.ndarray, lb: Letterboxed) -> np.ndarray:
    """입력 좌표 xyxy → 원본 좌표 xyxy (원본 범위로 자름)."""
    out = boxes.astype(np.float64).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - lb.left) / lb.ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - lb.top) / lb.ratio
    h, w = lb.orig_hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def read_yolo_labels(path: Path, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """YOLO 라벨(cls xc yc w h, 0~1) → 원본 좌표 xyxy, cls. 파일이 없거나 비면 GT 없음."""
    h, w = hw
    rows = []
    if path.is_file():
        for ln in path.read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts[:5]])
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
    a = np.array(rows)
    xc, yc, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    boxes = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)
    return boxes, a[:, 0].astype(np.int64)


# --------------------------------------------------------------------------- #
# 러너
# --------------------------------------------------------------------------- #
class OnnxRunner:
    kind = "onnx"

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.path = path
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) != 4 or not all(isinstance(v, int) for v in shape[2:]):
            raise DxconvError(f"ONNX 입력이 정적 NCHW 가 아닙니다: {shape}")
        self.hw = (int(shape[2]), int(shape[3]))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.sess.run(None, {self.input_name: x})[0]
        if out.ndim != 3 or out.shape[-1] != 6:
            raise DxconvError(f"end2end 출력 [1, N, 6] 이 아닙니다: {out.shape}")
        return out[0]


class PtRunner:
    kind = "pt"

    def __init__(self, path: Path, imgsz: int, end2end: bool) -> None:
        import torch
        from ultralytics import YOLO

        self.path = path
        self.torch = torch
        self.net = YOLO(str(path)).model.float().eval()
        head = self.net.model[-1]
        if end2end:
            if not hasattr(head, "end2end"):
                raise DxconvError("이 PT 모델의 head 에 end2end 속성이 없습니다.")
            head.end2end = True
        self.end2end = bool(getattr(head, "end2end", False))
        if not self.end2end:
            raise DxconvError("PT 기준선은 end2end head 출력만 지원합니다 (export.end2end=true).")
        self.hw = (imgsz, imgsz)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            y = self.net(self.torch.from_numpy(x))
        y = y[0] if isinstance(y, (list, tuple)) else y
        return y[0].cpu().numpy()


# --------------------------------------------------------------------------- #
# 평가
# --------------------------------------------------------------------------- #
def predict_split(
    runner: Any, images: list[Path], labels_dir: Path, *, conf: float, max_det: int,
    pad_value: int = 114,
) -> tuple[list[ImageDets], list[dict[str, Any]], float]:
    """(이미지별 GT/예측, COCO 형식 예측 목록, 이미지당 평균 추론 ms)."""
    dets: list[ImageDets] = []
    coco: list[dict[str, Any]] = []
    total = 0.0
    for i, path in enumerate(images, 1):
        lb = letterbox_input(path, *runner.hw, pad_value=pad_value)
        t0 = time.perf_counter()
        out = runner(lb.tensor)
        total += time.perf_counter() - t0
        out = out[out[:, 4] >= conf]
        out = out[np.argsort(-out[:, 4], kind="stable")][:max_det]
        boxes = unletterbox(out[:, :4], lb)
        gt_boxes, gt_cls = read_yolo_labels(labels_dir / f"{path.stem}.txt", lb.orig_hw)
        dets.append(ImageDets(gt_boxes, gt_cls, boxes, out[:, 4].astype(np.float64),
                              out[:, 5].astype(np.int64)))
        for b, s, c in zip(boxes, out[:, 4], out[:, 5], strict=True):
            coco.append({"image_id": i, "file_name": path.name, "category_id": int(c),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                         "score": float(s)})
    return dets, coco, total / max(len(images), 1) * 1000


def summarize(dets: list[ImageDets], num_classes: int, pr_conf: float) -> dict[str, Any]:
    n_gt = int(sum(len(d.gt_cls) for d in dets))
    return {
        "images": len(dets),
        "gt_objects": n_gt,
        "predictions": int(sum(len(d.scores) for d in dets)),
        "coco": coco_metrics(dets, num_classes),
        "ultralytics": ultralytics_metrics(dets, num_classes),
        "pr_at_conf": pr_at_conf(dets, conf=pr_conf, iou=0.5),
    }


def write_predictions(coco: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 표
# --------------------------------------------------------------------------- #
def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.2f}"


def results_table(results: dict[str, dict[str, Any]], names: list[str],
                  reference: str = "pt") -> str:
    """마크다운 표. results[split][model] = summarize() 결과.

    DXNN 칸(mAP50-95, Δ)은 비워 둔다 (NPU 평가는 convert/sweep --remote 에서).
    """
    cls_cols = " | ".join(f"AP50-95 {n}" for n in names)
    head = (f"| 분할 | 모델 | 예측 수 | mAP50 | mAP50-95 | {cls_cols} | P@conf | R@conf "
            f"| UL mAP50 | UL mAP50-95 | UL P | UL R | Δ mAP50-95 vs {reference.upper()} (%p) "
            "| DXNN mAP50-95 | Δ DXNN vs PT (%p) |")
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [head, sep]
    for split, models in results.items():
        ref = models.get(reference)
        for model, r in models.items():
            c, u, pr = r["coco"], r["ultralytics"], r["pr_at_conf"]
            per = " | ".join(_pct(v) for v in c["ap50_95_per_class"])
            delta = ("—" if ref is None or model == reference
                     else f"{(c['map50_95'] - ref['coco']['map50_95']) * 100:+.2f}")
            lines.append(
                f"| {split} | {model.upper()} | {r['predictions']} | {_pct(c['map50'])} "
                f"| {_pct(c['map50_95'])} | {per} | {_pct(pr['precision'])} | {_pct(pr['recall'])} "
                f"| {_pct(u['map50'])} | {_pct(u['map50_95'])} | {_pct(u['precision'])} "
                f"| {_pct(u['recall'])} | {delta} |  |  |"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# FP32 ONNX ↔ PT 일치 확인 (sweep A 단계)
# --------------------------------------------------------------------------- #
def compare_outputs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """두 검출 집합 [N, 6] 의 최대 좌표 차이(px)와 최대 점수 차이.

    점수가 비슷한 검출은 순서가 바뀔 수 있어, a 의 각 검출마다 같은 클래스의 b 중
    좌표가 가장 가까운 것과 비교한다 (양방향 중 큰 값).
    """
    def one_way(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        box, score = 0.0, 0.0
        for d in x:
            cand = y[y[:, 5] == d[5]]
            if len(cand) == 0:
                return float("inf"), float("inf")
            dist = np.abs(cand[:, :4] - d[:4]).max(1)
            j = int(dist.argmin())
            box, score = max(box, float(dist[j])), max(score, float(abs(cand[j, 4] - d[4])))
        return box, score

    if len(a) != len(b):
        return float("inf"), float("inf")
    if len(a) == 0:
        return 0.0, 0.0
    b1, s1 = one_way(a, b)
    b2, s2 = one_way(b, a)
    return max(b1, b2), max(s1, s2)


def check_onnx_vs_pt(
    pt_outputs: dict[Path, np.ndarray], onnx_runner: Any, *, conf: float = 0.001,
    box_tol: float = 0.5, score_tol: float = 1e-3, pad_value: int = 114,
) -> dict[str, Any]:
    """PT 출력(미리 계산)과 FP32 ONNX 출력이 같은지. 좌표 box_tol px, 점수 score_tol 이내면 통과."""
    worst_box = worst_score = 0.0
    count_mismatch = 0
    for path, pt_out in pt_outputs.items():
        lb = letterbox_input(path, *onnx_runner.hw, pad_value=pad_value)
        out = onnx_runner(lb.tensor)
        out = out[out[:, 4] >= conf]
        if len(out) != len(pt_out):
            count_mismatch += 1
            continue
        bd, sd = compare_outputs(out, pt_out)
        worst_box, worst_score = max(worst_box, bd), max(worst_score, sd)
    ok = count_mismatch == 0 and worst_box <= box_tol and worst_score <= score_tol
    reason = ("" if ok else f"검출 수 다름 {count_mismatch}장" if count_mismatch
              else f"좌표 차 {worst_box:.3g}px / 점수 차 {worst_score:.3g}")
    return {"ok": ok, "images": len(pt_outputs), "count_mismatch": count_mismatch,
            "max_box_diff": worst_box, "max_score_diff": worst_score,
            "box_tol": box_tol, "score_tol": score_tol, "reason": reason}


def pt_reference_outputs(pt_runner: Any, images: list[Path], *, conf: float = 0.001,
                         pad_value: int = 114) -> dict[Path, np.ndarray]:
    out: dict[Path, np.ndarray] = {}
    for p in images:
        lb = letterbox_input(p, *pt_runner.hw, pad_value=pad_value)
        y = pt_runner(lb.tensor)
        out[p] = y[y[:, 4] >= conf]
    return out
