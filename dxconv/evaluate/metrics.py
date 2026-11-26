"""검출 평가 지표.

세 가지 정의를 함께 계산한다 (기존 실험과 같은 정의로 비교하기 위해):

- COCO 방식 (`coco_metrics`): pycocotools COCOeval(bbox, areaRng=all, maxDets=100)과 같은 계산을 numpy 로 구현.
  기존 Evaluation 600 평가(02_Calibration_Accuracy/10_dxnn_eval)가 이 정의다.
- Ultralytics 방식 (`ultralytics_metrics`): `ultralytics val` 과 같은 매칭/AP 계산
  (설치된 ultralytics 의 match_predictions, ap_per_class 를 그대로 호출). P/R 은 평균 F1 최대 conf 에서의 값.
  기존 Tuning 200 FP32 확인(04_fp32_check)이 이 정의다.
- 고정 conf P/R (`pr_at_conf`): conf ≥ 0.25 예측만, IoU ≥ 0.5, 점수 순 greedy 매칭의 TP/FP/FN.

좌표는 모두 원본 이미지 픽셀 기준 xyxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
REC_THRESHOLDS = np.linspace(0.0, 1.0, 101)
COCO_MAX_DETS = 100


@dataclass
class ImageDets:
    """이미지 한 장의 GT 와 예측 (원본 좌표 xyxy)."""

    gt_boxes: np.ndarray  # (M, 4)
    gt_cls: np.ndarray  # (M,)
    boxes: np.ndarray  # (N, 4)
    scores: np.ndarray  # (N,)
    cls: np.ndarray  # (N,)


@dataclass
class EvalResult:
    """평가 결과. device_agent 가 반환하는 JSON 과 같은 구조."""

    map50: float | None = None
    map50_95: float | None = None
    fps: float | None = None
    latency_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        known = {k: d.get(k) for k in ("map50", "map50_95", "fps", "latency_ms")}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**known, extra=extra)


def map_drop_pp(baseline: EvalResult, candidate: EvalResult) -> float | None:
    """기준 대비 mAP50-95 하락폭 (%p, 양수 = 하락)."""
    if baseline.map50_95 is None or candidate.map50_95 is None:
        return None
    return (baseline.map50_95 - candidate.map50_95) * 100.0


# --------------------------------------------------------------------------- #
# 공통
# --------------------------------------------------------------------------- #
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy → (N,M) IoU. 넓이는 (x2-x1)*(y2-y1) (pycocotools 와 같은 연속 좌표)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
def _coco_eval_image(gt: np.ndarray, dt: np.ndarray, scores: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """COCOeval.evaluateImg (iscrowd 없음, ignore 없음). (dt 점수, dt 매칭 여부[T, D]) 반환."""
    order = np.argsort(-scores, kind="mergesort")[:COCO_MAX_DETS]
    dt, scores = dt[order], scores[order]
    ious = box_iou(dt, gt)
    matched = np.zeros((len(IOU_THRESHOLDS), len(dt)), dtype=bool)
    for ti, t in enumerate(IOU_THRESHOLDS):
        gt_used = np.zeros(len(gt), dtype=bool)
        for d in range(len(dt)):
            best, m = min(t, 1 - 1e-10), -1
            for g in range(len(gt)):
                if gt_used[g] or ious[d, g] < best:
                    continue
                best, m = ious[d, g], g
            if m >= 0:
                gt_used[m] = True
                matched[ti, d] = True
    return scores, matched


def coco_metrics(images: list[ImageDets], num_classes: int) -> dict[str, Any]:
    """COCO bbox AP (AP@[.5:.95], AP50) 와 클래스별 AP, AR@IoU0.5(maxDets 100)."""
    ap = np.full((len(IOU_THRESHOLDS), num_classes), -1.0)
    ar50 = np.full(num_classes, -1.0)
    for c in range(num_classes):
        all_scores: list[np.ndarray] = []
        all_matched: list[np.ndarray] = []
        npig = 0
        for im in images:
            gt = im.gt_boxes[im.gt_cls == c]
            sel = im.cls == c
            npig += len(gt)
            if len(gt) == 0 and not sel.any():
                continue
            s, m = _coco_eval_image(gt, im.boxes[sel], im.scores[sel])
            all_scores.append(s)
            all_matched.append(m)
        if npig == 0:
            continue
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        matched = (np.concatenate(all_matched, axis=1) if all_matched
                   else np.zeros((len(IOU_THRESHOLDS), 0), dtype=bool))
        order = np.argsort(-scores, kind="mergesort")
        tps = np.cumsum(matched[:, order], axis=1).astype(np.float64)
        fps = np.cumsum(~matched[:, order], axis=1).astype(np.float64)
        for ti in range(len(IOU_THRESHOLDS)):
            tp, fp = tps[ti], fps[ti]
            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))
            q = np.zeros(len(REC_THRESHOLDS))
            if len(tp):
                pr = pr.tolist()
                for i in range(len(pr) - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]
                inds = np.searchsorted(rc, REC_THRESHOLDS, side="left")
                for ri, pi in enumerate(inds):
                    if pi < len(pr):
                        q[ri] = pr[pi]
            ap[ti, c] = q.mean()
            if ti == 0:
                ar50[c] = rc[-1] if len(rc) else 0.0
    valid = ap[0] > -1
    per_class_ap = ap.mean(0)
    return {
        "map50": float(ap[0, valid].mean()) if valid.any() else None,
        "map50_95": float(ap[:, valid].mean()) if valid.any() else None,
        "ap50_per_class": [float(x) if v else None for x, v in zip(ap[0], valid, strict=True)],
        "ap50_95_per_class": [float(x) if v else None
                              for x, v in zip(per_class_ap, valid, strict=True)],
        "ar50_per_class": [float(x) if x > -1 else None for x in ar50],
    }


# --------------------------------------------------------------------------- #
# Ultralytics
# --------------------------------------------------------------------------- #
def ultralytics_metrics(images: list[ImageDets], num_classes: int) -> dict[str, Any]:
    """`ultralytics val` 과 같은 계산 (설치된 ultralytics 의 함수 사용)."""
    from types import SimpleNamespace

    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class

    fake = SimpleNamespace(iouv=torch.from_numpy(IOU_THRESHOLDS))
    tps, confs, pcls, tcls = [], [], [], []
    for im in images:
        tcls.append(im.gt_cls)
        if len(im.boxes) == 0:
            continue
        order = np.argsort(-im.scores, kind="stable")
        boxes, scores, cls = im.boxes[order], im.scores[order], im.cls[order]
        if len(im.gt_boxes) == 0:
            correct = np.zeros((len(boxes), len(IOU_THRESHOLDS)), dtype=bool)
        else:
            iou = torch.from_numpy(box_iou(im.gt_boxes, boxes))
            correct = BaseValidator.match_predictions(
                fake, torch.from_numpy(cls.astype(np.int64)),
                torch.from_numpy(im.gt_cls.astype(np.int64)), iou).numpy()
        tps.append(correct)
        confs.append(scores)
        pcls.append(cls)
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    pred_cls = np.concatenate(pcls) if pcls else np.zeros(0)
    target_cls = np.concatenate(tcls) if tcls else np.zeros(0)
    res = ap_per_class(tp, conf, pred_cls, target_cls)
    p, r, ap, uniq = res[2], res[3], res[5], res[6]
    ap50_c: list[float | None] = [None] * num_classes
    ap_c: list[float | None] = [None] * num_classes
    p_c: list[float | None] = [None] * num_classes
    r_c: list[float | None] = [None] * num_classes
    for i, c in enumerate(uniq.astype(int)):
        ap50_c[c], ap_c[c] = float(ap[i, 0]), float(ap[i].mean())
        p_c[c], r_c[c] = float(p[i]), float(r[i])
    return {
        "map50": float(ap[:, 0].mean()) if len(ap) else None,
        "map50_95": float(ap.mean()) if len(ap) else None,
        "precision": float(p.mean()) if len(p) else None,
        "recall": float(r.mean()) if len(r) else None,
        "ap50_per_class": ap50_c,
        "ap50_95_per_class": ap_c,
        "precision_per_class": p_c,
        "recall_per_class": r_c,
    }


# --------------------------------------------------------------------------- #
# 고정 conf P/R
# --------------------------------------------------------------------------- #
def pr_at_conf(images: list[ImageDets], conf: float = 0.25, iou: float = 0.5) -> dict[str, Any]:
    """conf 이상 예측만 사용, 이미지·클래스별로 점수 순 greedy 매칭 (IoU ≥ iou)."""
    tp = fp = fn = 0
    for im in images:
        keep = im.scores >= conf
        boxes, scores, cls = im.boxes[keep], im.scores[keep], im.cls[keep]
        for c in np.union1d(np.unique(cls), np.unique(im.gt_cls)):
            gt = im.gt_boxes[im.gt_cls == c]
            sel = cls == c
            dt = boxes[sel][np.argsort(-scores[sel], kind="mergesort")]
            ious = box_iou(dt, gt)
            used = np.zeros(len(gt), dtype=bool)
            for d in range(len(dt)):
                cand = np.where(used, -1.0, ious[d]) if len(gt) else np.zeros(0)
                if len(cand) and cand.max() >= iou:
                    used[int(cand.argmax())] = True
                    tp += 1
                else:
                    fp += 1
            fn += int((~used).sum())
    return {
        "conf": conf, "iou": iou, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
