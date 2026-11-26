#!/usr/bin/env python3
"""NPU 장비에서 DXNN 추론 → 이미지별 예측 JSON (mAP 계산은 호스트 dxconv 가 한다).

사용 예 (장비에서):
    python eval_map.py --dxnn fire.dxnn --images-dir data/fire/images \
        --list tuning_200.txt --conf 0.001 --out pred.json

- 전처리: Ultralytics LetterBox(auto=False, scale_fill=False, scaleup=True, center=True,
  INTER_LINEAR, 114) 와 같은 계산 → BGR→RGB → uint8 NHWC [1, H, W, 3].
  DX-COM 이 /255 와 transpose 는 NPU 그래프에 넣으므로 장비에서는 하지 않는다
  (dxconv 가 만든 dxnn 의 입력 dtype 이 uint8 인지 실행 시 확인).
- 출력: end2end [1, max_det, 6] (x1, y1, x2, y2, score, cls, 입력 좌표) 중 score >= conf.
  좌표 역변환·mAP 는 호스트에서 기준선 평가와 같은 코드로 계산한다 (계산 방식 일치).
- 이 스크립트는 dxconv 패키지에 의존하지 않는다 (장비에 단독 배포). 필요: dx_engine, numpy, opencv.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def letterbox(img: np.ndarray, hw: tuple[int, int], pad: int = 114) -> tuple[np.ndarray, dict]:
    """Ultralytics LetterBox(auto=False, scaleup=True, center=True) 와 같은 계산."""
    h0, w0 = img.shape[:2]
    H, W = hw
    r = min(H / h0, W / w0)
    nw, nh = max(round(w0 * r), 1), max(round(h0 * r), 1)
    dw, dh = (W - nw) / 2, (H - nh) / 2
    if (w0, h0) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad,) * 3)
    return img, {"orig_hw": [h0, w0], "ratio": r, "left": left, "top": top}


def list_images(images_dir: Path, list_file: Path | None) -> list[Path]:
    if list_file is not None:
        names = [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip()]
        return [images_dir / n for n in names]
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def dxrt_version() -> str | None:
    try:
        import dx_engine

        return getattr(dx_engine, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dxnn", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--list", default=None, help="이미지 이름 목록 파일 (없으면 폴더 전체)")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--pad", type=int, default=114)
    ap.add_argument("--out", default="-")
    a = ap.parse_args()

    from dx_engine import InferenceEngine

    ie = InferenceEngine(a.dxnn)
    tin = ie.get_input_tensors_info()[0]
    tout = ie.get_output_tensors_info()[0]
    shape = list(tin["shape"])
    if len(shape) != 4 or shape[3] != 3 or np.dtype(tin["dtype"]) != np.uint8:
        sys.exit(f"지원하지 않는 입력: {tin} (uint8 NHWC [1,H,W,3] 만 지원)")
    if len(tout["shape"]) != 3 or tout["shape"][-1] != 6:
        sys.exit(f"end2end 출력 [1, N, 6] 이 아닙니다: {tout}")
    hw = (int(shape[1]), int(shape[2]))

    images = list_images(Path(a.images_dir), Path(a.list) if a.list else None)
    results, missing = [], []
    t_pre = t_inf = 0.0
    t_all = time.perf_counter()
    for p in images:
        t0 = time.perf_counter()
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            missing.append(p.name)
            continue
        lb, meta = letterbox(img, hw, a.pad)
        x = np.ascontiguousarray(lb[..., ::-1])[None]  # BGR→RGB, NHWC
        t1 = time.perf_counter()
        out = np.asarray(ie.run([x])[0])[0]
        t2 = time.perf_counter()
        t_pre += t1 - t0
        t_inf += t2 - t1
        out = out[out[:, 4] >= a.conf]
        results.append({"file": p.name, **meta,
                        "dets": [[round(float(v), 4) for v in row[:5]] + [int(row[5])] for row in out]})
    n = max(len(results), 1)
    doc = {
        "dxnn": str(Path(a.dxnn).resolve()),
        "model_version": str(ie.get_model_version()) if hasattr(ie, "get_model_version") else None,
        "dx_engine": dxrt_version(),
        # 전처리 resize 결과가 OpenCV 버전마다 조금 다르다 (5.0.0 vs 5.1.0 에서 mAP50-95 0.25%p 차이 확인).
        # 호스트(캘리브레이션 letterbox)와 같은 버전을 쓰는지 dxconv 가 비교한다.
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "host": platform.node(),
        "input": {"name": tin["name"], "shape": shape, "dtype": "uint8", "layout": "NHWC", "color": "RGB",
                  "letterbox": "ultralytics(auto=False, center=True, INTER_LINEAR)", "pad": a.pad},
        "output": {"name": tout["name"], "shape": list(tout["shape"])},
        "conf": a.conf,
        "count": len(results),
        "missing": missing,
        "timing_ms": {"preprocess_mean": t_pre / n * 1000, "run_mean": t_inf / n * 1000,
                      "total_sec": time.perf_counter() - t_all},
        "results": results,
    }
    text = json.dumps(doc, ensure_ascii=False)
    if a.out == "-":
        print(text)
    else:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"{len(results)} images → {a.out} (run {doc['timing_ms']['run_mean']:.2f} ms/img)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
