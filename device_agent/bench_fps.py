#!/usr/bin/env python3
"""NPU 장비에서 DXNN FPS 측정.

두 가지를 잰다:
- sync   : 한 장씩 동기 추론 (run) 반복 → 단일 스트림 지연/처리량
- bench  : DX-RT run_model -b (최대 처리량, NPU 코어 모두 사용) 출력 파싱
입력은 0 으로 채운 텐서 (FPS 는 입력 내용과 무관). 후처리(CPU 그래프)는 dxnn 안에 포함돼 측정된다.

사용 예:
    python bench_fps.py --dxnn fire.dxnn --seconds 10 --out fps.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def bench_sync(path: str, seconds: float, warmup: int) -> dict:
    from dx_engine import InferenceEngine

    ie = InferenceEngine(path)
    tin = ie.get_input_tensors_info()[0]
    x = np.zeros(tin["shape"], dtype=np.dtype(tin["dtype"]))
    for _ in range(warmup):
        ie.run([x])
    lat = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        ie.run([x])
        lat.append(time.perf_counter() - t0)
    a = np.array(lat) * 1000
    return {"fps": len(a) / (a.sum() / 1000), "latency_ms_mean": float(a.mean()),
            "latency_ms_p50": float(np.percentile(a, 50)), "latency_ms_p95": float(np.percentile(a, 95)),
            "runs": len(a)}


def bench_run_model(path: str, seconds: float) -> dict | None:
    exe = shutil.which("run_model")  # DX-RT 실행 파일. PATH 에 없으면 remote.yaml 의 dxrt_bin 으로 추가
    if exe is None:
        return None
    p = subprocess.run([exe, "-m", path, "-b", "-t", str(int(max(seconds, 1)))],
                       capture_output=True, text=True, timeout=seconds + 120)
    text = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + p.stderr)
    fps = re.findall(r"FPS\s*[:=]\s*([\d.]+)", text, re.I)
    return {"returncode": p.returncode, "fps": float(fps[-1]) if fps else None,
            "tail": text.strip().splitlines()[-15:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dxnn", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-run-model", action="store_true")
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    doc = {"dxnn": str(Path(a.dxnn).resolve()), "seconds": a.seconds,
           "sync": bench_sync(a.dxnn, a.seconds, a.warmup)}
    if not a.no_run_model:
        doc["run_model"] = bench_run_model(a.dxnn, a.seconds)
    text = json.dumps(doc, ensure_ascii=False, indent=1)
    if a.out == "-":
        print(text)
    else:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"sync {doc['sync']['fps']:.1f} FPS → {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
