"""원격 NPU 평가: dxnn 업로드 → 장비 추론(eval_map) + FPS(bench_fps) → 호스트에서 mAP.

sweep --remote 와 최종 평가에서 쓴다. 결과 원본 JSON 은 out_dir 에 보관한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dxconv import DxconvError
from dxconv.evaluate.npu import summarize_device
from dxconv.remote.client import RemoteClient, fps_value


def dataset_key(ds: Any) -> str:
    """장비 쪽 데이터 폴더 이름 (같은 이미지 폴더면 같은 키 → 재업로드 없음)."""
    src = str(Path(ds.cfg.images_dir).resolve())
    return f"{ds.name}_{hashlib.sha256(src.encode()).hexdigest()[:12]}"


def split_key(ds: Any, split: str) -> str:
    """PT 기준선 캐시 키: 이미지 폴더 + 분할 목록 내용 (같은 폴더라도 분할이 다르면 다시 계산)."""
    lst = getattr(ds.cfg, split, None)
    digest = hashlib.sha256(Path(lst).read_bytes()).hexdigest()[:12] if lst and Path(lst).is_file() else "-"
    return f"{dataset_key(ds)}:{split}:{digest}"


def check_format(client_info: dict[str, Any], dxnn_format: int | None) -> None:
    fmts = client_info.get("dxnn_formats") or []
    if dxnn_format and fmts and dxnn_format not in fmts:
        raise DxconvError(
            f"장비 {client_info.get('host')} 의 DX-RT {client_info.get('dxrt')} 는 dxnn v{dxnn_format} 을 "
            f"읽을 수 없습니다 (가능: {', '.join(f'v{f}' for f in fmts)}). "
            "장비 DX-RT 를 올리거나 다른 DX-COM 버전(compile.dxcom_version)으로 컴파일하세요.")


def evaluate_remote(
    client: RemoteClient, ds: Any, split: str, dxnn: Path, dxnn_sha256: str, out_dir: Path, *,
    num_classes: int, conf: float = 0.001, pr_conf: float = 0.25, max_det: int = 300,
    fps_seconds: float = 10.0, with_fps: bool = True,
) -> dict[str, Any]:
    """반환: {"summary": summarize 결과(+device), "fps": 선택 기준 FPS, "bench": bench_fps 원본}."""
    if ds.cfg.labels_dir is None:
        raise DxconvError("NPU mAP 평가에는 라벨 폴더가 필요합니다 (--labels).")
    images = ds.split_images(split)
    names = [p.relative_to(ds.cfg.images_dir).as_posix() for p in images]
    r_images, r_list = client.sync_images(dataset_key(ds), Path(ds.cfg.images_dir), names, split)
    r_dxnn = client.upload_dxnn(dxnn, dxnn_sha256)
    tag = dxnn_sha256[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = client.eval_map(r_dxnn, r_images, r_list, f"{tag}_{split}.json", conf=conf)
    (out_dir / f"npu_{split}_pred.json").write_text(json.dumps(doc), encoding="utf-8")
    summary = summarize_device(doc, Path(ds.cfg.labels_dir), num_classes, pr_conf=pr_conf,
                               max_det=max_det, expected=[Path(n).name for n in names])
    res: dict[str, Any] = {"summary": summary, "fps": None, "bench": None}
    if with_fps:
        bench = client.bench_fps(r_dxnn, f"{tag}_fps.json", seconds=fps_seconds)
        (out_dir / "npu_fps.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
        res["bench"], res["fps"] = bench, fps_value(bench)
    (out_dir / f"npu_{split}_summary.json").write_text(
        json.dumps({k: v for k, v in res.items() if k != "bench"}, indent=1, ensure_ascii=False,
                   default=str), encoding="utf-8")
    return res


def verify_dxnn(client: RemoteClient, ds: Any, dxnn: Path, cfg: Any, out_dir: Path, *,
                pt: Path | None = None, console: Any = None) -> dict[str, Any]:
    """추천 모드 검증: 만든 dxnn 을 평가 분할(없으면 튜닝)로 NPU 평가 + PT 기준선 비교."""
    from dxconv.evaluate.baseline import PtRunner, predict_split, summarize

    split = "evaluation" if ds.cfg.evaluation is not None else "tuning"
    sha = hashlib.sha256(dxnn.read_bytes()).hexdigest()
    max_det = cfg.evaluate.max_det or cfg.export.max_det
    num_classes = 0
    pt_sum = None
    if pt is not None:
        runner = PtRunner(pt, cfg.export.imgsz, cfg.export.end2end)
        num_classes = len(runner.net.names)
        if console is not None:
            console.print(f"  PT 기준선 ({split} {len(ds.split_images(split))}장, CPU)")
        dets, _, _ = predict_split(runner, ds.split_images(split), Path(ds.cfg.labels_dir),
                                   conf=cfg.evaluate.conf, max_det=max_det,
                                   pad_value=cfg.compile.preprocess.pad_value)
        pt_sum = summarize(dets, num_classes, cfg.postprocess.conf)
    if console is not None:
        console.print(f"  NPU 평가 ({client.name}, {split})")
    res = evaluate_remote(client, ds, split, dxnn, sha, out_dir, num_classes=num_classes or 80,
                          conf=cfg.evaluate.conf, pr_conf=cfg.postprocess.conf, max_det=max_det,
                          fps_seconds=cfg.evaluate.fps_seconds)
    s = res["summary"]
    out: dict[str, Any] = {
        "split": split, "images": s["images"], "device": client.check(),
        "npu": {"map50": s["coco"]["map50"], "map50_95": s["coco"]["map50_95"], "pr_at_conf": s["pr_at_conf"],
                "fps": res["fps"], "fps_sync": ((res["bench"] or {}).get("sync") or {}).get("fps")},
        "warnings": s.get("warnings"),
    }
    if pt_sum is not None:
        out["pt"] = {"map50": pt_sum["coco"]["map50"], "map50_95": pt_sum["coco"]["map50_95"],
                     "pr_at_conf": pt_sum["pr_at_conf"]}
        out["delta_map50_95_pp"] = (s["coco"]["map50_95"] - pt_sum["coco"]["map50_95"]) * 100
        out["delta_map50_pp"] = (s["coco"]["map50"] - pt_sum["coco"]["map50"]) * 100
    return out
