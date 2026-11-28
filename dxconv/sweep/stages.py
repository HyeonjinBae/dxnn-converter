"""설정 비교 실험 (sweep): 탐색 공간(configs/sweep/*.yaml)의 단계를 차례로 컴파일·평가해 선택.

기본 공간 (default_space.yaml): B 단계만
- compile.calibration_method × compile.calib_repeat (사진 묶음, 캘리브레이션 풀에서 추출)
  판정: 묶음을 뺀 설정별 Tuning mAP50-95 평균 최대 → 그 설정 중 Tuning 최대 묶음 (objective._select_calibration)
- opset·simplify·opt_level·aggressive 는 추천값 고정 (실측상 결과 동일 또는 항상 최고)

전체 공간 (full_space.yaml): A → B → C
- A: export.opset × export.simplify — 컴파일 성공 + dxnn 설정 일치 + FP32 ONNX 출력이 PT 와 같음 (로컬)
- B: compile.calibration_method × compile.calibration_num — Tuning mAP50-95 최대
- C: compile.opt_level × compile.aggressive_partitioning — 목적식 (objective.py)
이전 단계의 선택값을 고정하고 다음 단계로 간다. 평가가 없는 단계에서는 선택하지 않고 멈춘다 (pending).

데이터 분할 (자동 분할 또는 --dataset):
- 캘리브레이션: calibration_num 목록 (calib_repeat 0) 또는 풀에서 묶음 번호로 추출 (1 이상)
- 선택: Tuning 분할의 NPU 평가 (--remote, 또는 --compile-only 후 import-eval)
- 최종 보고: Evaluation 분할 (선택에 쓰지 않음)

조합마다 convert 를 실행하고 report.json 의 dxnn_settings(requested/actual)를 DB 에 기록한다.
설정 불일치·읽기 실패 조합은 excluded 로 표시하고 선택에서 제외한다. 같은 설정은 캐시(재컴파일 안 함).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from dxconv import DxconvError
from dxconv.config import ConfigError, RunCfg, deep_merge, find_config_dir, merge_layers
from dxconv.dataset import Dataset, fit_calibration_num
from dxconv.datasplit import resolve_dataset_input
from dxconv.report.writer import sha256_file
from dxconv.sweep.cache import cache_key
from dxconv.sweep.objective import parse_objective, select_stage
from dxconv.sweep.space import STAGE_ORDER, get_dotted, iter_combinations, load_space, to_nested
from dxconv.sweep.store import SweepStore, save_preset, write_pending_eval, write_summary


def sweep_root(out_root: Path, model: str, dxcom_version: str | None = None) -> Path:
    """기준 버전(2.5.0)은 sweep/<model>, 그 외는 sweep/<model>@dxcom<버전> (선택 결과가 섞이지 않게)."""
    from dxconv.config import REFERENCE_VERSION

    name = model if not dxcom_version or str(dxcom_version) == REFERENCE_VERSION else \
        f"{model}@dxcom{dxcom_version}"
    return Path(out_root) / "sweep" / name


def _resolve_space(space: Path, config_dir: Path) -> Path:
    if space.is_file():
        return space
    alt = config_dir / "sweep" / space.name
    if alt.is_file():
        return alt
    raise ConfigError(f"탐색 공간 파일이 없습니다: {space}")


def _calib_signature(cfg: RunCfg, calib_dir: Path | None, ds: Dataset | None) -> dict[str, Any]:
    if calib_dir is not None:
        return {"dir": str(calib_dir.resolve()), "seed": cfg.compile.calib_seed,
                "sampling": cfg.compile.calib_sampling}
    assert ds is not None
    num, rep = cfg.compile.calibration_num, cfg.compile.calib_repeat
    lst = ds.calibration_list(num)
    if rep == 0 or ds.calibration_sample(num, rep) == ds.calibration_images(num):
        # 풀이 작아 묶음 사진이 분할 목록과 같으면 같은 캐시 키 (같은 조합)
        return {"list_sha256": sha256_file(lst)}
    import hashlib

    names = "\n".join(str(p) for p in ds.calibration_sample(num, rep))
    return {"sample_sha256": hashlib.sha256(names.encode()).hexdigest()}


class _Ctx:
    """한 번의 sweep 실행에 필요한 공용 상태."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)
        self._pt_ref: dict[Path, Any] | None = None

    def pt_runner(self, cfg: RunCfg) -> Any:
        if getattr(self, "_pt_runner", None) is None:
            from dxconv.evaluate.baseline import PtRunner

            self._pt_runner = PtRunner(self.pt, cfg.export.imgsz, cfg.export.end2end)
        return self._pt_runner

    def pt_reference(self, cfg: RunCfg) -> dict[Path, Any]:
        if self._pt_ref is None:
            from dxconv.evaluate.baseline import pt_reference_outputs

            self._pt_ref = pt_reference_outputs(self.pt_runner(cfg), self.check_images,
                                                pad_value=cfg.compile.preprocess.pad_value)
        return self._pt_ref

    def num_classes(self, cfg: RunCfg) -> int:
        return len(self.pt_runner(cfg).net.names)


def _latest_run(root: Path) -> Path | None:
    runs = sorted((root / "convert").glob("*")) if (root / "convert").is_dir() else []
    return runs[-1] if runs else None


def _run_combo(ctx: _Ctx, key: str, cfg: RunCfg) -> None:
    from dxconv.pipeline import RunOptions, execute

    store: SweepStore = ctx.store
    store.upsert(key, config=cfg.model_dump(mode="json"), status="running", error=None)
    combo_root = store.root / "combos" / key
    try:
        result = execute("convert", cfg, ctx.config_dir,
                         RunOptions(out_root=combo_root, timeout=ctx.timeout),
                         pt=ctx.pt, calib_dir=ctx.calib_dir, dataset=ctx.ds, console=ctx.console)
        run_dir = result.run_dir
    except DxconvError as e:
        run_dir = _latest_run(combo_root)
        store.upsert(key, status="failed", error=str(e).splitlines()[0][:500],
                     run_dir=str(run_dir) if run_dir else None)
        ctx.console.print(f"  [red]실패[/] {escape(str(e).splitlines()[0])}")
        return
    rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    settings = rep.get("dxnn_settings") or {}
    mismatches = settings.get("mismatches") or []
    readable = (settings.get("actual") or {}).get("calibration_method") is not None
    ok = readable and not mismatches
    dxnn = Path(rep["outputs"]["dxnn"])
    summ = rep.get("compile_summary") or {}
    store.upsert(
        key, status="compiled" if ok else "excluded", run_dir=str(run_dir),
        dxnn_path=str(dxnn), dxnn_sha256=sha256_file(dxnn), dxnn_size=dxnn.stat().st_size,
        compile_sec=(rep.get("timings_sec") or {}).get("compile"),
        npu_groups=summ.get("npu_partitions"), cpu_groups=summ.get("cpu_partitions"),
        settings=settings, settings_ok=ok, onnx_exporter=rep.get("onnx_exporter"),
        warnings=rep.get("warnings"),
        error=None if ok else ("dxnn 설정 불일치: " + "; ".join(mismatches) if mismatches
                               else "dxnn 에 기록된 설정을 읽지 못함: " + "; ".join(
                                   w for w in rep.get("warnings") or [] if "dxnn" in w)[:400]),
    )
    if not ok:
        ctx.console.print(f"  [yellow]제외 (설정 검증 실패)[/] {escape('; '.join(mismatches))}")


def _fp32_check(ctx: _Ctx, key: str, cfg: RunCfg) -> None:
    from dxconv.evaluate.baseline import OnnxRunner, check_onnx_vs_pt

    row = ctx.store.get(key)
    if not row or row["status"] != "compiled" or row.get("fp32_ok") is not None:
        return
    onnx = Path(row["run_dir"]) / f"{cfg.model}.onnx"
    res = check_onnx_vs_pt(ctx.pt_reference(cfg), OnnxRunner(onnx),
                           pad_value=cfg.compile.preprocess.pad_value)
    ctx.store.upsert(key, fp32=res, fp32_ok=res["ok"])
    ctx.console.print(f"  FP32 ONNX = PT: {'예' if res['ok'] else '아니오 — ' + res['reason']}")


def _pt_baseline(ctx: _Ctx, cfg: RunCfg, split: str) -> dict[str, Any]:
    """PT 기준선 mAP (호스트 CPU). sweep 데이터셋의 분할로 계산해 meta 에 둔다."""
    from dxconv.evaluate.baseline import predict_split, summarize
    from dxconv.evaluate.remote_eval import split_key

    key = f"pt_{split}"
    cached = ctx.store.get_meta(key)
    dkey = split_key(ctx.ds, split)
    if cached and cached.get("dataset_key") == dkey and cached.get("pt_sha256") == ctx.pt_sha:
        return cached
    images = ctx.ds.split_images(split)
    ctx.console.print(f"  PT 기준선 계산 ({split} {len(images)}장, CPU)")
    dets, _, _ = predict_split(ctx.pt_runner(cfg), images, Path(ctx.ds.cfg.labels_dir),
                               conf=cfg.evaluate.conf, max_det=cfg.evaluate.max_det or cfg.export.max_det,
                               pad_value=cfg.compile.preprocess.pad_value)
    s = summarize(dets, ctx.num_classes(cfg), cfg.postprocess.conf)
    res = {"map50": s["coco"]["map50"], "map50_95": s["coco"]["map50_95"], "images": len(images),
           "pr_at_conf": s["pr_at_conf"], "dataset_key": dkey, "pt_sha256": ctx.pt_sha,
           "source": f"dxconv sweep (PT, {split} {len(images)}장)"}
    ctx.store.set_meta(key, res)
    return res


def _remote_eval_stage(ctx: _Ctx, st: str, cfg_for: dict[str, RunCfg]) -> None:
    """이 단계의 유효 조합 중 평가가 없는 것을 원격 NPU 에서 평가 (Tuning mAP + FPS)."""
    from datetime import datetime

    from dxconv.evaluate.remote_eval import evaluate_remote

    store: SweepStore = ctx.store
    pt = store.get_meta("pt_baseline") or {}
    for r in store.stage_rows(st):
        if r["status"] != "compiled" or (r.get("tuning_map50_95") is not None and r.get("fps") is not None):
            continue
        cfg = cfg_for[r["hash"]]
        ctx.console.print(f"  NPU 평가 {r['hash']} {escape(json.dumps(r['params'], ensure_ascii=False))}")
        res = evaluate_remote(
            ctx.client, ctx.ds, "tuning", Path(r["dxnn_path"]), r["dxnn_sha256"],
            Path(r["run_dir"]) / "npu_eval", num_classes=ctx.num_classes(cfg), conf=cfg.evaluate.conf,
            pr_conf=cfg.postprocess.conf, max_det=cfg.evaluate.max_det or cfg.export.max_det,
            fps_seconds=cfg.evaluate.fps_seconds)
        s = res["summary"]
        fields: dict[str, Any] = {
            "tuning_map50": s["coco"]["map50"], "tuning_map50_95": s["coco"]["map50_95"], "fps": res["fps"],
            "eval_source": f"remote:{ctx.client.name}", "eval_at": datetime.now().isoformat(timespec="seconds"),
            "npu": {"device": ctx.device, "fps_sync": ((res["bench"] or {}).get("sync") or {}).get("fps"),
                    "fps_max": ((res["bench"] or {}).get("run_model") or {}).get("fps"),
                    "pr_at_conf": s["pr_at_conf"], "per_class_ap50_95": s["coco"].get("per_class"),
                    "opencv": s["device"].get("opencv"), "warnings": s.get("warnings")}}
        if pt.get("map50_95") is not None:
            fields["delta_pp"] = (s["coco"]["map50_95"] - pt["map50_95"]) * 100
        store.upsert(r["hash"], **fields)
        for w in s.get("warnings") or []:
            ctx.console.print(f"  [yellow]경고:[/] {escape(w)}")
        ctx.console.print(f"    mAP50-95 {s['coco']['map50_95'] * 100:.2f}  mAP50 {s['coco']['map50'] * 100:.2f}"
                          f"  FPS {res['fps'] or 0:.1f}"
                          + (f"  Δ vs PT {fields['delta_pp']:+.2f}%p" if "delta_pp" in fields else ""))


def _final_eval(ctx: _Ctx, sp: dict[str, Any], base_raw: dict[str, Any]) -> None:
    """모든 단계 선택 후: 최종 조합을 평가(evaluation) 분할로 NPU 평가 + PT 기준선 (보고용)."""
    from dxconv.evaluate.remote_eval import evaluate_remote

    store: SweepStore = ctx.store
    best = selected_params(store, sp)
    if best is None or ctx.ds is None or ctx.ds.cfg.evaluation is None:
        return
    last = [s for s in STAGE_ORDER if s in sp][-1]
    sel = store.get_selection(last)
    row = store.get(sel["hash"])
    prev = store.get_meta("final_eval") or {}
    if prev.get("hash") == row["hash"] and prev.get("dxnn_sha256") == row["dxnn_sha256"]:
        return
    cfg = RunCfg.model_validate(deep_merge(base_raw, to_nested(best)))
    ctx.console.print(f"[bold]▶ 최종 평가[/] {row['hash']} (evaluation 분할, 선택에는 쓰지 않음)")
    pt = _pt_baseline(ctx, cfg, "evaluation")
    res = evaluate_remote(ctx.client, ctx.ds, "evaluation", Path(row["dxnn_path"]), row["dxnn_sha256"],
                          Path(row["run_dir"]) / "npu_eval", num_classes=ctx.num_classes(cfg),
                          conf=cfg.evaluate.conf, pr_conf=cfg.postprocess.conf,
                          max_det=cfg.evaluate.max_det or cfg.export.max_det, with_fps=False)
    s = res["summary"]
    final = {"hash": row["hash"], "dxnn_sha256": row["dxnn_sha256"], "dxnn_path": row["dxnn_path"],
             "params": best, "images": s["images"], "map50": s["coco"]["map50"],
             "map50_95": s["coco"]["map50_95"], "pr_at_conf": s["pr_at_conf"], "fps": row.get("fps"),
             "pt": {"map50": pt["map50"], "map50_95": pt["map50_95"], "pr_at_conf": pt["pr_at_conf"]},
             "delta_map50_95_pp": (s["coco"]["map50_95"] - pt["map50_95"]) * 100,
             "delta_map50_pp": (s["coco"]["map50"] - pt["map50"]) * 100, "device": ctx.device}
    store.set_meta("final_eval", final)
    ctx.console.print(f"  [green]최종[/] mAP50-95 {final['map50_95'] * 100:.2f} (PT {pt['map50_95'] * 100:.2f}, "
                      f"Δ {final['delta_map50_95_pp']:+.2f}%p)  mAP50 {final['map50'] * 100:.2f} "
                      f"(Δ {final['delta_map50_pp']:+.2f}%p)  FPS {final['fps'] or 0:.1f}")


def _fixed_for(stage: str, store: SweepStore, base: dict[str, Any], space: dict[str, Any],
               console: Console) -> tuple[dict[str, Any], dict[str, str]]:
    """이전 단계 선택값. 선택이 없으면 기본 설정 값을 쓰고 출처를 'base' 로 표시."""
    fixed: dict[str, Any] = {}
    source: dict[str, str] = {}
    for prev in STAGE_ORDER[:STAGE_ORDER.index(stage)]:
        if prev not in space:
            continue
        sel = store.get_selection(prev)
        if sel and sel["status"] == "selected":
            row = next(r for r in store.stage_rows(prev) if r["hash"] == sel["hash"])
            fixed.update(row["params"])
            source[prev] = f"selected:{sel['hash']}"
        else:
            vals = {k: get_dotted(base, k) for k in space[prev]}
            fixed.update(vals)
            source[prev] = "base"
            console.print(f"[yellow]경고:[/] {prev} 단계가 선택되지 않아 기본 설정 값 "
                          f"{escape(json.dumps(vals, ensure_ascii=False))} 으로 고정합니다.")
    return fixed, source


def _select(store: SweepStore, stage: str, objective: Any, base: dict[str, Any],
            space: dict[str, Any]) -> Any:
    rows = store.stage_rows(stage)
    base_params = {k: get_dotted(base, k) for k in space[stage]}
    pt = store.get_meta("pt_baseline") or {}
    sel = select_stage(stage, rows, objective, base_params, pt)
    store.set_selection(stage, sel.status, sel.hash, sel.reason)
    return sel


def run_sweep(
    *,
    pt: Path,
    model: str,
    calib_dir: Path | None,
    space: Path,
    objective: str | None,
    remote: str | None,
    save_preset: bool,
    dataset: str | None = None,
    images: Path | None = None,
    labels: Path | None = None,
    target_dxrt: str | None = None,
    strict_objective: bool = False,
    stage: str | None = None,
    compile_only: bool = False,
    out_root: Path = Path("runs"),
    config: Path | None = None,
    set_items: list[str] | None = None,
    retry_failed: bool = False,
    check_images: int = 20,
    timeout: float | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    """sweep 실행. 반환: {stage: selection status, ..., "root": 결과 폴더}"""
    console = console or Console(stderr=True)
    if remote and compile_only:
        console.print("[yellow]안내:[/] --compile-only 가 있어 --remote 평가는 하지 않습니다.")
        remote = None
    if stage is not None and stage not in STAGE_ORDER:
        raise DxconvError(f"알 수 없는 --stage: {stage} (A | B | C)")
    if not pt.is_file():
        raise DxconvError(f"pt 파일이 없습니다: {pt}")

    config_dir = find_config_dir()
    set_items = list(set_items or [])
    client, device = None, None
    if remote:
        from dxconv.remote.client import RemoteClient, load_host

        client = RemoteClient(remote, load_host(remote, config_dir))
        console.print(f"[bold]원격 장비[/] {remote} 점검·동기화")
        client.setup()
        device = client.check()
        if not device["ok"]:
            raise DxconvError("원격 장비 점검 실패: " + "; ".join(device["problems"]))
        console.print(f"  DX-RT {device['dxrt']} / 드라이버 {device['rt_driver']} / FW {device['firmware']} "
                      f"/ dxnn {', '.join(f'v{f}' for f in device['dxnn_formats'])}")
    explicit_version = any(i.startswith("compile.dxcom_version=") for i in set_items)
    dxrt = target_dxrt or (device or {}).get("dxrt")
    if dxrt and not explicit_version:
        from dxconv.env.compat import load_compat, version_for_dxrt

        ver, fmt = version_for_dxrt(dxrt, load_compat(config_dir))
        set_items.append(f"compile.dxcom_version={ver}")
        console.print(f"[cyan]대상 DX-RT {dxrt} → dxnn v{fmt} (DX-COM {ver}) 로 만듭니다.[/]")
    base_raw = merge_layers(model, user_config=config, set_items=set_items,
                            config_dir=config_dir)
    try:
        base_cfg = RunCfg.model_validate(base_raw)
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"기본 설정 검증 실패: {e}") from e
    base = base_cfg.model_dump(mode="json")

    ds = None
    if calib_dir is not None and (dataset is not None or images is not None):
        raise DxconvError("--calib-dir 는 --dataset / --images 와 함께 쓸 수 없습니다.")
    if calib_dir is None:
        ds = resolve_dataset_input(model, config_dir, dataset=dataset, images=images, labels=labels,
                                   split_cfg=base_cfg.split, out_root=out_root, console=console)
    space_def = load_space(_resolve_space(space, config_dir))
    sp = space_def["stages"]
    if ds is not None and ds.cfg.is_auto:
        # 자동 분할: calibration_num 후보는 이미지 수에 따라 정해진다 → 기본값·B 탐색값을 맞춘다
        cands = sorted(ds.cfg.calibration)
        num = base_raw.get("compile", {}).get("calibration_num", base_cfg.compile.calibration_num)
        fitted = fit_calibration_num(int(num), ds)
        if fitted != num:
            console.print(f"[yellow]안내:[/] 자동 분할 후보 {cands} 에 맞춰 기본 calibration_num "
                          f"{num} → {fitted}")
            base_raw = deep_merge(base_raw, {"compile": {"calibration_num": fitted}})
            base_cfg = RunCfg.model_validate(base_raw)
            base = base_cfg.model_dump(mode="json")
        for st_def in sp.values():
            if "compile.calibration_num" in st_def:
                # 탐색값을 분할 후보에 맞춘다 (예: 300장이면 100 → 60)
                st_def["compile.calibration_num"] = sorted({fit_calibration_num(int(v), ds)
                                                            for v in st_def["compile.calibration_num"]})
    obj = parse_objective(objective or space_def.get("objective"), strict=strict_objective)

    if ds is not None and ds.cfg.tuning is not None:
        imgs = ds.split_images("tuning")[:check_images]
    elif ds is not None:
        imgs = ds.calibration_images(min(ds.cfg.calibration))[:check_images]
    else:
        from dxconv.compile.calib import scan_images
        imgs = scan_images(calib_dir)[0][:check_images]  # type: ignore[arg-type]

    if client is not None:
        from dxconv.env.compat import find_compat_entry, load_compat
        from dxconv.evaluate.remote_eval import check_format

        if ds is None or ds.cfg.labels_dir is None or ds.cfg.tuning is None:
            raise DxconvError("--remote 평가에는 라벨이 있는 데이터셋(튜닝 분할)이 필요합니다 "
                              "(--images + --labels 또는 --dataset).")
        entry = find_compat_entry(str(base_cfg.compile.dxcom_version), load_compat(config_dir)) or {}
        check_format(device, entry.get("dxnn_format"))
    store = SweepStore(sweep_root(out_root, model, base_cfg.compile.dxcom_version))
    store.set_meta("objective", obj.text)
    store.set_meta("strict_objective", obj.strict)
    if device:
        store.set_meta("device", device)
    pt_sha = sha256_file(pt)
    ctx = _Ctx(store=store, config_dir=config_dir, pt=pt, calib_dir=calib_dir, ds=ds,
               timeout=timeout, console=console, check_images=imgs, client=client, device=device,
               pt_sha=pt_sha)
    stages = [stage] if stage else [s for s in STAGE_ORDER if s in sp]
    outcome: dict[str, Any] = {"root": str(store.root)}
    try:
        for st in stages:
            if st not in sp:
                raise DxconvError(f"탐색 공간에 {st} 단계가 없습니다.")
            fixed, source = _fixed_for(st, store, base, sp, console)
            combos = iter_combinations(sp[st])
            console.print(f"[bold]▶ {st} 단계[/] {len(combos)}개 조합 "
                          f"(고정: {escape(json.dumps(fixed, ensure_ascii=False))})")
            store.clear_members(st)
            cfg_for: dict[str, RunCfg] = {}
            for i, params in enumerate(combos, 1):
                raw = deep_merge(base_raw, to_nested({**fixed, **params}))
                cfg = RunCfg.model_validate(raw)
                calib_error = None
                try:
                    calib_sig = _calib_signature(cfg, calib_dir, ds)
                except DxconvError as e:  # 예: 이 calibration_num 의 목록이 데이터셋에 없음
                    calib_error = str(e)
                    calib_sig = {"error": calib_error}
                comp = cfg.compile.model_dump(mode="json")
                comp.pop("calib_repeat", None)  # 사진 목록은 calib 서명에 들어 있음
                if str(cfg.compile.dxcom_version).startswith("2.4"):
                    comp["runner"] = "python_api"  # 2.4: CLI 는 캘리브레이션 설정을 무시
                key = cache_key({"export": cfg.export.model_dump(mode="json"),
                                 "compile": comp,
                                 "pt": pt_sha, "calib": calib_sig})
                if key in cfg_for:  # 풀이 작아 묶음끼리 사진이 같음 → 같은 조합
                    console.print(f"  [{i}/{len(combos)}] {escape(json.dumps(params, ensure_ascii=False))} "
                                  f"→ 앞 조합과 같음 ({key})")
                    continue
                store.add_member(st, key, params, {**fixed, "_source": source}, i)
                cfg_for[key] = cfg
                row = store.get(key)
                label = escape(json.dumps(params, ensure_ascii=False))
                cached = row and (row["status"] in ("compiled", "excluded")
                                  or (row["status"] == "failed" and not retry_failed))
                if calib_error is not None:
                    store.upsert(key, config=cfg.model_dump(mode="json"), status="failed",
                                 error=calib_error)
                    console.print(f"  [{i}/{len(combos)}] {label} → [red]실패[/] "
                                  f"{escape(calib_error)}")
                elif cached:
                    console.print(f"  [{i}/{len(combos)}] {label} → 캐시 ({row['status']}, {key})")
                else:
                    console.print(f"  [{i}/{len(combos)}] {label} → 컴파일 ({key})")
                    _run_combo(ctx, key, cfg)
                if st == "A":
                    _fp32_check(ctx, key, cfg)
            if client is not None and st != "A":
                from dxconv.evaluate.remote_eval import split_key

                cur = store.get_meta("pt_baseline") or {}
                if cur.get("dataset_key") != split_key(ds, "tuning") or cur.get("pt_sha256") != pt_sha:
                    ptb = _pt_baseline(ctx, base_cfg, "tuning")
                    store.set_meta("pt_baseline", {k: ptb[k] for k in
                                                   ("map50", "map50_95", "source", "dataset_key", "pt_sha256")})
                _remote_eval_stage(ctx, st, cfg_for)
            sel = _select(store, st, obj, base, sp)
            outcome[st] = sel.status
            if sel.status == "pending":
                path = write_pending_eval(store, st, sel.missing or [])
                console.print(f"[yellow]{st} 단계 pending:[/] {escape(sel.reason)}. "
                              + ("NPU 평가는 아직 연결되지 않았습니다 (--compile-only). "
                                 if compile_only else "")
                              + f"평가 결과를 {escape(str(path))} 형식으로 채워 "
                              "`dxconv sweep import-eval` 로 넣으세요.")
                break
            if sel.status != "selected":
                console.print(f"[red]{st} 단계 선택 불가:[/] {escape(sel.reason)}")
                break
            if sel.fallback:
                console.print(f"[yellow]{st} 단계 대체 선택:[/] {sel.hash} ({escape(sel.reason)})")
            else:
                console.print(f"[green]{st} 단계 선택:[/] {sel.hash} ({escape(sel.reason)})")
        if client is not None and stage is None:
            _final_eval(ctx, sp, base_raw)
    finally:
        write_summary(store, model, STAGE_ORDER)
        console.print(f"[dim]요약: {store.root / 'summary.md'}[/]")
    if save_preset:
        best = selected_params(store, sp)
        if best is None:
            console.print("[yellow]--save-preset:[/] 모든 단계가 선택되지 않아 프리셋을 저장하지 않았습니다.")
        else:
            path = save_preset_file(model, best, config_dir, store, base_cfg.compile.dxcom_version)
            console.print(f"[green]프리셋 저장:[/] {path}")
    store.close()
    return outcome


def selected_params(store: SweepStore, space: dict[str, Any]) -> dict[str, Any] | None:
    best: dict[str, Any] = {}
    for st in (s for s in STAGE_ORDER if s in space):
        sel = store.get_selection(st)
        if not sel or sel["status"] != "selected":
            return None
        best.update(next(r for r in store.stage_rows(st) if r["hash"] == sel["hash"])["params"])
    return best


def save_preset_file(model: str, best: dict[str, Any], config_dir: Path,
                     store: SweepStore, dxcom_version: str | None = None) -> Path:
    from dxconv.config import preset_overlay_path

    overlay = preset_overlay_path(config_dir, model, dxcom_version)
    if overlay is not None:  # 기준이 아닌 버전: presets/<model>@dxcom<버전>.yaml
        best = {**best, "compile.dxcom_version": str(dxcom_version)}
    return save_preset(model, best, config_dir, note=f"sweep DB: {store.db_path}",
                       path=overlay)


# --------------------------------------------------------------------------- #
# 평가 결과 수동 입력
# --------------------------------------------------------------------------- #
def import_eval(
    *, model: str, json_path: Path, out_root: Path = Path("runs"),
    baseline_report: Path | None = None, space: Path | None = None,
    console: Console | None = None, dxcom_version: str | None = None,
) -> dict[str, Any]:
    """NPU 평가 결과(JSON)를 sweep DB 에 넣고 단계 선택을 다시 계산한다.

    JSON: {"split": "tuning", "pt": {"map50": .., "map50_95": ..},
           "results": [{"hash" | "dxnn_sha256": .., "map50": .., "map50_95": .., "fps": ..}]}
    값은 0~1 비율 (mAP). pending_eval.json 이 템플릿이다.
    """
    console = console or Console(stderr=True)
    root = sweep_root(out_root, model, dxcom_version)
    if not (root / "sweep.db").is_file():
        raise DxconvError(f"sweep DB 가 없습니다: {root / 'sweep.db'} (먼저 dxconv sweep 실행)")
    try:
        doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DxconvError(f"평가 JSON 을 읽을 수 없습니다: {json_path} ({e})") from e
    split = doc.get("split", "tuning")
    if split != "tuning":
        raise DxconvError(f"sweep 선택에는 tuning 분할 결과만 넣습니다 (입력: {split}).")
    store = SweepStore(root)
    pt = dict(store.get_meta("pt_baseline") or {})
    if baseline_report is not None:
        rep = json.loads(Path(baseline_report).read_text(encoding="utf-8"))
        try:
            coco = rep["baseline"]["tuning"]["pt"]["coco"]
        except KeyError as e:
            raise DxconvError(f"baseline report 에 tuning PT 결과가 없습니다: {baseline_report}") from e
        pt = {"map50": coco["map50"], "map50_95": coco["map50_95"], "source": str(baseline_report)}
    elif any((doc.get("pt") or {}).get(k) is not None for k in ("map50", "map50_95")):
        pt = {**doc["pt"], "source": str(json_path)}
    if pt:
        store.set_meta("pt_baseline", pt)

    updated, problems = [], []
    for item in doc.get("results", []):
        key = item.get("hash") or item.get("dxnn_sha256")
        if not key:
            problems.append(f"hash/dxnn_sha256 없음: {item}")
            continue
        if item.get("hash") and len(item["hash"]) < 8:
            problems.append(f"hash 는 8자 이상: {item['hash']}")
            continue
        found = store.find(key)
        if len(found) != 1:
            problems.append(f"{key}: 일치하는 조합 {len(found)}개")
            continue
        if all(item.get(k) is None for k in ("map50", "map50_95", "fps")):
            continue
        h = found[0]["hash"]
        fields = {"eval_source": str(json_path), "eval_at": doc.get("evaluated_at")}
        for src, dst in (("map50", "tuning_map50"), ("map50_95", "tuning_map50_95"), ("fps", "fps")):
            if item.get(src) is not None:
                fields[dst] = float(item[src])
                old = found[0].get(dst)
                if old is not None and abs(old - fields[dst]) > 1e-9:
                    # 같은 dxnn 은 단계가 달라도 한 조합(캐시 단위)이라 값이 하나다
                    problems.append(f"{h}: {dst} 기존 값 {old:g} → {fields[dst]:g} 로 덮어씀 "
                                    "(다른 단계 선택에도 영향)")
        store.upsert(h, **fields)
        updated.append(h)
    # Δ vs PT 다시 계산
    if pt.get("map50_95") is not None:
        for r in store.conn.execute("SELECT hash, tuning_map50_95 FROM combos").fetchall():
            if r["tuning_map50_95"] is not None:
                store.upsert(r["hash"], delta_pp=(r["tuning_map50_95"] - pt["map50_95"]) * 100)

    # 선택 다시 계산 (이전 단계 선택이 바뀌면 이후 단계는 stale)
    config_dir = find_config_dir()
    sp = load_space(_resolve_space(space or Path("configs/sweep/default_space.yaml"),
                                   config_dir))["stages"]
    vset = [f"compile.dxcom_version={dxcom_version}"] if dxcom_version else []
    base = RunCfg.model_validate(merge_layers(model, set_items=vset, config_dir=config_dir)).model_dump(mode="json")
    obj = parse_objective(store.get_meta("objective"), strict=bool(store.get_meta("strict_objective")))
    current: dict[str, Any] = {}
    selections: dict[str, str] = {}
    for st in (s for s in STAGE_ORDER if s in sp):
        rows = store.stage_rows(st)
        if not rows:
            continue
        fixed = {k: v for k, v in rows[0]["fixed"].items() if k != "_source"}
        if any(current.get(k, v) != v for k, v in fixed.items()):
            store.set_selection(st, "stale", None, "이전 단계 선택이 바뀜 → 이 단계를 다시 실행하세요")
            selections[st] = "stale"
            break
        sel = _select(store, st, obj, base, sp)
        selections[st] = sel.status
        if sel.status != "selected":
            break
        current.update(next(r for r in rows if r["hash"] == sel.hash)["params"])
    write_summary(store, model, STAGE_ORDER)
    store.close()
    for p in problems:
        console.print(f"[yellow]경고:[/] {escape(p)}")
    console.print(f"평가 결과 {len(updated)}개 반영. 선택: {selections}")
    return {"updated": updated, "problems": problems, "selections": selections, "pt": pt}


def sweep_status(model: str, out_root: Path = Path("runs"), dxcom_version: str | None = None) -> Path:
    root = sweep_root(out_root, model, dxcom_version)
    if not (root / "sweep.db").is_file():
        raise DxconvError(f"sweep DB 가 없습니다: {root / 'sweep.db'}")
    store = SweepStore(root)
    path = write_summary(store, model, STAGE_ORDER)
    store.close()
    return path
