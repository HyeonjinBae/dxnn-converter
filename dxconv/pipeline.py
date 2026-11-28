"""export / compile / convert 실행 흐름.

CLI(cli.py)는 인수 해석만 하고, 실제 단계 실행·report 기록은 여기서 한다.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from dxconv import DxconvError
from dxconv.compile.adapters import get_adapter
from dxconv.compile.calib import (
    LetterboxSpec,
    letterbox_impl_info,
    prepare_calib,
    prepare_calib_from_list,
)
from dxconv.compile.compiler import run_compile
from dxconv.compile.config_builder import (
    build_dxcom_config,
    input_hw,
    onnx_info_from_cfg,
    preprocess_steps,
    write_config,
)
from dxconv.compile.onnx_check import OnnxInfo, check_onnx
from dxconv.config import RunCfg, save_resolved
from dxconv.dataset import Dataset
from dxconv.env.compat import check_compat, load_compat, same_version
from dxconv.env.detect import EnvInfo, detect_env
from dxconv.export.exporter import export_onnx
from dxconv.export.inspect_pt import inspect_pt
from dxconv.export.rules import check_export_rules
from dxconv.report.writer import Report, create_run_dir, sha256_file, write_json

log = logging.getLogger(__name__)


@dataclass
class RunOptions:
    out_root: Path = Path("runs")
    dry_run: bool = False
    dxcom_version: str | None = None  # --dxcom-version (CLI 명시값)
    timeout: float | None = None
    use_api: bool = False


@dataclass
class RunResult:
    run_dir: Path
    report: Report
    onnx: Path | None = None
    dxnn: Path | None = None
    dxcom_config: Path | None = None


def check_end2end_output(info: OnnxInfo, cfg: RunCfg) -> list[str]:
    """end2end 로 요청했는데 출력이 [1, max_det, N] 형태가 아니면 경고 (raw head 로 빠진 경우)."""
    if not cfg.export.end2end or not info.outputs:
        return []
    shape = info.outputs[0].shape
    if len(shape) == 3 and shape[1] == cfg.export.max_det:
        return []
    return [
        f"end2end=True 이지만 ONNX 출력 shape {shape} 이 [1, {cfg.export.max_det}, N] 형태가 "
        "아닙니다. end2end head 가 적용되지 않았을 수 있습니다."
    ]


# --------------------------------------------------------------------------- #
# 단계
# --------------------------------------------------------------------------- #
def export_stage(
    cfg: RunCfg, pt: Path, run_dir: Path, report: Report, config_dir: Path, dry_run: bool
) -> Path | None:
    compat = load_compat(config_dir)
    if not pt.is_file():
        raise DxconvError(f"pt 파일이 없습니다: {pt}")
    with report.stage("hash_pt"):
        report.section("inputs")["pt"] = {"path": str(pt.resolve()), "sha256": sha256_file(pt)}

    # 1) 설정만으로 검증 (빠른 실패)
    check_export_rules(cfg.export, cfg.postprocess, compat).raise_if_error()

    # 2) pt 내용 검사 후 재검증
    pt_info = None
    with report.stage("inspect_pt"):
        try:
            pt_info = inspect_pt(pt)
            report.set("pt_info", pt_info.to_dict())
        except DxconvError as e:
            if not dry_run:
                raise
            report.warn(f"[dry-run] pt 검사를 건너뜁니다: {e}")
    rules = check_export_rules(cfg.export, cfg.postprocess, compat, pt_info)
    report.warn(*rules.warnings)
    rules.raise_if_error()

    if dry_run:
        return None

    dest = run_dir / f"{cfg.model}.onnx"
    with report.stage("export"):
        export_onnx(pt, cfg.export, dest)
    with report.stage("onnx_check"):
        info, warns = check_onnx(dest, compat.get("opset_range"), expected_opset=cfg.export.opset)
    report.warn(*warns)
    report.warn(*check_end2end_output(info, cfg))
    report.set("onnx_info", info.to_dict())
    # ONNX 를 만든 ultralytics 버전 (ONNX 메타데이터). 환경 버전은 versions.ultralytics
    report.set("onnx_exporter", onnx_producer_info(dest))
    report.section("outputs")["onnx"] = str(dest)
    return dest


def compile_stage(
    cfg: RunCfg,
    onnx: Path | None,
    calib_dir: Path | None,
    run_dir: Path,
    report: Report,
    config_dir: Path,
    env: EnvInfo,
    opts: RunOptions,
    dataset: Dataset | None = None,
) -> tuple[Path, Path | None]:
    """config.json 생성 (+ dry-run 이 아니면 컴파일). (config_path, dxnn_path) 반환."""
    compat = load_compat(config_dir)
    cres = check_compat(opts.dxcom_version, env, compat, cfg.compile.dxcom_version)
    report.warn(*cres.warnings)
    report.section("versions")["dxcom"] = {"selected": cres.version, "source": cres.source,
                                           "compat": cres.compat_entry}
    adapter = get_adapter(cres.version)

    if onnx is None:
        info = onnx_info_from_cfg(cfg)
        report.warn("[dry-run] ONNX 가 없어 설정값(export.imgsz/batch)으로 입력 shape 을 추정했습니다.")
    else:
        with report.stage("onnx_check"):
            info, warns = check_onnx(onnx, compat.get("opset_range"))
        report.warn(*warns)
        report.set("onnx_info", info.to_dict())

    pre = cfg.compile.preprocess
    h, w = input_hw(info.primary_input, pre.layout)
    lb_spec = LetterboxSpec(h, w, pre.pad_value) if pre.letterbox_mode == "host" else None
    with report.stage("calib"):
        if calib_dir is not None:
            subset, cinfo, samples = prepare_calib(
                calib_dir, cfg.compile.calibration_num, cfg.compile.calib_seed,
                run_dir / "calib_subset", letterbox=lb_spec,
                sampling=cfg.compile.calib_sampling,
            )
        elif dataset is not None:
            # 데이터셋 설정의 고정 목록 (calib_repeat 0) 또는 풀에서 묶음 번호로 추출한 목록
            subset, cinfo, samples = prepare_calib_from_list(
                dataset.calibration_sample(cfg.compile.calibration_num, cfg.compile.calib_repeat),
                run_dir / "calib_subset", letterbox=lb_spec, root=dataset.cfg.images_dir,
            )
        else:
            raise DxconvError("캘리브레이션 입력이 없습니다 (--calib-dir 또는 --dataset).")
    report.warn(*cinfo.warnings)
    pre_info: dict[str, Any] = {
        "letterbox_mode": pre.letterbox_mode,
        "target_hw": [h, w],
        "pad_value": pre.pad_value,
        "color": pre.color,
        "scale": pre.scale,
        "layout": pre.layout,
        "steps": [s["op"] for s in preprocess_steps(pre, h, w)],
    }
    if lb_spec is not None:
        pre_info["host_letterbox"] = letterbox_impl_info()
    report.set("preprocess", pre_info)
    if cinfo.broken:
        report.warn(f"열리지 않는 캘리브레이션 이미지 {len(cinfo.broken)} 개를 제외했습니다: "
                    + ", ".join(str(p) for p in cinfo.broken[:10]))
    report.section("inputs")["calib"] = {
        "dir": str((calib_dir or cinfo.root).resolve()),
        "source": "calib_dir" if calib_dir is not None else "dataset",
        "list": (str(dataset.calibration_list(cfg.compile.calibration_num))
                 if calib_dir is None and dataset is not None else None),
        "total_images": len(cinfo.images),
        "broken": [str(p) for p in cinfo.broken],
        "sampled": len(samples),
        "seed": cfg.compile.calib_seed if calib_dir is not None else None,
        "repeat": cfg.compile.calib_repeat if calib_dir is None and dataset is not None else None,
        "sampling": cinfo.sampling,
        "sampling_requested": cinfo.sampling_requested,
        "class_totals": cinfo.class_totals,
        "class_counts": cinfo.class_counts,
        "subset_dir": str(subset),
    }
    write_json([str(p) for p in samples], run_dir / "calib_samples.json")

    config = build_dxcom_config(cfg, info, subset, adapter)
    config_path = write_config(config, run_dir / "dxcom_config.json")
    report.section("outputs")["dxcom_config"] = str(config_path)

    onnx_for_cmd = onnx or run_dir / f"{cfg.model}.onnx"
    report.set("dxcom_command", adapter.build_command(
        onnx_for_cmd, config_path, run_dir / "dxcom_out", cfg))
    report.set("dxcom_adapter", type(adapter).__name__)

    if opts.dry_run:
        return config_path, None

    local = [d for d in env.dxcom if d.get("source") in ("pip", "path")]
    mismatch = bool(local) and not any(same_version(d.get("version"), cres.version) for d in local)
    cfg_run = cfg.model_copy(update={"compile": cfg.compile.model_copy(
        update={"dxcom_version": cres.version})})
    with report.stage("compile"):
        result = run_compile(adapter, cfg_run, onnx_for_cmd, config_path, run_dir,
                             timeout=opts.timeout, use_api=opts.use_api,
                             local_version_mismatch=mismatch)
    report.set("compile_summary", result.log_summary)
    report.set("dxcom_runner", result.runner)
    if cfg.compile.export_html:
        if result.html_reports:
            report.section("outputs")["html_reports"] = [str(p) for p in result.html_reports]
        else:
            report.warn("export_html=true 이지만 DX-COM 출력 폴더에서 HTML 리포트를 찾지 못했습니다.")
    report.warn(*[f"[DX-COM] {w}" for w in result.log_summary.get("warnings", [])[:10]])
    check_dxnn_settings(cfg, config, result, run_dir, report, calib_images=len(samples))
    report.section("outputs")["dxnn"] = str(result.dxnn_path)
    report.section("outputs")["compile_log"] = str(run_dir / "compile.log")
    return config_path, result.dxnn_path


def check_dxnn_settings(
    cfg: RunCfg, dxcom_config: dict[str, Any], result: Any, run_dir: Path, report: Report,
    calib_images: int | None = None,
) -> dict[str, Any]:
    """dxnn 에 실제 기록된 설정을 읽어 요청 설정과 비교 → report.dxnn_settings, 다르면 경고."""
    from dxconv.compile.adapters.v2_5 import FLAG_AGGRESSIVE_PARTITIONING
    from dxconv.compile.dxnn_info import DxnnFormatError, compare_settings, read_dxnn

    requested = {
        "calibration_method": cfg.compile.calibration_method,
        "calibration_num": cfg.compile.calibration_num,
        "opt_level": cfg.compile.opt_level,
        "aggressive_partitioning": cfg.compile.aggressive_partitioning,
        "preprocessings": [next(iter(p)) for p in
                           dxcom_config.get("default_loader", {}).get("preprocessings", []) if p],
        "input_shapes": dxcom_config.get("inputs"),
    }
    info = None
    try:
        info = read_dxnn(result.dxnn_path)
    except (OSError, DxnnFormatError) as e:
        report.warn(f"dxnn 에 기록된 실제 설정을 읽지 못했습니다: {e}")
    log_path = run_dir / "compile.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    res = compare_settings(requested, info, log_text, calib_images=calib_images)
    from dxconv.compile.adapters.v2_4 import aggressive_in_command

    cmd = result.command or []
    res["actual"]["aggressive_flag_in_command"] = (aggressive_in_command(cmd, FLAG_AGGRESSIVE_PARTITIONING)
                                                   if cmd else None)
    if cmd and res["actual"]["aggressive_flag_in_command"] != cfg.compile.aggressive_partitioning:
        res["mismatches"].append(
            f"aggressive_partitioning: 요청 {cfg.compile.aggressive_partitioning} ≠ "
            f"명령행 플래그 {res['actual']['aggressive_flag_in_command']}")
    report.set("dxnn_settings", res)
    report.warn(*[f"[dxnn 설정 불일치] {m}" for m in res["mismatches"]])
    return res


# --------------------------------------------------------------------------- #
# 명령 실행
# --------------------------------------------------------------------------- #
def execute(
    command: str,
    cfg: RunCfg,
    config_dir: Path,
    opts: RunOptions,
    *,
    pt: Path | None = None,
    onnx: Path | None = None,
    calib_dir: Path | None = None,
    dataset: Dataset | None = None,
    console: Console | None = None,
) -> RunResult:
    """command: export | compile | convert"""
    if command not in ("export", "compile", "convert"):
        raise ValueError(command)
    console = console or Console(stderr=True)
    run_dir = create_run_dir(opts.out_root, command, cfg.model)
    report = Report(run_dir=run_dir, command=command, model=cfg.model, dry_run=opts.dry_run)
    result = RunResult(run_dir=run_dir, report=report)
    save_resolved(cfg, run_dir / "resolved.yaml")
    report.set("params", cfg.model_dump(mode="json"))

    try:
        with report.stage("env"):
            env = detect_env()
        write_json(env.to_dict(), run_dir / "env.json")
        report.section("versions")["dxcom_detected"] = env.dxcom
        report.section("versions")["dxrt"] = env.dxrt

        if command in ("export", "convert"):
            assert pt is not None
            console.print("[bold]▶ PT → ONNX[/]" + (" (dry-run: 검증만)" if opts.dry_run else ""))
            result.onnx = export_stage(cfg, pt, run_dir, report, config_dir, opts.dry_run)
        else:
            assert onnx is not None
            if not onnx.is_file():
                raise DxconvError(f"ONNX 파일이 없습니다: {onnx}")
            dest = run_dir / f"{cfg.model}.onnx"
            shutil.copy2(onnx, dest)
            report.section("inputs")["onnx"] = {"path": str(onnx.resolve()),
                                                "sha256": sha256_file(onnx)}
            try:  # 입력 ONNX 를 만든 ultralytics 버전 (메타데이터)
                report.section("inputs")["onnx"].update(onnx_producer_info(onnx))
            except Exception as e:  # noqa: BLE001 - 메타데이터는 참고 정보
                report.warn(f"ONNX 메타데이터를 읽지 못했습니다: {e}")
            result.onnx = dest

        if command in ("compile", "convert"):
            if calib_dir is None and dataset is None:
                raise DxconvError("캘리브레이션 입력이 없습니다 (--calib-dir 또는 --dataset).")
            if dataset is not None:
                report.set("dataset", dataset.describe())
            console.print("[bold]▶ ONNX → DXNN[/]"
                          + (" (dry-run: config.json 생성까지)" if opts.dry_run else ""))
            result.dxcom_config, result.dxnn = compile_stage(
                cfg, result.onnx, calib_dir, run_dir, report, config_dir, env, opts, dataset
            )
    except BaseException as e:
        msg = str(e) if isinstance(e, DxconvError) else f"{type(e).__name__}: {e}"
        report.write(success=False, error=msg)
        raise
    report.write(success=True)
    return result


# --------------------------------------------------------------------------- #
# 기준선 (PT / FP32 ONNX mAP)
# --------------------------------------------------------------------------- #
def onnx_producer_info(path: Path) -> dict[str, Any]:
    """ONNX 메타데이터에서 export 도구 버전 (Ultralytics 는 metadata_props 'version')."""
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    props = {p.key: p.value for p in m.metadata_props}
    return {"producer": f"{m.producer_name} {m.producer_version}".strip(),
            "ultralytics_version": props.get("version"), "opset": m.opset_import[0].version}


def run_baseline(
    cfg: RunCfg,
    config_dir: Path,
    opts: RunOptions,
    *,
    pt: Path,
    dataset: Dataset,
    splits: list[str],
    onnx: list[Path] | None = None,
    console: Console | None = None,
) -> RunResult:
    """PT 와 FP32 ONNX 의 mAP 를 데이터셋 분할(tuning/evaluation)별로 계산한다.

    onnx 를 주지 않으면 현재 설정으로 export 한다. 여러 개를 주면 각각 평가한다
    (예: 다른 ultralytics 버전으로 만든 ONNX 비교).
    """
    from dxconv.evaluate.baseline import (
        OnnxRunner,
        PtRunner,
        predict_split,
        results_table,
        summarize,
        write_predictions,
    )

    console = console or Console(stderr=True)
    run_dir = create_run_dir(opts.out_root, "baseline", cfg.model)
    report = Report(run_dir=run_dir, command="baseline", model=cfg.model, dry_run=False)
    result = RunResult(run_dir=run_dir, report=report)
    save_resolved(cfg, run_dir / "resolved.yaml")
    report.set("params", cfg.model_dump(mode="json"))
    report.set("dataset", dataset.describe())
    max_det = cfg.evaluate.max_det or cfg.export.max_det
    report.set("eval_settings", {
        "conf": cfg.evaluate.conf, "max_det": max_det, "pr_conf": cfg.postprocess.conf,
        "imgsz": cfg.export.imgsz, "pad_value": cfg.compile.preprocess.pad_value,
        "preprocess": "ultralytics LetterBox (auto=False, center) → BGR2RGB → /255 → NCHW",
        "pt_head": "end2end (one2one)" if cfg.export.end2end else "raw",
        "metrics": ["coco (pycocotools 방식, maxDets 100)", "ultralytics val 방식",
                    f"P/R @conf {cfg.postprocess.conf}, IoU 0.5"],
    })
    try:
        if not pt.is_file():
            raise DxconvError(f"pt 파일이 없습니다: {pt}")
        report.section("inputs")["pt"] = {"path": str(pt.resolve()), "sha256": sha256_file(pt)}
        onnx_paths = list(onnx or [])
        if not onnx_paths:
            console.print("[bold]▶ PT → ONNX[/] (기준선용)")
            exported = export_stage(cfg, pt, run_dir, report, config_dir, dry_run=False)
            assert exported is not None
            onnx_paths = [exported]
        runners: dict[str, Any] = {"pt": PtRunner(pt, cfg.export.imgsz, cfg.export.end2end)}
        onnx_info: dict[str, Any] = {}
        for i, path in enumerate(onnx_paths):
            name = "onnx" if len(onnx_paths) == 1 else f"onnx{i + 1}"
            runners[name] = OnnxRunner(path)
            onnx_info[name] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                               **onnx_producer_info(path)}
        report.section("inputs")["onnx"] = onnx_info
        names = [v for _, v in sorted(runners["pt"].net.names.items())]
        labels_dir = dataset.cfg.labels_dir or dataset.cfg.images_dir.parent / "labels"
        results: dict[str, dict[str, Any]] = {}
        for split in splits:
            images = dataset.split_images(split)
            results[split] = {}
            for name, runner in runners.items():
                console.print(f"[bold]▶ {split} ({len(images)}장) · {name.upper()}[/]")
                with report.stage(f"{split}_{name}"):
                    dets, coco, ms = predict_split(
                        runner, images, labels_dir, conf=cfg.evaluate.conf, max_det=max_det,
                        pad_value=cfg.compile.preprocess.pad_value)
                summary = summarize(dets, len(names), cfg.postprocess.conf)
                summary["mean_infer_ms"] = round(ms, 2)
                summary["predictions_file"] = str(
                    write_predictions(coco, run_dir / "predictions" / f"{split}_{name}.json"))
                results[split][name] = summary
        report.set("class_names", names)
        report.set("baseline", results)
        table = results_table(results, names)
        (run_dir / "baseline_table.md").write_text(table, encoding="utf-8")
        report.section("outputs")["table"] = str(run_dir / "baseline_table.md")
        console.print(table)
    except BaseException as e:
        msg = str(e) if isinstance(e, DxconvError) else f"{type(e).__name__}: {e}"
        report.write(success=False, error=msg)
        raise
    report.write(success=True)
    return result
