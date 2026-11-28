"""dxconv CLI 진입점 (typer)."""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape

from dxconv import DxconvError, __version__
from dxconv.config import find_config_dir, resolve_config

T = TypeVar("T")

console = Console(stderr=True)
out = Console()
_state: dict[str, Any] = {"debug": False}

app = typer.Typer(
    name="dxconv",
    help="DEEPX NPU용 Ultralytics YOLO 변환 자동화 (PT → ONNX → DXNN)",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
remote_app = typer.Typer(help="원격 NPU 장비 설정/점검 (configs/remote.yaml)", no_args_is_help=True)
app.add_typer(remote_app, name="remote")
dataset_app = typer.Typer(help="데이터셋 분할 설정 (configs/datasets/<model>.local.yaml)",
                          no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


# --------------------------------------------------------------------------- #
# 공통
# --------------------------------------------------------------------------- #
def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=debug, rich_tracebacks=debug)],
        force=True,
    )


def _version_cb(value: bool) -> None:
    if value:
        out.print(f"dxconv {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    debug: bool = typer.Option(False, "--debug", help="스택트레이스 및 디버그 로그 출력"),
    version: bool = typer.Option(False, "--version", callback=_version_cb, is_eager=True,
                                 help="버전 출력"),
) -> None:
    _state["debug"] = debug
    _setup_logging(debug)


def _guard(fn: Callable[[], T]) -> T:
    """오류를 사람이 읽을 수 있는 메시지로 바꾸고 종료 코드 1로 끝낸다."""
    try:
        return fn()
    except typer.Exit:
        raise
    except DxconvError as e:
        if _state["debug"]:
            console.print(traceback.format_exc(), markup=False, highlight=False)
        console.print(f"[bold red]오류:[/] {escape(str(e))}", highlight=False)
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("[red]중단됨[/]")
        raise typer.Exit(130) from None
    except Exception as e:  # noqa: BLE001
        if _state["debug"]:
            console.print(traceback.format_exc(), markup=False, highlight=False)
        else:
            console.print("[dim](자세한 내용은 --debug 로 확인)[/]")
        console.print(f"[bold red]예상치 못한 오류:[/] {type(e).__name__}: {escape(str(e))}",
                      highlight=False)
        raise typer.Exit(1) from None


def _finish(result: Any) -> None:  # noqa: ANN401
    from dxconv.report.table import summary_table

    rep = result.report
    for w in rep.warnings:
        console.print(f"[yellow]경고:[/] {escape(w)}", highlight=False)
    out.print(summary_table("실행 결과", {
        "run 폴더": result.run_dir,
        "ONNX": result.onnx,
        "DX-COM config": result.dxcom_config,
        "DXNN": result.dxnn,
        "report": result.run_dir / "report.json",
        "dry-run": rep.dry_run or None,
    }))
    out.print("[bold green]완료[/]")


# 공통 옵션 정의 ------------------------------------------------------------- #
ModelOpt = typer.Option(None, "--model",
                        help="모델 이름 (기본: pt/onnx 파일 이름). configs/presets/<이름>.yaml 이 있으면 함께 적용")
ModelReq = typer.Option(..., "--model", help="모델 이름 (sweep 을 실행한 이름)")


def _model_name(model: str | None, path: Path | None) -> str:
    """--model 이 없으면 입력 파일 이름(확장자 제외)을 모델 이름으로 쓴다."""
    import re

    if model:
        return model
    if path is None:
        raise DxconvError("--model 을 지정하세요.")
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", Path(path).stem) or "model"
    return name if name[0].isalnum() else f"m{name}"
ConfigOpt = typer.Option(None, "--config", exists=True, dir_okay=False,
                         help="사용자 설정 YAML (프리셋보다 우선)")
SetOpt = typer.Option(None, "--set", help="임의 키 오버라이드, 예: --set compile.calibration_num=200")
OutOpt = typer.Option(Path("runs"), "--out", help="결과 루트 폴더")
DryRunOpt = typer.Option(False, "--dry-run", help="설정 병합·검증·config.json 생성까지만")


def _cli_overrides(
    *,
    opset: int | None = None,
    imgsz: int | None = None,
    calib_num: int | None = None,
    calib_method: str | None = None,
    opt_level: int | None = None,
    aggressive: bool | None = None,
    export_html: bool | None = None,
    target_version: str | None = None,
) -> dict[str, Any]:
    return {
        "export": {"opset": opset, "imgsz": imgsz},
        "compile": {
            "calibration_num": calib_num,
            "calibration_method": calib_method,
            "opt_level": opt_level,
            "aggressive_partitioning": aggressive,
            "export_html": export_html,
            "dxcom_version": target_version,
        },
    }


TargetDxrtOpt = typer.Option(
    None, "--target-dxrt",
    help="배포 장비 DX-RT 버전 (예: 3.2.0). 읽을 수 있는 가장 새 dxnn 포맷·DX-COM 을 자동 선택. "
         "--remote 를 주면 장비에서 직접 읽는다")
DxnnFormatOpt = typer.Option(None, "--dxnn-format",
                             help="목표 dxnn 포맷: 9 (DX-COM 2.5.0, DX-RT>=3.5.0) | 8 (DX-COM 2.4.1, DX-RT>=3.1.0)")


def _format_version(fmt: int | None) -> str | None:
    """--dxnn-format → compile.dxcom_version (compat.yaml formats)."""
    if fmt is None:
        return None
    from dxconv.env.compat import load_compat, version_for_format

    return version_for_format(int(fmt), load_compat(find_config_dir()))


def _target_version(fmt: int | None, target_dxrt: str | None, remote: str | None) -> str | None:
    """--dxnn-format > --target-dxrt > --remote 장비의 DX-RT → 목표 DX-COM 버전 (없으면 설정값)."""
    if fmt is not None:
        return _format_version(fmt)
    if target_dxrt is None and remote is not None:
        from dxconv.remote.setup import client_for

        info = client_for(remote, find_config_dir()).check()
        if not info.get("dxrt"):
            raise DxconvError(f"원격 장비 {remote} 의 DX-RT 버전을 읽지 못했습니다: {info.get('problems')}")
        target_dxrt = info["dxrt"]
    if target_dxrt is None:
        return None
    from dxconv.env.compat import load_compat, version_for_dxrt

    ver, f = version_for_dxrt(target_dxrt, load_compat(find_config_dir()))
    console.print(f"[cyan]대상 DX-RT {target_dxrt} → dxnn v{f} (DX-COM {ver})[/]", highlight=False)
    return ver


def _verify(host: str, ds: Any, result: Any, cfg: Any, pt: Path | None) -> None:
    """convert/compile --remote: 만든 dxnn 을 장비에서 평가해 report.json 의 npu_verify 에 기록."""
    import json

    from dxconv.evaluate.remote_eval import verify_dxnn
    from dxconv.remote.setup import client_for

    console.print("[bold]▶ NPU 검증[/]")
    client = client_for(host, find_config_dir())
    client.setup()
    path = result.run_dir / "report.json"
    if ds is None or ds.cfg.labels_dir is None:
        # 라벨이 없으면 정확도는 잴 수 없고 FPS 만 측정
        import hashlib

        from dxconv.remote.client import fps_value

        sha = hashlib.sha256(result.dxnn.read_bytes()).hexdigest()
        bench = client.bench_fps(client.upload_dxnn(result.dxnn, sha), f"{sha[:16]}_fps.json")
        v = {"fps": fps_value(bench), "bench": bench,
             "note": "라벨 없음: 정확도(mAP) 미측정. --images + --labels 를 주면 PT 대비 하락까지 측정"}
        rep = json.loads(path.read_text(encoding="utf-8"))
        rep["npu_verify"] = v
        path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        out.print(f"  NPU FPS {v['fps'] or 0:.1f}  (라벨이 없어 mAP 미측정: --images + --labels 로 측정)")
        return
    v = verify_dxnn(client, ds, result.dxnn, cfg, result.run_dir / "npu_eval", pt=pt, console=console)
    rep = json.loads(path.read_text(encoding="utf-8"))
    rep["npu_verify"] = v
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    n = v["npu"]
    line = f"  NPU mAP50-95 {n['map50_95'] * 100:.2f}  mAP50 {n['map50'] * 100:.2f}  FPS {n['fps'] or 0:.1f}"
    if "pt" in v:
        line += (f"  (PT {v['pt']['map50_95'] * 100:.2f}, Δ mAP50-95 {v['delta_map50_95_pp']:+.2f}%p, "
                 f"Δ mAP50 {v['delta_map50_pp']:+.2f}%p)")
    out.print(line)


def _resolve_dataset(
    command: str, model: str, calib_dir: Path | None, dataset: str | None, config_dir: Path,
    images: Path | None = None, labels: Path | None = None, cfg: Any = None,
    out_root: Path = Path("runs"),
) -> Any:
    """캘리브레이션 입력 결정: --calib-dir > --images(+--labels) 자동 분할 > --dataset > <model>.local.yaml."""
    from dxconv.datasplit import resolve_dataset_input

    if command not in ("compile", "convert"):
        return None
    if calib_dir is not None and (dataset is not None or images is not None):
        raise DxconvError("--calib-dir 는 --dataset / --images 와 함께 쓸 수 없습니다.")
    if calib_dir is not None:
        return None
    return resolve_dataset_input(model, config_dir, dataset=dataset, images=images, labels=labels,
                                 split_cfg=cfg.split if cfg is not None else None,
                                 out_root=out_root, required=False, console=console)


def _run_pipeline(
    command: str,
    *,
    model: str,
    config: Path | None,
    set_items: list[str] | None,
    overrides: dict[str, Any],
    out_root: Path,
    dry_run: bool,
    pt: Path | None = None,
    onnx: Path | None = None,
    calib_dir: Path | None = None,
    dataset: str | None = None,
    images: Path | None = None,
    labels: Path | None = None,
    dxcom_version: str | None = None,
    timeout: float | None = None,
    use_api: bool = False,
    verify_remote: str | None = None,
) -> None:
    from dxconv.datasplit import fit_cfg_to_dataset
    from dxconv.pipeline import RunOptions, execute

    def _go() -> None:
        config_dir = find_config_dir()
        cfg = resolve_config(model, user_config=config, set_items=set_items or [],
                             cli_overrides=overrides, config_dir=config_dir)
        ds = _resolve_dataset(command, model, calib_dir, dataset, config_dir, images, labels, cfg,
                              out_root)
        calib = calib_dir
        if command in ("compile", "convert") and calib is None and ds is None:
            # pt 만 입력: 기본 캘리브레이션 사진 + 추천값의 settings_no_data
            from dxconv.compile.default_calib import default_calib_dir

            cfg = resolve_config(model, user_config=config, set_items=set_items or [],
                                 cli_overrides=overrides, config_dir=config_dir, no_data=True)
            calib = default_calib_dir(cfg.compile.default_calib_url, console=console)
            console.print(f"[cyan]이미지 입력 없음:[/] 기본 캘리브레이션 사진 사용 ({calib}). "
                          "배포 현장 사진이 있으면 --images 로 지정하세요.", highlight=False)
        explicit = (overrides.get("compile") or {}).get("calibration_num") is not None or any(
            str(i).startswith("compile.calibration_num=") for i in set_items or [])
        cfg = fit_cfg_to_dataset(cfg, ds, explicit_num=explicit, console=console)
        opts = RunOptions(out_root=out_root, dry_run=dry_run, dxcom_version=dxcom_version,
                          timeout=timeout, use_api=use_api)
        try:
            result = execute(command, cfg, config_dir, opts, pt=pt, onnx=onnx,
                             calib_dir=calib, dataset=ds, console=console)
        except BaseException:
            console.print("[dim]실패 내역은 run 폴더의 report.json 에 기록되었습니다.[/]")
            raise
        if verify_remote and result.dxnn is not None and not dry_run:
            _verify(verify_remote, ds, result, cfg, pt)
        _finish(result)

    _guard(_go)


# --------------------------------------------------------------------------- #
# 변환 명령 (env / export / compile / convert)
# --------------------------------------------------------------------------- #
@app.command("env")
def cmd_env(
    dxcom_version: str | None = typer.Option(None, "--dxcom-version", help="사용할 DX-COM 버전"),
    json_out: bool = typer.Option(False, "--json", help="JSON 으로 출력"),
) -> None:
    """설치된 DX-COM / DX-RT 버전 탐지 및 호환성 검사."""

    def _go() -> None:
        import json

        from dxconv.env import check_compat, detect_env, load_compat
        from dxconv.report.table import env_table

        env = detect_env()
        compat = load_compat(find_config_dir())
        cres = check_compat(dxcom_version, env, compat)
        if json_out:
            data = env.to_dict()
            data["selected_dxcom"] = {"version": cres.version, "source": cres.source,
                                      "warnings": cres.warnings}
            out.print_json(json.dumps(data, ensure_ascii=False))
            return
        out.print(env_table(env, cres))
        for n in env.notes:
            console.print(f"[cyan]안내:[/] {escape(n)}", highlight=False)
        for w in cres.warnings:
            console.print(f"[yellow]경고:[/] {escape(w)}", highlight=False)

    _guard(_go)


@app.command("export")
def cmd_export(
    pt: Path = typer.Option(..., "--pt", help="Ultralytics YOLO .pt 파일"),
    model: str | None = ModelOpt,
    config: Path | None = ConfigOpt,
    opset: int | None = typer.Option(None, "--opset", help="ONNX opset"),
    imgsz: int | None = typer.Option(None, "--imgsz", help="입력 크기 (32의 배수)"),
    set_items: list[str] | None = SetOpt,
    out_root: Path = OutOpt,
    dry_run: bool = typer.Option(False, "--dry-run", help="설정 병합·규칙 검증까지만"),
) -> None:
    """PT → ONNX 변환 (+ 규칙 검증, ONNX 검사)."""
    _run_pipeline("export", model=_model_name(model, pt), config=config, set_items=set_items,
                  overrides=_cli_overrides(opset=opset, imgsz=imgsz),
                  out_root=out_root, dry_run=dry_run, pt=pt)


_CompileOpts = dict(
    calib_dir=typer.Option(None, "--calib-dir",
                           help="캘리브레이션 이미지 폴더 (없으면 --dataset / datasets/<model>.local.yaml)"),
    dataset=typer.Option(None, "--dataset",
                         help="데이터셋 설정 이름(configs/datasets/<이름>.local.yaml) 또는 YAML 경로"),
    images=typer.Option(None, "--images", help="이미지 폴더 → 비율로 자동 분할 (split 설정)"),
    labels=typer.Option(None, "--labels", help="YOLO 라벨 폴더 (<stem>.txt). --images 와 함께"),
    calib_num=typer.Option(None, "--calib-num", help="캘리브레이션 이미지 수"),
    calib_method=typer.Option(None, "--calib-method", help="minmax | ema | iqr"),
    opt_level=typer.Option(None, "--opt-level", help="DX-COM 최적화 레벨 (0|1)"),
    aggressive=typer.Option(None, "--aggressive-partitioning/--no-aggressive-partitioning",
                            help="aggressive partitioning 사용 여부"),
    export_html=typer.Option(None, "--export-html/--no-export-html",
                             help="dxcom --export_html 로 HTML 요약을 만들어 run 폴더에 복사"),
    dxcom_version=typer.Option(None, "--dxcom-version", help="사용할 DX-COM 버전 (기본: 탐지된 최신)"),
    timeout=typer.Option(None, "--timeout", help="컴파일 타임아웃(초), 기본 없음"),
    use_api=typer.Option(False, "--use-api", help="CLI 대신 dx_com Python API 사용 (실험적)"),
)


@app.command("compile")
def cmd_compile(
    onnx: Path = typer.Option(..., "--onnx", help="입력 ONNX 파일"),
    model: str | None = ModelOpt,
    calib_dir: Path | None = _CompileOpts["calib_dir"],
    dataset: str | None = _CompileOpts["dataset"],
    images: Path | None = _CompileOpts["images"],
    labels: Path | None = _CompileOpts["labels"],
    config: Path | None = ConfigOpt,
    calib_num: int | None = _CompileOpts["calib_num"],
    calib_method: str | None = _CompileOpts["calib_method"],
    opt_level: int | None = _CompileOpts["opt_level"],
    aggressive: bool | None = _CompileOpts["aggressive"],
    export_html: bool | None = _CompileOpts["export_html"],
    dxcom_version: str | None = _CompileOpts["dxcom_version"],
    dxnn_format: int | None = DxnnFormatOpt,
    target_dxrt: str | None = TargetDxrtOpt,
    remote: str | None = typer.Option(None, "--remote", help="이 장비의 DX-RT 버전에 맞춰 포맷 선택 (remote.yaml)"),
    timeout: float | None = _CompileOpts["timeout"],
    use_api: bool = _CompileOpts["use_api"],
    set_items: list[str] | None = SetOpt,
    out_root: Path = OutOpt,
    dry_run: bool = DryRunOpt,
) -> None:
    """ONNX → DXNN 컴파일 (DX-COM config.json 자동 생성)."""
    _run_pipeline("compile", model=_model_name(model, onnx), config=config, set_items=set_items,
                  overrides=_cli_overrides(calib_num=calib_num, calib_method=calib_method,
                                           opt_level=opt_level, aggressive=aggressive,
                                           export_html=export_html,
                                           target_version=_target_version(dxnn_format, target_dxrt, remote)),
                  out_root=out_root, dry_run=dry_run, onnx=onnx, calib_dir=calib_dir, dataset=dataset,
                  images=images, labels=labels, dxcom_version=dxcom_version, timeout=timeout, use_api=use_api,
                  verify_remote=remote)


@app.command("convert")
def cmd_convert(
    pt: Path = typer.Option(..., "--pt", help="Ultralytics YOLO .pt 파일"),
    model: str | None = ModelOpt,
    calib_dir: Path | None = _CompileOpts["calib_dir"],
    dataset: str | None = _CompileOpts["dataset"],
    images: Path | None = _CompileOpts["images"],
    labels: Path | None = _CompileOpts["labels"],
    config: Path | None = ConfigOpt,
    opset: int | None = typer.Option(None, "--opset", help="ONNX opset"),
    imgsz: int | None = typer.Option(None, "--imgsz", help="입력 크기 (32의 배수)"),
    calib_num: int | None = _CompileOpts["calib_num"],
    calib_method: str | None = _CompileOpts["calib_method"],
    opt_level: int | None = _CompileOpts["opt_level"],
    aggressive: bool | None = _CompileOpts["aggressive"],
    export_html: bool | None = _CompileOpts["export_html"],
    dxcom_version: str | None = _CompileOpts["dxcom_version"],
    dxnn_format: int | None = DxnnFormatOpt,
    target_dxrt: str | None = TargetDxrtOpt,
    remote: str | None = typer.Option(None, "--remote", help="이 장비의 DX-RT 버전에 맞춰 포맷 선택 (remote.yaml)"),
    timeout: float | None = _CompileOpts["timeout"],
    use_api: bool = _CompileOpts["use_api"],
    set_items: list[str] | None = SetOpt,
    out_root: Path = OutOpt,
    dry_run: bool = DryRunOpt,
) -> None:
    """PT → ONNX → DXNN 한 번에 실행."""
    _run_pipeline("convert", model=_model_name(model, pt), config=config, set_items=set_items,
                  overrides=_cli_overrides(opset=opset, imgsz=imgsz, calib_num=calib_num,
                                           calib_method=calib_method, opt_level=opt_level,
                                           aggressive=aggressive, export_html=export_html,
                                           target_version=_target_version(dxnn_format, target_dxrt, remote)),
                  out_root=out_root, dry_run=dry_run, pt=pt, calib_dir=calib_dir, dataset=dataset,
                  images=images, labels=labels, dxcom_version=dxcom_version, timeout=timeout, use_api=use_api,
                  verify_remote=remote)


# --------------------------------------------------------------------------- #
# 기준선 / sweep / 추천값 / 데이터셋 / 장비
# --------------------------------------------------------------------------- #
@app.command("baseline")
def cmd_baseline(
    pt: Path = typer.Option(..., "--pt", help="Ultralytics YOLO .pt 파일"),
    model: str | None = ModelOpt,
    onnx: list[Path] | None = typer.Option(
        None, "--onnx", help="평가할 FP32 ONNX (여러 번 사용 가능). 없으면 현재 설정으로 export"),
    split: list[str] | None = typer.Option(
        None, "--split", help="tuning | evaluation (여러 번 사용 가능, 기본: 둘 다)"),
    dataset: str | None = typer.Option(
        None, "--dataset", help="데이터셋 설정 이름 또는 YAML 경로 (기본: <model>.local.yaml)"),
    images: Path | None = _CompileOpts["images"],
    labels: Path | None = _CompileOpts["labels"],
    config: Path | None = ConfigOpt,
    set_items: list[str] | None = SetOpt,
    out_root: Path = OutOpt,
) -> None:
    """PT / FP32 ONNX 기준선 mAP (tuning / evaluation 분할)."""

    def _go() -> None:
        from dxconv.datasplit import resolve_dataset_input
        from dxconv.pipeline import RunOptions, run_baseline

        config_dir = find_config_dir()
        name = _model_name(model, pt)
        cfg = resolve_config(name, user_config=config, set_items=set_items or [],
                             config_dir=config_dir)
        ds = resolve_dataset_input(name, config_dir, dataset=dataset, images=images, labels=labels,
                                   split_cfg=cfg.split, out_root=out_root, console=console)
        if ds.cfg.labels_dir is None:
            raise DxconvError("기준선 mAP 에는 라벨이 필요합니다 (--labels).")
        splits = split or ["tuning", "evaluation"]
        bad = [s for s in splits if s not in ("tuning", "evaluation")]
        if bad:
            raise DxconvError(f"알 수 없는 --split: {', '.join(bad)} (tuning | evaluation)")
        for o in onnx or []:
            if not o.is_file():
                raise DxconvError(f"ONNX 파일이 없습니다: {o}")
        result = run_baseline(cfg, config_dir, RunOptions(out_root=out_root), pt=pt, dataset=ds,
                              splits=splits, onnx=list(onnx or []), console=console)
        out.print(f"[bold green]완료[/] report: {result.run_dir / 'report.json'}")

    _guard(_go)


sweep_app = typer.Typer(help="설정 조합 비교 실험 (추천값 재검증용, 기본 사용에는 불필요)")
app.add_typer(sweep_app, name="sweep")


@sweep_app.callback(invoke_without_command=True)
def cmd_sweep(
    ctx: typer.Context,
    pt: Path | None = typer.Option(None, "--pt", help="Ultralytics YOLO .pt 파일"),
    model: str | None = typer.Option(None, "--model", help="모델 프리셋 이름"),
    stage: str | None = typer.Option(None, "--stage", help="이 단계만 실행 (A | B | C). 기본: A부터 순서대로"),
    compile_only: bool = typer.Option(False, "--compile-only",
                                      help="컴파일 + 검증까지만 (NPU 평가 없이, 평가는 pending)"),
    calib_dir: Path | None = typer.Option(None, "--calib-dir",
                                          help="캘리브레이션 폴더 (없으면 데이터셋 설정 사용)"),
    dataset: str | None = typer.Option(None, "--dataset",
                                       help="데이터셋 설정 이름 또는 YAML 경로 (기본: <model>.local.yaml)"),
    images: Path | None = typer.Option(None, "--images", help="이미지 폴더 → 비율로 자동 분할"),
    labels: Path | None = typer.Option(None, "--labels", help="YOLO 라벨 폴더 (--images 와 함께)"),
    space: Path = typer.Option(Path("configs/sweep/default_space.yaml"), "--space"),
    objective: str | None = typer.Option(None, "--objective",
                                         help='예: "PT 대비 mAP 하락 <= 1%p 중 FPS 최대"'),
    remote: str | None = typer.Option(None, "--remote", help="원격 NPU 호스트 이름 (configs/remote.yaml)"),
    dxnn_format: int | None = DxnnFormatOpt,
    target_dxrt: str | None = TargetDxrtOpt,
    strict_objective: bool = typer.Option(False, "--strict-objective",
                                          help="--objective 조건을 만족하는 조합이 없으면 고르지 않고 멈춤 "
                                               "(기본: 하락이 가장 작은 조합으로 대체 선택)"),
    save_preset: bool = typer.Option(False, "--save-preset", help="모든 단계 선택 시 프리셋으로 저장"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="캐시된 실패 조합도 다시 컴파일"),
    check_images: int = typer.Option(20, "--check-images", help="A 단계 FP32 ONNX=PT 확인 이미지 수"),
    timeout: float | None = typer.Option(None, "--timeout", help="조합당 컴파일 타임아웃(초)"),
    config: Path | None = ConfigOpt,
    set_items: list[str] | None = SetOpt,
    out_root: Path = OutOpt,
) -> None:
    """단계별 sweep. 결과: <out>/sweep/<model>/ (sweep.db, summary.md, pending_eval.json)."""
    if ctx.invoked_subcommand is not None:
        return

    def _go() -> None:
        from dxconv.sweep.stages import run_sweep

        if pt is None:
            raise DxconvError("--pt 가 필요합니다 (예: dxconv sweep --pt model.pt --images 이미지폴더 "
                              "--labels 라벨폴더).")
        name = _model_name(model, pt)
        ver = _format_version(dxnn_format)
        run_sweep(pt=pt, model=name, calib_dir=calib_dir, dataset=dataset,
                  images=images, labels=labels, target_dxrt=target_dxrt,
                  strict_objective=strict_objective,
                  space=space, objective=objective, remote=remote, save_preset=save_preset,
                  stage=stage.upper() if stage else None, compile_only=compile_only,
                  out_root=out_root, config=config,
                  set_items=[*(set_items or []), *([f"compile.dxcom_version={ver}"] if ver else [])],
                  retry_failed=retry_failed, check_images=check_images, timeout=timeout,
                  console=console)

    _guard(_go)


@sweep_app.command("import-eval")
def cmd_sweep_import_eval(
    json_path: Path = typer.Argument(..., help="평가 결과 JSON (pending_eval.json 형식)"),
    model: str = ModelReq,
    baseline_report: Path | None = typer.Option(
        None, "--baseline-report", help="dxconv baseline 의 report.json (PT Tuning 기준선)"),
    space: Path = typer.Option(Path("configs/sweep/default_space.yaml"), "--space"),
    dxnn_format: int | None = DxnnFormatOpt,
    out_root: Path = OutOpt,
) -> None:
    """NPU 평가 결과(Tuning mAP, FPS)를 수동으로 넣고 단계 선택을 다시 계산."""

    def _go() -> None:
        from dxconv.sweep.stages import import_eval

        import_eval(model=model, json_path=json_path, out_root=out_root,
                    baseline_report=baseline_report, space=space, console=console,
                    dxcom_version=_format_version(dxnn_format))

    _guard(_go)


@sweep_app.command("status")
def cmd_sweep_status(model: str = ModelReq, out_root: Path = OutOpt,
                     dxnn_format: int | None = DxnnFormatOpt) -> None:
    """sweep 결과 요약(summary.md)을 다시 만들고 출력."""

    def _go() -> None:
        from dxconv.sweep.stages import sweep_status

        path = sweep_status(model, out_root, _format_version(dxnn_format))
        out.print(path.read_text(encoding="utf-8"), markup=False, highlight=False)

    _guard(_go)


recommend_app = typer.Typer(help="추천 설정 (configs/recommend.yaml): dxnn 포맷별로 한 번에 컴파일할 값",
                            no_args_is_help=True)
app.add_typer(recommend_app, name="recommend")


@recommend_app.command("show")
def cmd_recommend_show() -> None:
    """포맷별 추천 설정과 근거."""

    def _go() -> None:
        from rich.table import Table

        from dxconv.recommend import _flatten, load

        doc = load(find_config_dir())
        for fmt, entry in sorted((doc.get("formats") or {}).items()):
            t = Table(title=f"dxnn v{fmt} 추천 설정")
            t.add_column("설정")
            t.add_column("값")
            for k, v in _flatten(entry.get("settings") or {}).items():
                t.add_row(k, str(v))
            out.print(t)
            for ev in entry.get("evidence") or []:
                out.print(f"  - {escape(str(ev))}", highlight=False)

    _guard(_go)


@recommend_app.command("update")
def cmd_recommend_update(
    roots: list[Path] = typer.Argument(..., help="sweep 결과 폴더 (예: runs/sweep/my_model)"),
) -> None:
    """sweep 선택 결과로 추천 설정을 갱신 (.bak 백업)."""

    def _go() -> None:
        from dxconv.recommend import update

        changed = update(roots, find_config_dir())
        for fmt, diff in changed.items():
            if not diff:
                out.print(f"v{fmt}: 변경 없음")
            for k, (old, new) in diff.items():
                out.print(f"v{fmt}: {k}: {old} → {new}")
        out.print("[green]갱신:[/] configs/recommend.yaml")

    _guard(_go)


@dataset_app.command("split")
def cmd_dataset_split(
    images: Path = typer.Option(..., "--images", help="이미지 폴더"),
    labels: Path | None = typer.Option(None, "--labels", help="YOLO 라벨 폴더 (<stem>.txt)"),
    model: str | None = ModelOpt,
    config: Path | None = ConfigOpt,
    set_items: list[str] | None = SetOpt,
    out_root: Path = OutOpt,
) -> None:
    """이미지/라벨 폴더를 비율(split 설정)로 캘리브레이션/튜닝/평가로 나눈다 (결과: <out>/datasets/)."""

    def _go() -> None:
        from rich.table import Table

        from dxconv.datasplit import auto_split, split_summary

        name = _model_name(model, images)
        cfg = resolve_config(name, user_config=config, set_items=set_items or [],
                             config_dir=find_config_dir())
        path = auto_split(images, labels, cfg.split, out_root / "datasets", name)
        info = split_summary(path)
        t = Table(title=f"자동 분할 {name} → {path}")
        for c in ("분할", "장수", "대표 클래스 분포 (이미지 수)"):
            t.add_column(c)
        sz = info["sizes"]
        rows = [("전체 (중복 제외)", info["used_images"], info["classes"]["all"]["group"]),
                (f"캘리브레이션 후보 {sz['calibration_candidates']}", sz["calibration_pool"],
                 info["classes"]["calibration"]["group"]),
                ("튜닝", sz["tuning"], info["classes"]["tuning"]["group"]),
                ("평가", sz["evaluation"], info["classes"]["evaluation"]["group"])]
        for name, n, grp in rows:
            t.add_row(name, str(n), ", ".join(f"{k} {v}" for k, v in grp.items()))
        out.print(t)
        for w in info["warnings"]:
            console.print(f"[yellow]경고:[/] {escape(w)}", highlight=False)
        out.print(f"다음: dxconv sweep --pt <pt> --model {name} --dataset {path}")

    _guard(_go)


@dataset_app.command("check")
def cmd_dataset_check(
    dataset: str = typer.Argument(..., help="모델/데이터셋 이름(예: fire) 또는 YAML 경로"),
    json_out: bool = typer.Option(False, "--json", help="JSON 으로 출력"),
) -> None:
    """분할 검증: 파일 존재, 캘리브레이션 중첩, 분할 간 겹침(SHA-256)."""

    def _go() -> None:
        import json

        from rich.table import Table

        from dxconv.dataset import load_dataset

        ds = load_dataset(dataset, find_config_dir())
        res = ds.check()
        if json_out:
            out.print_json(json.dumps({"dataset": ds.describe(), "check": res}, ensure_ascii=False))
        else:
            t = Table(title=f"데이터셋 {ds.name} ({ds.source})")
            t.add_column("항목")
            t.add_column("값")
            t.add_row("images_dir", str(ds.cfg.images_dir))
            for k, n in res["counts"].items():
                t.add_row(k, f"{n} 장")
            t.add_row("캘리브레이션 중첩", str(res["nested"]))
            for k, n in res["overlaps"].items():
                t.add_row(f"겹침 {k}", f"{n} 장")
            t.add_row("결과", "[green]OK[/]" if res["ok"] else "[red]문제 있음[/]")
            out.print(t)
            for msg in res["problems"]:
                console.print(f"[red]문제:[/] {escape(msg)}", highlight=False)
        if not res["ok"]:
            raise DxconvError("데이터셋 검증 실패")

    _guard(_go)


def _print_remote(info: dict[str, Any]) -> None:
    from rich.table import Table

    t = Table(title=f"원격 장비 {info['host']} ({info['target']})")
    t.add_column("항목")
    t.add_column("값")
    rows = [("hostname / arch", f"{info.get('hostname')} / {info.get('arch')}"),
            ("NPU", f"{info.get('device')} (코어 {info.get('npu_cores')})"),
            ("DX-RT", info.get("dxrt")), ("RT 드라이버", info.get("rt_driver")),
            ("PCIe 드라이버", info.get("pcie_driver")), ("펌웨어", info.get("firmware")),
            ("읽을 수 있는 dxnn 포맷", ", ".join(f"v{f}" for f in info.get("dxnn_formats", []))),
            ("평가 python", f"{info.get('python_path')} ({info.get('python_source')})"),
            ("dx_engine / OpenCV / numpy", f"{info.get('dx_engine')} / {info.get('opencv')} / {info.get('numpy')}"),
            ("작업 폴더", info.get("workdir")),
            ("디스크 여유", f"{info.get('disk_free_gb')} GB"), ("측정 lock", info.get("lock")),
            ("기록", info.get("saved"))]
    for k, v in rows:
        t.add_row(k, str(v))
    t.add_row("결과", "[green]OK[/]" if info["ok"] else "[red]문제 있음[/]")
    out.print(t)
    for msg in info["problems"]:
        console.print(f"[red]문제:[/] {escape(msg)}", highlight=False)


@remote_app.command("setup")
def cmd_remote_setup(host: str = typer.Option(..., "--host", help="configs/remote.yaml 의 호스트"),
                     out_root: Path = OutOpt) -> None:
    """원격 장비에 작업 폴더와 device_agent 를 동기화하고 상태를 점검."""

    def _go() -> None:
        from dxconv.remote.setup import setup_remote

        info = setup_remote(host, find_config_dir(), out_root)
        _print_remote(info)
        if not info["ok"]:
            raise DxconvError("원격 장비 점검 실패")

    _guard(_go)


@remote_app.command("check")
def cmd_remote_check(host: str = typer.Option(..., "--host", help="configs/remote.yaml 의 호스트"),
                     out_root: Path = OutOpt) -> None:
    """원격 장비 상태 점검 (DX-RT/드라이버/펌웨어 버전, 읽을 수 있는 dxnn 포맷, 평가 python)."""

    def _go() -> None:
        from dxconv.remote.setup import check_remote

        info = check_remote(host, find_config_dir(), out_root)
        _print_remote(info)
        if not info["ok"]:
            raise DxconvError("원격 장비 점검 실패")

    _guard(_go)


def main() -> None:
    try:
        app()
    except SystemExit as e:
        sys.exit(e.code)


if __name__ == "__main__":
    main()
