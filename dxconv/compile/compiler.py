"""DX-COM 실행 (subprocess 또는 Python API)."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from dxconv import DxconvError
from dxconv.compile.adapters.base import DxcomAdapter
from dxconv.config import RunCfg

log = logging.getLogger(__name__)


@dataclass
class CompileResult:
    dxnn_path: Path
    command: list[str] | None
    returncode: int
    duration_sec: float
    log_summary: dict[str, Any] = field(default_factory=dict)
    html_reports: list[Path] = field(default_factory=list)
    runner: str = "local"  # local | api


def find_dxnn(out_dir: Path, started_at: float | None = None) -> Path | None:
    """out_dir 아래에서 가장 최근의 .dxnn 파일을 찾는다."""
    cands = [p for p in out_dir.rglob("*.dxnn") if p.is_file()]
    if started_at is not None:
        fresh = [p for p in cands if p.stat().st_mtime >= started_at - 1]
        cands = fresh or cands
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def find_html_reports(out_dir: Path, started_at: float | None = None) -> list[Path]:
    """--export_html 결과 HTML 을 찾는다 (quant_diagnosis/ 아래 진단 리포트는 제외)."""
    cands = [p for p in out_dir.rglob("*.html")
             if p.is_file() and "quant_diagnosis" not in p.relative_to(out_dir).parts]
    if started_at is not None:
        cands = [p for p in cands if p.stat().st_mtime >= started_at - 1]
    return sorted(cands)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _status_text(line: str) -> str:
    """DX-COM 출력 한 줄 → 상태 표시용 텍스트.

    진행바(tqdm)는 \r 로 같은 줄을 덮어쓰므로 마지막 조각만 쓰고, 제어 문자를 지운다
    (그대로 넘기면 상태 표시가 끝난 뒤에도 진행바 조각이 화면에 남음).
    """
    seg = [x for x in _ANSI.sub("", line).replace("\r", "\n").split("\n") if x.strip()]
    return "".join(ch for ch in seg[-1] if ch.isprintable()).strip() if seg else ""


def _stream_process(
    cmd: list[str], log_path: Path, timeout: float | None, console: Console,
    on_kill: Any = None,
) -> tuple[int, str, bool]:
    """프로세스를 실행하며 출력을 log 파일에 실시간 기록. (returncode, 전체 출력, timeout 여부)"""
    chunks: list[str] = []
    timed_out = threading.Event()
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"$ {' '.join(cmd)}\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace",
            )
        except OSError as e:
            raise DxconvError(f"DX-COM 실행 실패: {e}") from e

        timer: threading.Timer | None = None
        if timeout:
            def _kill() -> None:
                timed_out.set()
                proc.kill()
                if on_kill:
                    on_kill()
            timer = threading.Timer(timeout, _kill)
            timer.daemon = True
            timer.start()
        try:
            with console.status("[bold cyan]DX-COM 컴파일 중...") as status:
                assert proc.stdout is not None
                for line in proc.stdout:
                    logf.write(line)
                    logf.flush()
                    chunks.append(line)
                    last = _status_text(line)
                    if last:
                        status.update(f"[bold cyan]DX-COM 컴파일 중...[/] [dim]{escape(last[:100])}")
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            if on_kill:
                on_kill()
            raise
        finally:
            if timer:
                timer.cancel()
    return rc, "".join(chunks), timed_out.is_set()


def run_compile(
    adapter: DxcomAdapter,
    cfg: RunCfg,
    onnx: Path,
    config_path: Path,
    run_dir: Path,
    *,
    timeout: float | None = None,
    use_api: bool = False,
    console: Console | None = None,
    local_version_mismatch: bool = False,
) -> CompileResult:
    """DX-COM 컴파일 실행 후 결과 .dxnn (및 --export_html 결과) 을 run 폴더 최상위로 복사한다."""
    console = console or Console(stderr=True)
    out_dir = run_dir / "dxcom_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "compile.log"
    t0 = time.time()

    cmd: list[str] | None = None
    runner = "local"
    if use_api:
        runner = "api"
        with log_path.open("a", encoding="utf-8") as f:
            f.write("# DX-COM Python API 경로 사용\n")
        with console.status("[bold cyan]DX-COM 컴파일 중 (Python API)..."):
            try:
                adapter.compile_via_api(onnx, config_path, out_dir, cfg)
            except DxconvError:
                raise
            except Exception as e:  # noqa: BLE001
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"ERROR: {e}\n")
                raise DxconvError(f"DX-COM(Python API) 컴파일 실패: {e}") from e
        rc, text = 0, log_path.read_text(encoding="utf-8")
    else:
        if local_version_mismatch:
            want = cfg.compile.dxcom_version
            raise DxconvError(
                f"설치된 DX-COM 버전이 목표 버전({want})과 다릅니다. 목표 버전은 별도 폴더에 "
                f"`DXCOM_VERSION={want} ./scripts/install_host.sh` 로 설치한 가상환경에서 실행하세요 "
                "(가상환경 하나에 DX-COM 한 버전).")
        exe = adapter.find_executable()
        cmd = adapter.build_command(onnx, config_path, out_dir, cfg, executable=exe)  # type: ignore[call-arg]
        rc, text, timed_out = _stream_process(cmd, log_path, timeout, console)
        if timed_out:
            raise DxconvError(f"DX-COM 컴파일 타임아웃 ({timeout}초). 로그: {log_path}")
        if rc != 0:
            tail = "\n".join(text.strip().splitlines()[-15:])
            raise DxconvError(
                f"DX-COM 컴파일 실패 (종료 코드 {rc}). 로그: {log_path}\n--- 마지막 출력 ---\n{tail}"
            )

    found = find_dxnn(out_dir, started_at=t0)
    if found is None:
        raise DxconvError(f"컴파일은 끝났지만 .dxnn 파일을 찾을 수 없습니다: {out_dir}")
    dest = run_dir / f"{cfg.model}.dxnn"
    shutil.copy2(found, dest)

    html_reports: list[Path] = []
    if cfg.compile.export_html:
        for src in find_html_reports(out_dir, started_at=t0):
            h_dest = run_dir / src.name
            if h_dest.exists():
                h_dest = run_dir / f"{src.parent.name}_{src.name}"
            shutil.copy2(src, h_dest)
            html_reports.append(h_dest)

    return CompileResult(
        dxnn_path=dest,
        command=cmd,
        returncode=rc,
        duration_sec=round(time.time() - t0, 3),
        log_summary=adapter.parse_log(text),
        html_reports=html_reports,
        runner=runner,
    )
