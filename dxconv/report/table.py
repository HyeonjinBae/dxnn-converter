"""rich 콘솔 표 출력."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.table import Table

from dxconv.env.compat import CompatResult
from dxconv.env.detect import EnvInfo


def env_table(env: EnvInfo, compat: CompatResult | None = None) -> Table:
    t = Table(title="DEEPX 설치 환경", show_lines=False)
    t.add_column("항목", style="bold")
    t.add_column("출처")
    t.add_column("버전")
    t.add_column("경로/정보", overflow="fold")
    if env.dxcom:
        for d in env.dxcom:
            t.add_row("DX-COM", d["source"], str(d.get("version") or "?"), str(d.get("path") or ""))
    else:
        t.add_row("DX-COM", "-", "[red]미설치[/]", "")
    if env.dxrt:
        info = f"driver={env.dxrt.get('driver') or '?'}, firmware={env.dxrt.get('firmware') or '?'}"
        t.add_row("DX-RT", "dxrt-cli", str(env.dxrt.get("version") or "?"), info)
    else:
        t.add_row("DX-RT", "-", "[yellow]없음[/]", "NPU 없음")
    t.add_row("NPU", "-", "[green]있음[/]" if env.npu_present else "[yellow]없음[/]", "")
    if compat:
        t.add_row("선택된 DX-COM", compat.source, compat.version,
                  str(compat.compat_entry) if compat.compat_entry else "compat.yaml 정보 없음")
    return t


def summary_table(title: str, rows: dict[str, Any]) -> Table:
    t = Table(title=title, show_header=False)
    t.add_column("key", style="bold")
    t.add_column("value", overflow="fold")
    for k, v in rows.items():
        if v is None:
            continue
        t.add_row(k, str(v) if not isinstance(v, Path) else str(v))
    return t
