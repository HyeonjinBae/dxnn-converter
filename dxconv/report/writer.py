"""run 폴더 생성과 report.json 기록.

runs/<command>/<model>_<YYYYMMDD_HHMMSS>/
├── <model>.onnx
├── <model>.dxnn
├── dxcom_config.json
├── resolved.yaml
├── env.json
├── compile.log
└── report.json
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dxconv import __version__


def create_run_dir(out_root: str | Path, command: str, model: str) -> Path:
    """runs/<command>/<model>_<YYYYMMDD_HHMMSS> 폴더 생성 (충돌 시 _1, _2 ...)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(out_root) / command / f"{model}_{stamp}"
    path = base
    i = 1
    while path.exists():
        path = base.with_name(f"{base.name}_{i}")
        i += 1
    path.mkdir(parents=True)
    return path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def write_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    return path


def _pkg_version(name: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass
class Report:
    run_dir: Path
    command: str
    model: str
    dry_run: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    _t0: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.data.update({
            "command": self.command,
            "model": self.model,
            "dry_run": self.dry_run,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "inputs": {},
            "outputs": {},
            "versions": {
                "dxconv": __version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "ultralytics": _pkg_version("ultralytics"),
                "onnx": _pkg_version("onnx"),
            },
        })

    def warn(self, *msgs: str) -> None:
        for m in msgs:
            if m not in self.warnings:
                self.warnings.append(m)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def section(self, key: str) -> dict[str, Any]:
        return self.data.setdefault(key, {})

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - t, 3)

    def write(self, success: bool, error: str | None = None) -> Path:
        self.data.update({
            "success": success,
            "error": error,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "total_sec": round(time.time() - self._t0, 3),
            "timings_sec": self.timings,
            "warnings": self.warnings,
        })
        return write_json(self.data, self.run_dir / "report.json")
