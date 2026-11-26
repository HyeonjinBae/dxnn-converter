"""`dxconv remote setup|check`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dxconv.remote.client import RemoteClient, load_host


def client_for(host: str, config_dir: Path) -> RemoteClient:
    return RemoteClient(host, load_host(host, config_dir))


def check_remote(host: str, config_dir: Path, out_root: Path = Path("runs")) -> dict[str, Any]:
    """장비 상태를 읽어 <out>/remote/<host>.json 에 기록하고 반환."""
    info = client_for(host, config_dir).check()
    path = Path(out_root) / "remote" / f"{host}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    info["saved"] = str(path)
    return info


def setup_remote(host: str, config_dir: Path, out_root: Path = Path("runs")) -> dict[str, Any]:
    client_for(host, config_dir).setup()
    return check_remote(host, config_dir, out_root)
