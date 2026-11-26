#!/usr/bin/env python3
"""NPU 장비 환경 점검 → JSON (dxconv remote check 와 같은 정보를 장비에서 직접 볼 때)."""

from __future__ import annotations

import json
import re
import subprocess


def main() -> None:
    info: dict = {}
    try:
        st = subprocess.run(["dxrt-cli", "--status"], capture_output=True, text=True, timeout=30).stdout
        st = re.sub(r"\x1b\[[0-9;]*m", "", st)
        for key, pat in (("dxrt", r"DXRT v([\d.]+)"), ("rt_driver", r"RT Driver version\s*:\s*v([\d.]+)"),
                         ("pcie_driver", r"PCIe Driver version\s*:\s*v([\d.]+)"),
                         ("firmware", r"FW version\s*:\s*v([\d.]+)")):
            m = re.search(pat, st)
            info[key] = m.group(1) if m else None
    except (OSError, subprocess.TimeoutExpired) as e:
        info["dxrt_error"] = str(e)
    for mod in ("dx_engine", "cv2", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            info[mod] = f"error: {e}"
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
