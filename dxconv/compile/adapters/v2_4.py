"""DX-COM 2.4.x 어댑터 (dxnn 포맷 v8, DX-RT >= 3.1.0).

config.json 키·CLI 플래그는 2.5 와 같다 (dxcom --help 2.4.1 비교). 단, 실측 결과:

- `dxcom -c config.json` (CLI) 로 컴파일하면 config 의 calibration_method / calibration_num 이
  반영되지 않고 dxnn 에 ema / 100 으로 기록된다 (실제 사용 장수는 넣어 준 이미지 수, 최대 100).
  기존 Fire 2.4.1 실험(minmax/50/200 요청 → 모두 ema/100)과 같은 현상.
- Python API `dx_com.compile(..., calibration_method=..., calibration_num=...)` 로 값을 직접 넘기면
  요청대로 기록된다 (2026-11-25, minmax/50 확인).

그래서 2.4 는 CLI 대신 `python -c "import dx_com; dx_com.compile(...)"` 명령으로 컴파일한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from dxconv.compile.adapters.v2_5 import GEN_LOG_DEFAULT, DxcomV25Adapter
from dxconv.config import RunCfg

API_MARK = "dx_com.compile("


class DxcomV24Adapter(DxcomV25Adapter):
    series = "2.4"
    executables = ("python3", "python")

    def build_command(
        self, onnx: Path, config_path: Path, out_dir: Path, cfg: RunCfg,
        executable: str | None = None,
    ) -> list[str]:
        kw = {
            "model": str(onnx),
            "output_dir": str(out_dir),
            "config": str(config_path),
            "calibration_method": cfg.compile.calibration_method,
            "calibration_num": cfg.compile.calibration_num,
            "opt_level": cfg.compile.opt_level,
            "aggressive_partitioning": cfg.compile.aggressive_partitioning,
            "gen_log": GEN_LOG_DEFAULT,
            "export_html": cfg.compile.export_html,
        }
        args = ", ".join(f"{k}={json.dumps(v) if not isinstance(v, bool) else v}" for k, v in kw.items())
        script = f"import dx_com; {API_MARK}{args})"
        # dx_com 2.4 가 설치된 python 으로 실행
        return [executable or "python3", "-c", script]

    def find_executable(self) -> str:
        """dx_com 2.4 가 import 되는 python 을 찾는다."""
        import shutil
        import subprocess

        from dxconv import DxconvError

        for name in self.executables:
            path = shutil.which(name)
            if not path:
                continue
            p = subprocess.run([path, "-c", "import dx_com,sys; print(dx_com.__version__)"],
                               capture_output=True, text=True, timeout=60)
            if p.returncode == 0 and p.stdout.strip().startswith(self.series):
                return path
        raise DxconvError(f"dx_com {self.series}.x 가 설치된 python 이 없습니다 "
                          "(DXCOM_VERSION=2.4.1 ./scripts/install_host.sh).")


def aggressive_in_command(cmd: list[str], flag: str) -> bool:
    """명령행에 aggressive partitioning 이 켜져 있는지 (CLI 플래그 또는 API 인자)."""
    if flag in cmd:
        return True
    return any(API_MARK in c and "aggressive_partitioning=True" in c for c in cmd)
