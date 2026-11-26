"""DX-COM 버전별 어댑터 추상 클래스."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dxconv import DxconvError
from dxconv.compile.onnx_check import OnnxInfo
from dxconv.config import RunCfg


class DxcomAdapter(ABC):
    """DX-COM 버전별 차이(config.json 키, CLI 플래그, 로그 형식)를 흡수한다."""

    #: 이 어댑터가 담당하는 버전 계열 (예: "2.5")
    series: str = ""
    #: PATH 에서 찾을 실행 파일 이름 후보
    executables: tuple[str, ...] = ("dxcom", "dx_com")

    def __init__(self, version: str) -> None:
        self.version = version

    @abstractmethod
    def build_config(self, cfg: RunCfg, onnx_info: OnnxInfo, calib_dir: Path) -> dict[str, Any]:
        """DX-COM config.json 내용을 만든다."""

    @abstractmethod
    def build_command(
        self, onnx: Path, config_path: Path, out_dir: Path, cfg: RunCfg
    ) -> list[str]:
        """DX-COM CLI 명령행을 만든다."""

    @abstractmethod
    def parse_log(self, text: str) -> dict[str, Any]:
        """컴파일 로그에서 NPU/CPU 파티션 정보, 경고 등을 추출한다."""

    def compile_via_api(
        self, onnx: Path, config_path: Path, out_dir: Path, cfg: RunCfg
    ) -> None:
        """Python API 로 컴파일 (지원 버전만). 기본은 미지원."""
        raise DxconvError(f"DX-COM {self.version} 어댑터는 Python API 경로를 지원하지 않습니다.")

    def find_executable(self) -> str:
        for name in self.executables:
            path = shutil.which(name)
            if path:
                return path
        raise DxconvError(
            "DX-COM 실행 파일을 찾을 수 없습니다 "
            f"(찾은 이름: {', '.join(self.executables)}). "
            "`dxconv env` 로 설치 상태를 확인하거나, --dry-run 으로 config.json 만 생성하세요."
        )
