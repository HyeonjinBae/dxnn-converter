"""DX-COM 2.5.x 어댑터.

CLI 플래그는 `dxcom --help` (DX-COM 2.5.0), config 키·로그 형식은 dxcom:2.5.0 / pip dx-com 2.5.0 실측으로 확인.
기본 경로(CLI + host letterbox)는 macOS(docker)와 Linux/WSL2(pip)에서 검증됨.
dxcom 쪽 letterbox(letterbox_mode: dxcom)와 Python API(--use-api) 경로는 미검증 (아래 주석).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

from dxconv import DxconvError
from dxconv.compile.adapters.base import DxcomAdapter
from dxconv.compile.calib import IMAGE_EXTS
from dxconv.compile.config_builder import input_hw, preprocess_steps
from dxconv.compile.onnx_check import OnnxInfo
from dxconv.config import RunCfg

# --------------------------------------------------------------------------- #
# CLI 플래그 (dxcom --help 2.5.0 으로 확인)
# --------------------------------------------------------------------------- #
FLAG_MODEL = "-m"  # --model_path
FLAG_CONFIG = "-c"  # --config_path
FLAG_OUTPUT = "-o"  # --output_dir
FLAG_OPT_LEVEL = "--opt_level"  # {0,1}, 기본 1
FLAG_GEN_LOG = "--gen_log"  # 값 없는 스위치. 출력 폴더에 compiler.log 생성
FLAG_AGGRESSIVE_PARTITIONING = "--aggressive_partitioning"  # 값 없는 스위치
FLAG_EXPORT_HTML = "--export_html"  # 값 없는 스위치. 컴파일 후 self-contained HTML 요약
GEN_LOG_DEFAULT = True

# --------------------------------------------------------------------------- #
# config.json 키
# --------------------------------------------------------------------------- #
KEY_INPUTS = "inputs"
KEY_CALIB_METHOD = "calibration_method"
KEY_CALIB_NUM = "calibration_num"
KEY_DEFAULT_LOADER = "default_loader"
KEY_DATASET_PATH = "dataset_path"
KEY_FILE_EXTENSIONS = "file_extensions"
KEY_PREPROCESSINGS = "preprocessings"

#: calibration_method 값. 도움말의 observer 목록(--recalibration_method: minmax/ema/iqr)과 같은
#: 소문자 표기를 그대로 config.json 에 쓴다 (dxconv.config.CalibMethod 에서 정규화됨).
CALIB_METHOD_VALUES = {"minmax": "minmax", "ema": "ema", "iqr": "iqr"}

#: 전처리 op → DX-COM preprocessings 항목 (도움말에 없음 → 매뉴얼로 확인 필요).
#: 실제 실행(dxcom:2.5.0)으로 확인: resize(width/height), convertColor(form=BGR2RGB), div(x),
#: transpose(axis), expandDim(axis) 는 config 로 받아들여지고 컴파일에 성공했다.
PRE_RESIZE = "resize"
PRE_RESIZE_PAD_MODE = "pad"
# letterbox_mode: dxcom 일 때만 쓰임. DX-COM 2.5.0 에서 "pad_location: CENTER not implemented in Resize_Pad"
# 오류가 나서 Ultralytics LetterBox(center=True) 를 재현할 수 없음 → 기본값은 host (PC 에서 미리 letterbox).
PRE_RESIZE_PAD_LOCATION = "center"
PRE_CONVERT_COLOR = "convertColor"
PRE_BGR2RGB_FORM = "BGR2RGB"
PRE_DIV = "div"
PRE_TRANSPOSE = "transpose"
PRE_EXPAND_DIM = "expandDim"

# --------------------------------------------------------------------------- #
# 로그 파싱 패턴
# 실제 DX-COM 2.5.0 로그(docker dxcom:2.5.0)에서 확인한 줄:
#   "[INFO] - Final result: 1 NPU groups, 0 CPU groups"
# 아래 _RE_NPU/_RE_CPU 는 위 줄이 없을 때의 예비 패턴 (미검증).
# --------------------------------------------------------------------------- #
_RE_FINAL = re.compile(r"Final result:\s*(\d+)\s*NPU groups?,\s*(\d+)\s*CPU groups?", re.I)
_RE_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_RE_NPU = re.compile(r"\bnpu[_ ]?(?:sub)?(?:graph|partition)s?\b\D{0,20}(\d+)", re.I)
_RE_CPU = re.compile(r"\bcpu[_ ]?(?:sub)?(?:graph|partition)s?\b\D{0,20}(\d+)", re.I)
_RE_WARN = re.compile(r"\b(warn(?:ing)?)\b", re.I)
# 로그 레벨이 WARNING 인 줄: "[WARNING] - ...", "WARNING: ...", "<시각> - WARNING - ..."
_RE_WARN_LEVEL = re.compile(r"^\s*\[?\s*warn(?:ing)?\s*\]?\s*[-:]|\s-\s*warn(?:ing)?\s*-\s", re.I)
_RE_ERROR = re.compile(r"\b(error|exception|traceback|failed)\b", re.I)


def _pre_to_dxcom(step: dict[str, Any]) -> dict[str, Any]:
    op = step["op"]
    if op == "letterbox":
        v = step["pad_value"]
        return {PRE_RESIZE: {
            "mode": PRE_RESIZE_PAD_MODE,
            "size": [step["width"], step["height"]],  # dxcom letterbox 경로 미검증 (위 주석)
            "pad_location": PRE_RESIZE_PAD_LOCATION,
            "pad_value": [v, v, v],
        }}
    if op == "resize":
        return {PRE_RESIZE: {"width": step["width"], "height": step["height"]}}
    if op == "bgr2rgb":
        return {PRE_CONVERT_COLOR: {"form": PRE_BGR2RGB_FORM}}
    if op == "div":
        return {PRE_DIV: {"x": step["value"]}}
    if op == "transpose":
        return {PRE_TRANSPOSE: {"axis": step["axis"]}}
    if op == "expand_dim":
        return {PRE_EXPAND_DIM: {"axis": step["axis"]}}
    raise DxconvError(f"알 수 없는 전처리 단계: {op}")


class DxcomV25Adapter(DxcomAdapter):
    series = "2.5"

    # ------------------------------------------------------------------ config
    def build_config(self, cfg: RunCfg, onnx_info: OnnxInfo, calib_dir: Path) -> dict[str, Any]:
        if not onnx_info.inputs:
            raise DxconvError("ONNX 입력 정보가 없습니다.")
        pre = cfg.compile.preprocess
        inp = onnx_info.primary_input
        h, w = input_hw(inp, pre.layout)
        exts = sorted({e.lstrip(".") for e in IMAGE_EXTS} | {e.lstrip(".").upper()
                                                              for e in IMAGE_EXTS})
        return {
            KEY_INPUTS: {t.name: list(t.shape) for t in onnx_info.inputs},
            KEY_CALIB_METHOD: CALIB_METHOD_VALUES[cfg.compile.calibration_method],
            KEY_CALIB_NUM: cfg.compile.calibration_num,
            KEY_DEFAULT_LOADER: {
                KEY_DATASET_PATH: str(Path(calib_dir).resolve()),
                KEY_FILE_EXTENSIONS: exts,
                KEY_PREPROCESSINGS: [_pre_to_dxcom(s) for s in preprocess_steps(pre, h, w)],
            },
        }

    # ----------------------------------------------------------------- command
    def build_command(
        self, onnx: Path, config_path: Path, out_dir: Path, cfg: RunCfg,
        executable: str | None = None,
    ) -> list[str]:
        cmd = [
            executable or self.executables[0],
            FLAG_MODEL, str(onnx),
            FLAG_CONFIG, str(config_path),
            FLAG_OUTPUT, str(out_dir),
            FLAG_OPT_LEVEL, str(cfg.compile.opt_level),
        ]
        if GEN_LOG_DEFAULT:
            cmd.append(FLAG_GEN_LOG)
        if cfg.compile.aggressive_partitioning:
            cmd.append(FLAG_AGGRESSIVE_PARTITIONING)
        if cfg.compile.export_html:
            cmd.append(FLAG_EXPORT_HTML)
        return cmd

    # --------------------------------------------------------------------- api
    def compile_via_api(
        self, onnx: Path, config_path: Path, out_dir: Path, cfg: RunCfg
    ) -> None:
        """dx_com Python API 로 컴파일.

        실험 기능 (--use-api). 2.5.0 은 CLI 경로로 검증했고 Python API 인자명은 미검증이라,
        dx_com.compile 의 시그니처를 검사해 존재하는 인자만 전달한다.
        """
        try:
            import dx_com  # type: ignore[import-not-found]
        except ImportError as e:
            raise DxconvError("dx_com Python 패키지가 없습니다 (pip install dx-com==<버전>).") from e
        fn = getattr(dx_com, "compile", None)
        if fn is None:
            raise DxconvError("dx_com.compile 함수를 찾을 수 없습니다. CLI 경로를 사용하세요.")
        candidates: dict[str, Any] = {
            "model": str(onnx),
            "config": str(config_path),
            "output_dir": str(out_dir),
            "opt_level": cfg.compile.opt_level,
            "aggressive_partitioning": cfg.compile.aggressive_partitioning,
            "gen_log": GEN_LOG_DEFAULT,
            "export_html": cfg.compile.export_html,
        }
        try:
            params = inspect.signature(fn).parameters
            accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
            kwargs = candidates if accepts_kwargs else {
                k: v for k, v in candidates.items() if k in params
            }
        except (TypeError, ValueError):
            kwargs = candidates
        fn(**kwargs)

    # --------------------------------------------------------------------- log
    def parse_log(self, text: str) -> dict[str, Any]:
        text = _RE_ANSI.sub("", text)
        final = _RE_FINAL.findall(text)
        if final:
            npu = [int(final[-1][0])]
            cpu = [int(final[-1][1])]
        else:
            npu = [int(m.group(1)) for m in _RE_NPU.finditer(text)]
            cpu = [int(m.group(1)) for m in _RE_CPU.finditer(text)]
        lines = [ln.strip() for ln in text.splitlines()]
        warnings = [ln for ln in lines if _RE_WARN.search(ln)]
        # [WARNING] 줄은 문장에 'error' 가 들어 있어도 오류로 보지 않는다
        # (예: "... cosine similarity ... Skipping error.")
        errors = [ln for ln in lines if _RE_ERROR.search(ln) and not _RE_WARN_LEVEL.search(ln)]
        return {
            "npu_partitions": npu[-1] if npu else None,
            "cpu_partitions": cpu[-1] if cpu else None,
            "warnings": warnings[:50],
            "errors": errors[:50],
            "parser": "v2_5/final_result" if final else "v2_5/fallback (미검증 패턴)",
        }
