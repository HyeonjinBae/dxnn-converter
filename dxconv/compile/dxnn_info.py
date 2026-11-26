"""컴파일된 .dxnn 에 실제로 기록된 설정 읽기 · 요청 설정과 비교.

배경: DX-COM 2.4.1 CLI 는 config 에 minmax/50 을 요청해도 dxnn 에 ema/100 을 기록했다.
그래서 컴파일할 때마다 실제 설정을 확인한다.

dxnn 구조 (실측):
- v9 (DX-COM 2.5.0): b"DXNN" + 헤더 JSON {"schema_version", "version": 9, "payload_offset", "data": {...}}
- v8 (DX-COM 2.4.1): b"DXNN" + 헤더 JSON {"version": 8, "size": <헤더 영역 크기>, "data": {...}}
- data 의 각 섹션은 {"type", "offset", "size"} (offset 은 payload 시작 기준; v8 은 헤더 영역 크기 기준)
- compile_config 섹션: DX-COM 이 읽은 config (v9: calibration.{num, method}, v8: calibration_num/method)
- graph_info 섹션: NPU/CPU 그래프 분할 (각 그래프의 입력/출력 텐서)

기록되지 않는 값:
- opt_level: dxnn 에 없음 → DX-COM 로그의 "Optimization Level : N" 으로 확인
- aggressive_partitioning: dxnn·로그 어디에도 값이 없음 → 실제 값은 확인 불가.
  대신 그래프 분할 결과(CPU 그래프 입력 텐서)를 기록한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_RE_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_RE_OPT = re.compile(r"Optimization Level\s*:\s*(\d+)")


class DxnnFormatError(ValueError):
    """dxnn 헤더를 해석할 수 없음."""


@dataclass
class DxnnInfo:
    format_version: int | None
    compile_version: str | None
    calibration_method: str | None
    calibration_num: int | None
    input_shapes: dict[str, Any] | None
    preprocessings: list[str] = field(default_factory=list)  # 전처리 op 이름 순서
    dataset_path: str | None = None
    graphs: list[dict[str, Any]] = field(default_factory=list)  # [{name, device, inputs}]
    compile_config: dict[str, Any] = field(default_factory=dict)

    @property
    def npu_graphs(self) -> int:
        return sum(1 for g in self.graphs if g["device"] == "NPU")

    @property
    def cpu_graphs(self) -> int:
        return sum(1 for g in self.graphs if g["device"] == "CPU")

    @property
    def cpu_inputs(self) -> list[str]:
        return sorted(i for g in self.graphs if g["device"] == "CPU" for i in g["inputs"])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("compile_config")
        d.update(npu_graphs=self.npu_graphs, cpu_graphs=self.cpu_graphs)
        return d


def _header(raw: bytes) -> tuple[dict[str, Any], int]:
    if not raw.startswith(b"DXNN"):
        raise DxnnFormatError("DXNN 시그니처가 없습니다.")
    # 시그니처 뒤 고정 필드(버전, 크기, 체크섬 등) 다음에 JSON 헤더가 온다. 체크섬 바이트에 우연히
    # '{'(0x7B)가 들어갈 수 있으므로 (약 1.6%), '{' 위치마다 파싱해서 data 가 있는 첫 dict 를 쓴다.
    hdr: Any = None
    last_err: Exception | None = None
    pos = 4
    while hdr is None:
        start = raw.find(b"{", pos, 4096)
        if start < 0:
            break
        try:
            cand, _ = json.JSONDecoder().raw_decode(raw[start:start + (1 << 20)].decode("latin-1"))
            if isinstance(cand, dict) and "data" in cand:
                hdr = cand
        except ValueError as e:
            last_err = e
        pos = start + 1
    if hdr is None:
        raise DxnnFormatError("헤더 JSON 을 읽을 수 없습니다"
                              + (f": {last_err}" if last_err else " (data 섹션 목록 없음)"))
    if "payload_offset" in hdr:
        base = int(hdr["payload_offset"])
    elif "size" in hdr:
        base = int(hdr["size"])
    else:
        raise DxnnFormatError("payload 시작 위치(payload_offset/size)를 알 수 없습니다.")
    return hdr, base


def read_dxnn(path: str | Path) -> DxnnInfo:
    raw = Path(path).read_bytes()
    hdr, base = _header(raw)

    def section(name: str) -> bytes | None:
        node = hdr["data"].get(name)
        if not isinstance(node, dict) or "offset" not in node:
            return None
        off, size = base + int(node["offset"]), int(node["size"])
        if off + size > len(raw):
            raise DxnnFormatError(f"섹션 {name} 이 파일 범위를 벗어납니다.")
        return raw[off:off + size]

    def as_json(name: str) -> dict[str, Any]:
        b = section(name)
        if not b:
            return {}
        try:
            return json.loads(b.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise DxnnFormatError(f"섹션 {name} 이 JSON 이 아닙니다 (암호화 등): {e}") from e

    cc = as_json("compile_config")
    gi = as_json("graph_info")
    cal = cc.get("calibration") if isinstance(cc.get("calibration"), dict) else {}
    method = cal.get("method", cc.get("calibration_method"))
    num = cal.get("num", cc.get("calibration_num"))
    graphs = [{"name": g.get("name"), "device": g.get("device"),
               "inputs": [i.get("name") for i in g.get("inputs", [])]}
              for g in gi.get("graphs", [])]
    return DxnnInfo(
        format_version=hdr.get("version"),
        compile_version=cc.get("compile_version"),
        calibration_method=str(method).lower() if method is not None else None,
        calibration_num=int(num) if num is not None else None,
        input_shapes=cc.get("input_shapes"),
        preprocessings=[next(iter(p)) for p in cc.get("preprocessings", []) if p],
        dataset_path=cc.get("dataset_path"),
        graphs=graphs,
        compile_config=cc,
    )


def opt_level_from_log(text: str) -> int | None:
    m = _RE_OPT.findall(_RE_ANSI.sub("", text))
    return int(m[-1]) if m else None


def compare_settings(
    requested: dict[str, Any], info: DxnnInfo | None, log_text: str = "",
    calib_images: int | None = None,
) -> dict[str, Any]:
    """요청 설정과 dxnn/로그의 실제 설정 비교.

    requested: {calibration_method, calibration_num, opt_level, aggressive_partitioning,
                preprocessings(op 이름 목록), input_shapes}
    반환: {"requested", "actual", "mismatches": [...], "notes": [...]}
    """
    actual: dict[str, Any] = {
        "calibration_method": info.calibration_method if info else None,
        "calibration_num": info.calibration_num if info else None,
        "opt_level": opt_level_from_log(log_text),
        "aggressive_partitioning": None,  # dxnn·로그에 기록되지 않음
        "preprocessings": info.preprocessings if info else None,
        "input_shapes": info.input_shapes if info else None,
        "compile_version": info.compile_version if info else None,
        "dxnn_format": info.format_version if info else None,
        "partition": ({"npu_graphs": info.npu_graphs, "cpu_graphs": info.cpu_graphs,
                       "cpu_inputs": info.cpu_inputs} if info else None),
    }
    mismatches: list[str] = []
    notes: list[str] = []
    for key in ("calibration_method", "calibration_num", "opt_level", "preprocessings",
                "input_shapes"):
        req, act = requested.get(key), actual.get(key)
        if req is None:
            continue
        if act is None:
            notes.append(f"{key}: 실제 값을 확인할 수 없음 (요청 {req!r})")
        elif req != act:
            mismatches.append(f"{key}: 요청 {req!r} ≠ 실제 {act!r}")
    if requested.get("aggressive_partitioning") is not None:
        notes.append("aggressive_partitioning: dxnn·로그에 값이 기록되지 않아 실제 값은 확인 불가 "
                     "(CLI 플래그 전달 여부와 partition.cpu_inputs 로 판단)")
    num = actual["calibration_num"]
    if num is not None and calib_images is not None and calib_images < num:
        mismatches.append(f"calibration_num: dxnn 기록 {num} 이지만 캘리브레이션 이미지는 "
                          f"{calib_images} 장뿐 (실제 사용은 최대 {calib_images} 장)")
    return {"requested": requested, "actual": actual, "mismatches": mismatches, "notes": notes}
