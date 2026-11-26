"""sweep 탐색 공간 로드 · 조합 생성.

configs/sweep/default_space.yaml:
    stages:
      A: {export.opset: [...], export.simplify: [...]}
      B: {compile.calibration_method: [...], compile.calibration_num: [...]}
      C: {compile.opt_level: [...], compile.aggressive_partitioning: [...]}
    objective: "PT 대비 mAP 하락 <= 1%p 중 FPS 최대"
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from dxconv.config import ConfigError, load_yaml

STAGE_ORDER = ("A", "B", "C")


def load_space(path: Path) -> dict[str, Any]:
    """{"stages": {"A": {key: [values]}, ...}, "objective": str | None}"""
    raw = load_yaml(path)
    stages = raw.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ConfigError(f"탐색 공간에 stages 가 없습니다: {path}")
    unknown = [s for s in stages if s not in STAGE_ORDER]
    if unknown:
        raise ConfigError(f"알 수 없는 단계: {unknown} (A | B | C)")
    out: dict[str, dict[str, list[Any]]] = {}
    for st in STAGE_ORDER:
        if st not in stages:
            continue
        sp = stages[st] or {}
        if not isinstance(sp, dict) or not sp:
            raise ConfigError(f"{st} 단계 탐색 공간이 비어 있습니다: {path}")
        for k, v in sp.items():
            if not isinstance(v, list) or not v:
                raise ConfigError(f"{st}.{k}: 후보값 목록(list)이 필요합니다.")
            if "." not in k:
                raise ConfigError(f"{st}.{k}: 'section.key' 형식이어야 합니다 (예: export.opset).")
        out[st] = {k: list(v) for k, v in sp.items()}
    return {"stages": out, "objective": raw.get("objective")}


def iter_combinations(stage_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """{key: [v1, v2]} → [{key: v1}, {key: v2}, ...] (키 순서대로 곱집합)."""
    keys = list(stage_space)
    return [dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(stage_space[k] for k in keys))]


def to_nested(flat: dict[str, Any]) -> dict[str, Any]:
    """{"compile.opt_level": 1} → {"compile": {"opt_level": 1}}"""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return out


def get_dotted(d: dict[str, Any], key: str) -> Any:
    cur: Any = d
    for p in key.split("."):
        cur = cur[p]
    return cur
