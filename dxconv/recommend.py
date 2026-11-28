"""추천 설정 (configs/recommend.yaml): dxnn 포맷별로 한 번에 컴파일할 값.

- `dxconv convert` 는 이 값으로 한 번만 컴파일한다 (추천 모드). 병합 순서는 config.merge_layers 참고.
- `dxconv recommend update <sweep 폴더...>`: sweep 선택 결과를 모아 포맷별 추천값을 다시 정한다.
  키마다 여러 sweep 에서 가장 많이 선택된 값(동률이면 마지막 sweep 값)을 쓰고, 근거(evidence)를 남긴다.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dxconv import DxconvError
from dxconv.config import load_yaml, recommend_format
from dxconv.sweep.space import STAGE_ORDER, to_nested

HEADER = """# 추천 설정 (dxnn 포맷별). 실측(sweep + NPU 평가)으로 정한 값.
# 설정 병합 순서: defaults.yaml < recommend.yaml(목표 포맷) < presets/<model>.yaml < --config < --set < CLI 옵션
# `dxconv convert` 는 이 값으로 한 번만 컴파일한다. `dxconv sweep` (설정 비교 실험) 은 이 값의 재검증용.
# 갱신: dxconv recommend update <sweep 결과 폴더...>  /  확인: dxconv recommend show
"""


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def load(config_dir: Path) -> dict[str, Any]:
    path = config_dir / "recommend.yaml"
    return (load_yaml(path) if path.is_file() else {}) or {}


def sweep_result(root: Path, config_dir: Path) -> dict[str, Any]:
    """sweep 폴더 하나의 선택 결과 → {format, params, final, fallback, ...}."""
    from dxconv.sweep.store import SweepStore

    if not (root / "sweep.db").is_file():
        raise DxconvError(f"sweep DB 가 없습니다: {root / 'sweep.db'}")
    store = SweepStore(root)
    try:
        params: dict[str, Any] = {}
        fallback = False
        version = None
        for st in STAGE_ORDER:
            sel = store.get_selection(st)
            if sel is None:
                continue
            if sel["status"] != "selected":
                raise DxconvError(f"{root}: {st} 단계가 선택되지 않았습니다 ({sel['status']}).")
            row = next(r for r in store.stage_rows(st) if r["hash"] == sel["hash"])
            params.update(row["params"])
            fallback = fallback or (sel.get("reason") or "").startswith("[대체 선택]")
            version = ((row.get("config") or {}).get("compile") or {}).get("dxcom_version") or version
        if not params:
            raise DxconvError(f"{root}: 선택된 단계가 없습니다.")
        fmt = recommend_format(config_dir, version)
        return {"root": str(root), "format": fmt, "dxcom_version": version, "params": params,
                "fallback": fallback, "final": store.get_meta("final_eval"),
                "device": (store.get_meta("device") or {}).get("hostname")}
    finally:
        store.close()


def update(roots: list[Path], config_dir: Path) -> dict[int, dict[str, Any]]:
    """sweep 결과들로 포맷별 추천값을 갱신해 recommend.yaml 에 쓴다 (.bak 백업)."""
    results = [sweep_result(Path(r), config_dir) for r in roots]
    doc = load(config_dir)
    formats = doc.setdefault("formats", {})
    changed: dict[int, dict[str, Any]] = {}
    for fmt in sorted({r["format"] for r in results if r["format"] is not None}):
        rs = [r for r in results if r["format"] == fmt]
        entry = formats.setdefault(fmt, {})
        current = _flatten(entry.get("settings") or {})
        votes: dict[str, Counter[Any]] = {}
        for r in rs:
            for k, v in r["params"].items():
                if k == "compile.calib_repeat":  # 사진 묶음 번호는 모델·데이터마다 달라 추천값에 넣지 않음
                    continue
                votes.setdefault(k, Counter())[json.dumps(v)] += 1
        new = dict(current)
        for k, cnt in votes.items():
            top = max(cnt.values())
            winners = [v for v, n in cnt.items() if n == top]
            last = json.dumps(next(r["params"][k] for r in reversed(rs) if k in r["params"]))
            new[k] = json.loads(last if last in winners else winners[0])
        entry["settings"] = to_nested(new)
        ev = entry.setdefault("evidence", [])
        for r in rs:
            fin = r["final"] or {}
            ev.append(
                f"{datetime.now():%Y-%m-%d} sweep {Path(r['root']).name} ({r['dxcom_version']}, 장비 {r['device']}): "
                + ", ".join(f"{k}={v}" for k, v in r["params"].items())
                + (f" / 최종 Δ mAP50-95 {fin['delta_map50_95_pp']:+.2f}%p, {fin.get('fps') or 0:.1f} FPS"
                   if fin.get("delta_map50_95_pp") is not None else "")
                + (" [대체 선택: --objective 조건을 만족하는 조합 없음]" if r["fallback"] else ""))
        changed[fmt] = {k: (current.get(k), v) for k, v in new.items() if current.get(k) != v}
    path = config_dir / "recommend.yaml"
    if path.is_file():
        path.with_suffix(".yaml.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=200),
                    encoding="utf-8")
    return changed
