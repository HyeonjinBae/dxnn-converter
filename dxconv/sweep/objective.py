"""sweep 목적식 · 단계별 선택 규칙.

단계별 판정:
- A: 로컬 검증만 (컴파일 성공 + dxnn 설정 일치 + FP32 ONNX 가 PT 와 같음).
     통과 조합이 여럿이면 기본 설정 값과 같은 조합을 우선, 없으면 공간 순서상 첫 조합.
- B: Tuning mAP50-95 최대 (NPU 평가 필요).
- C: 목적식(objective)의 조건을 만족하는 조합 중 목표 지표 최대 (NPU 평가 필요).
평가가 필요한 단계에서 유효 조합 중 하나라도 평가 결과가 없으면 선택하지 않는다 (pending).

목적식 형식 (두 가지):
    "PT 대비 mAP 하락 <= 1%p 중 FPS 최대"        조건: 하락 ≤ X%p      → FPS 최대
    "FPS >= 20*15 중 mAP 최대"                  조건: FPS ≥ N (곱셈 가능) → mAP 최대
    "PT 대비 mAP50 하락 <= 1%p, FPS >= 300 중 mAP50 최대"   (조건 여러 개)
- mAP 는 기본 mAP50-95, "mAP50" 이라고 쓰면 mAP50 (하락 조건·최대 지표 각각).
- 해석할 수 없는 문자열은 에러 (조용히 다른 기준으로 선택하지 않는다).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from dxconv import DxconvError

DEFAULT_OBJECTIVE = "PT 대비 mAP 하락 <= 1%p 중 FPS 최대"
# 대체 선택에서 같은 정확도로 보는 폭 (%p). 캘리브레이션 샘플만 바꿨을 때 편차 0.26%p 보다 작게.
FALLBACK_TOL_PP = 0.1

_NUM = r"\d+(?:\.\d+)?"
_RE_DROP = re.compile(rf"(mAP(?:50-95|50)?)\s*하락\s*(?:<=|≤|이하)\s*({_NUM})\s*%p", re.I)
_RE_FPS = re.compile(rf"FPS\s*(?:>=|≥)\s*({_NUM}(?:\s*[*×x]\s*{_NUM})*)", re.I)
_RE_MAX = re.compile(r"중\s*(FPS|mAP(?:50-95|50)?)\s*최대\s*$", re.I)


LABEL = {"tuning_map50_95": "mAP50-95", "tuning_map50": "mAP50", "fps": "FPS"}


def _metric_key(word: str) -> str:
    """'mAP' / 'mAP50-95' → tuning_map50_95, 'mAP50' → tuning_map50."""
    return "tuning_map50" if word.lower() == "map50" else "tuning_map50_95"


@dataclass
class Objective:
    text: str
    max_drop_pp: float | None = 1.0   # PT 대비 하락 허용치 (%p). None = 조건 없음
    drop_metric: str = "tuning_map50_95"
    min_fps: float | None = None      # FPS 하한. None = 조건 없음
    maximize: str = "fps"             # fps | tuning_map50_95 | tuning_map50
    # 조건을 만족하는 조합이 없을 때: False(기본) → 하락이 가장 작은(mAP 최대) 조합으로 대체 선택,
    # True(--strict-objective) → 선택하지 않고 멈춤
    strict: bool = False

    def describe(self) -> str:
        cond = []
        if self.max_drop_pp is not None:
            cond.append(f"PT 대비 {LABEL[self.drop_metric]} 하락 <= {self.max_drop_pp:g}%p")
        if self.min_fps is not None:
            cond.append(f"FPS >= {self.min_fps:g}")
        return f"{', '.join(cond) or '조건 없음'} 중 {LABEL[self.maximize]} 최대"


def parse_objective(text: str | None, strict: bool = False) -> Objective:
    text = (text or DEFAULT_OBJECTIVE).strip()
    mmax = _RE_MAX.search(text)
    if not mmax:
        raise DxconvError(f'--objective 를 해석할 수 없습니다: "{text}" '
                          '(형식: "<조건> 중 FPS|mAP|mAP50 최대", 예: "PT 대비 mAP 하락 <= 1%p 중 FPS 최대")')
    head = text[:mmax.start()]
    mdrop, mfps = _RE_DROP.search(head), _RE_FPS.search(head)
    if not mdrop and not mfps:
        raise DxconvError(f'--objective 에 조건이 없습니다: "{text}" (예: "mAP 하락 <= 1%p", "FPS >= 20*15")')
    min_fps = None
    if mfps:
        min_fps = 1.0
        for n in re.split(r"[*×x]", mfps.group(1)):
            min_fps *= float(n)
    word = mmax.group(1)
    return Objective(
        text=text,
        max_drop_pp=float(mdrop.group(2)) if mdrop else None,
        drop_metric=_metric_key(mdrop.group(1)) if mdrop else "tuning_map50_95",
        min_fps=min_fps,
        maximize="fps" if word.lower() == "fps" else _metric_key(word),
        strict=strict,
    )


@dataclass
class Selection:
    status: str  # selected | pending | none
    hash: str | None = None
    reason: str = ""
    missing: list[str] | None = None  # 평가가 없는 조합 hash
    fallback: bool = False            # 목적식 조건 미충족 → 대체 선택


def select_stage(stage: str, rows: list[dict[str, Any]], objective: Objective,
                 base_params: dict[str, Any], pt: dict[str, Any] | float | None) -> Selection:
    """rows: 이 단계 조합들 (params, status, fp32_ok, tuning_map50(_95), fps).

    pt: PT 기준선 {"map50": .., "map50_95": ..} (float 이면 mAP50-95).
    """
    if isinstance(pt, (int, float)):
        pt = {"map50_95": float(pt)}
    pt = pt or {}
    valid = [r for r in rows if r["status"] == "compiled"]
    if stage == "A":
        valid = [r for r in valid if r.get("fp32_ok")]
        if not valid:
            return Selection("none", reason="통과한 조합 없음 (컴파일/설정/FP32 검증)")
        for r in valid:
            if all(base_params.get(k) == v for k, v in r["params"].items()):
                return Selection("selected", r["hash"], "통과 조합 중 기본 설정과 같은 조합")
        return Selection("selected", valid[0]["hash"], "통과 조합 중 탐색 공간 순서상 첫 조합")
    if not valid:
        return Selection("none", reason="유효한 조합 없음 (컴파일 실패 또는 설정 불일치)")
    if stage == "B":
        need = ["tuning_map50_95"]
    else:
        need = sorted({objective.maximize, "fps", "tuning_map50_95"}
                      | ({objective.drop_metric} if objective.max_drop_pp is not None else set()))
    missing = [r["hash"] for r in valid if any(r.get(k) is None for k in need)]
    if missing:
        what = " / ".join(k.removeprefix("tuning_") for k in need)
        return Selection("pending", reason=f"{len(missing)}개 조합의 평가 결과({what}) 없음",
                         missing=missing)
    if stage == "B":
        return _select_calibration(valid)

    ok = valid
    if objective.max_drop_pp is not None:
        pt_key = objective.drop_metric.removeprefix("tuning_")
        if pt.get(pt_key) is None:
            return Selection("pending", reason=f"PT 기준선(Tuning {pt_key}) 없음 → 하락폭 계산 불가")
        ok = [r for r in ok
              if (r[objective.drop_metric] - pt[pt_key]) * 100 >= -objective.max_drop_pp - 1e-9]
    if objective.min_fps is not None:
        ok = [r for r in ok if r["fps"] >= objective.min_fps]
    if not ok:
        if objective.strict:
            return Selection("none", reason=f"조건을 만족하는 조합 없음 ({objective.describe()})")
        metric = objective.drop_metric if objective.max_drop_pp is not None else "tuning_map50_95"
        top = max(r[metric] for r in valid)
        # 최고 mAP 에서 FALLBACK_TOL_PP 이내는 같은 정확도로 보고 그중 목표 지표(FPS 등) 최대
        near = [r for r in valid if (top - r[metric]) * 100 <= FALLBACK_TOL_PP + 1e-9]
        best = max(near, key=lambda r: (r[objective.maximize] or 0, r[metric]))
        return Selection("selected", best["hash"],
                         f"[대체 선택] 조건을 만족하는 조합 없음 ({objective.describe()}) → "
                         f"{LABEL[metric]} 최대(하락 최소)에서 {FALLBACK_TOL_PP:g}%p 이내 중 "
                         f"{LABEL[objective.maximize]} 최대", fallback=True)
    best = max(ok, key=lambda r: r[objective.maximize])
    return Selection("selected", best["hash"], objective.describe())


REPEAT_KEY = "compile.calib_repeat"


def _select_calibration(valid: list[dict[str, Any]]) -> Selection:
    """B 단계: 사진 묶음(calib_repeat)이 여러 개면 묶음을 뺀 설정별로 Tuning mAP50-95 평균을 내서 고르고,
    고른 설정 안에서 Tuning 이 가장 높은 묶음의 dxnn 을 쓴다. 묶음이 없으면 Tuning 최대."""
    if not any(REPEAT_KEY in r["params"] for r in valid):
        best = max(valid, key=lambda r: r["tuning_map50_95"])
        return Selection("selected", best["hash"], "Tuning mAP50-95 최대")
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in valid:
        g = {k: v for k, v in r["params"].items() if k != REPEAT_KEY}
        groups.setdefault(json.dumps(g, sort_keys=True), []).append(r)
    means = {g: sum(r["tuning_map50_95"] for r in rs) / len(rs) for g, rs in groups.items()}
    top = max(means, key=lambda g: means[g])
    best = max(groups[top], key=lambda r: r["tuning_map50_95"])

    def label(g: str) -> str:
        d = json.loads(g)
        return "·".join(str(v) for v in d.values()) or "기본"

    summary = ", ".join(f"{label(g)} {means[g] * 100:.2f} ({len(groups[g])}묶음)"
                        for g in sorted(means, key=lambda g: -means[g]))
    return Selection("selected", best["hash"],
                     f"사진 묶음 평균 Tuning mAP50-95 최대: {summary} → {label(top)}, "
                     f"그중 Tuning 최대 묶음 {best['params'].get(REPEAT_KEY)}")
