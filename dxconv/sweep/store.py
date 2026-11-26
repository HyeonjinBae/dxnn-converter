"""sweep 결과 저장 (sqlite) · summary.md · 프리셋 저장.

위치: <out>/sweep/<model>/
    sweep.db            결과 DB (중단 후 재시작 시 이어서 사용)
    summary.md          단계별 표 (평가 칸은 import-eval 로 채움)
    pending_eval.json   평가가 필요한 조합 목록 (import-eval 입력 템플릿)
    combos/<hash>/      조합별 dxconv convert run 폴더

테이블:
    combos(hash PK, ...)            조합 결과 (컴파일·검증·평가). 단계와 무관 (캐시 단위)
    members(stage, hash, params)    단계별 조합 목록
    selections(stage PK, ...)       단계별 선택 결과
    meta(key PK, value)             PT 기준선 등
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SCHEMA = """
CREATE TABLE IF NOT EXISTS combos (
    hash TEXT PRIMARY KEY,
    config TEXT NOT NULL,          -- 이 조합의 전체 설정 (export/compile)
    status TEXT NOT NULL,          -- running | compiled | failed | excluded
    run_dir TEXT, dxnn_path TEXT, dxnn_sha256 TEXT, dxnn_size INTEGER,
    compile_sec REAL, npu_groups INTEGER, cpu_groups INTEGER,
    settings TEXT,                 -- dxnn_settings (requested/actual/mismatches/notes)
    settings_ok INTEGER,
    fp32 TEXT, fp32_ok INTEGER,    -- A 단계: FP32 ONNX vs PT 비교
    onnx_exporter TEXT,
    warnings TEXT, error TEXT,
    tuning_map50 REAL, tuning_map50_95 REAL, delta_pp REAL, fps REAL,
    eval_source TEXT, eval_at TEXT,
    npu TEXT,                      -- 원격 NPU 평가 상세 (장비, FPS sync/max, P/R, OpenCV)
    created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS members (
    stage TEXT NOT NULL, hash TEXT NOT NULL, params TEXT NOT NULL, fixed TEXT NOT NULL,
    ord INTEGER NOT NULL, PRIMARY KEY (stage, hash)
);
CREATE TABLE IF NOT EXISTS selections (
    stage TEXT PRIMARY KEY, status TEXT NOT NULL, hash TEXT, reason TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

JSON_COLS = ("config", "settings", "fp32", "onnx_exporter", "warnings", "npu")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SweepStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "sweep.db"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(combos)").fetchall()}
        if "npu" not in cols:  # 이전 DB 이관
            self.conn.execute("ALTER TABLE combos ADD COLUMN npu TEXT")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- combos
    def get(self, h: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM combos WHERE hash=?", (h,)).fetchone()
        return self._decode(row) if row else None

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for k in JSON_COLS:
            if d.get(k) is not None:
                d[k] = json.loads(d[k])
        for k in ("settings_ok", "fp32_ok"):
            if d.get(k) is not None:
                d[k] = bool(d[k])
        return d

    def upsert(self, h: str, **fields: Any) -> None:
        fields = {k: (json.dumps(v, ensure_ascii=False, default=str) if k in JSON_COLS and v is not None
                      else (int(v) if isinstance(v, bool) else v)) for k, v in fields.items()}
        fields["updated_at"] = _now()
        if self.conn.execute("SELECT 1 FROM combos WHERE hash=?", (h,)).fetchone():
            sets = ", ".join(f"{k}=?" for k in fields)
            self.conn.execute(f"UPDATE combos SET {sets} WHERE hash=?", (*fields.values(), h))
        else:
            fields.setdefault("created_at", fields["updated_at"])
            cols = ", ".join(["hash", *fields])
            qs = ", ".join("?" * (len(fields) + 1))
            self.conn.execute(f"INSERT INTO combos ({cols}) VALUES ({qs})", (h, *fields.values()))
        self.conn.commit()

    def find(self, key: str) -> list[dict[str, Any]]:
        """hash(앞부분 8자 이상) 또는 dxnn_sha256 으로 조합 찾기."""
        rows = self.conn.execute(
            "SELECT * FROM combos WHERE hash LIKE ? OR dxnn_sha256=?", (f"{key}%", key)).fetchall()
        return [self._decode(r) for r in rows]

    # ------------------------------------------------------------ members
    def add_member(self, stage: str, h: str, params: dict[str, Any], fixed: dict[str, Any],
                   ord_: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO members (stage, hash, params, fixed, ord) VALUES (?,?,?,?,?)",
            (stage, h, json.dumps(params), json.dumps(fixed), ord_))
        self.conn.commit()

    def clear_members(self, stage: str) -> None:
        self.conn.execute("DELETE FROM members WHERE stage=?", (stage,))
        self.conn.commit()

    def stage_rows(self, stage: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT m.params, m.fixed, m.ord, c.* FROM members m JOIN combos c ON c.hash=m.hash "
            "WHERE m.stage=? ORDER BY m.ord", (stage,)).fetchall()
        out = []
        for r in rows:
            d = self._decode(r)
            d["params"], d["fixed"] = json.loads(d["params"]), json.loads(d["fixed"])
            out.append(d)
        return out

    def stages(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT stage FROM members ORDER BY stage").fetchall()]

    # --------------------------------------------------------- selections
    def set_selection(self, stage: str, status: str, h: str | None, reason: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO selections (stage, status, hash, reason, updated_at) "
            "VALUES (?,?,?,?,?)", (stage, status, h, reason, _now()))
        self.conn.commit()

    def get_selection(self, stage: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM selections WHERE stage=?", (stage,)).fetchone()
        return dict(row) if row else None

    def clear_selections_from(self, stages: list[str]) -> None:
        self.conn.executemany("DELETE FROM selections WHERE stage=?", [(s,) for s in stages])
        self.conn.commit()

    # --------------------------------------------------------------- meta
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                          (key, json.dumps(value, ensure_ascii=False)))
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default


# --------------------------------------------------------------------------- #
# summary.md
# --------------------------------------------------------------------------- #
def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v * 100:.2f}" if pct else f"{v:.2f}"
    return str(v)


def _fixed_line(fixed: dict[str, Any]) -> str:
    vals = {k: v for k, v in fixed.items() if k != "_source"}
    if not vals:
        return "- 고정값: 없음 (첫 단계)"
    src = fixed.get("_source") or {}
    srcs = ", ".join(f"{st}: {'기본 설정 (선택 전)' if v == 'base' else v.replace('selected:', '선택 ')}"
                     for st, v in src.items())
    return f"- 고정값: `{json.dumps(vals, ensure_ascii=False)}`" + (f" (출처 {srcs})" if srcs else "")


def write_summary(store: SweepStore, model: str, stage_order: tuple[str, ...]) -> Path:
    pt = store.get_meta("pt_baseline") or {}
    lines = [f"# sweep 요약: {model}", "",
             f"- 갱신: {_now()}  /  DB: `{store.db_path}`",
             f"- PT 기준선 Tuning mAP50-95: {_fmt(pt.get('map50_95'), True) or '(없음)'}",
             "- 평가 칸(Tuning mAP, Δ vs PT, FPS)은 `--remote` 평가 또는 `dxconv sweep import-eval` 로 채운다.", ""]
    dev = store.get_meta("device")
    if dev:
        lines[-1:-1] = [f"- NPU 장비: {dev.get('host')} ({dev.get('hostname')}), DX-RT {dev.get('dxrt')}, "
                        f"드라이버 {dev.get('rt_driver')}, FW {dev.get('firmware')}, OpenCV {dev.get('opencv')}. "
                        "FPS = run_model 최대 처리량"]
    fin = store.get_meta("final_eval")
    if fin:
        lines += ["## 최종 평가 (evaluation 분할, 선택에 쓰지 않음)", "",
                  f"- 조합 `{fin['hash']}`: `{json.dumps(fin['params'], ensure_ascii=False)}`",
                  f"- dxnn: `{fin['dxnn_path']}`", "",
                  "| | 이미지 | mAP50 | mAP50-95 | P@conf | R@conf | FPS |", "|---|---|---|---|---|---|---|",
                  f"| DXNN (NPU) | {fin['images']} | {fin['map50'] * 100:.2f} | "
                  f"{fin['map50_95'] * 100:.2f} | {fin['pr_at_conf']['precision']:.3f} | "
                  f"{fin['pr_at_conf']['recall']:.3f} | {_fmt(fin.get('fps'))} |",
                  f"| PT | {fin['images']} | {fin['pt']['map50'] * 100:.2f} | {fin['pt']['map50_95'] * 100:.2f} | "
                  f"{fin['pt']['pr_at_conf']['precision']:.3f} | {fin['pt']['pr_at_conf']['recall']:.3f} | |",
                  f"| Δ (%p) | | {fin['delta_map50_pp']:+.2f} | {fin['delta_map50_95_pp']:+.2f} | | | |", ""]
    for st in stage_order:
        rows = store.stage_rows(st)
        if not rows:
            continue
        sel = store.get_selection(st) or {}
        lines += [f"## {st} 단계", "",
                  f"- 선택: **{sel.get('status', '-')}**"
                  + (f" → `{sel['hash']}`" if sel.get("hash") else "")
                  + (f" ({sel['reason']})" if sel.get("reason") else ""),
                  _fixed_line(rows[0]["fixed"]), ""]
        keys = list(rows[0]["params"])
        head = (["hash", *keys, "상태", "설정 검증", "실제 method/num/opt", "agg 플래그", "CPU",
                 "dxnn bytes", "컴파일 s"] + (["FP32=PT"] if st == "A" else [])
                + ["Tuning mAP50", "Tuning mAP50-95", "Δ vs PT (%p)", "FPS", ""])
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in rows:
            act = (r.get("settings") or {}).get("actual") or {}
            mism = (r.get("settings") or {}).get("mismatches") or []
            check = ("—" if r["status"] in ("running", "failed") else
                     "일치" if r.get("settings_ok") else f"불일치 {len(mism)}")
            fp32 = []
            if st == "A":
                f = r.get("fp32") or {}
                fp32 = ["" if r.get("fp32_ok") is None else
                        ("예" if r["fp32_ok"] else f"아니오 ({f.get('reason', '')})")]
            mark = "✅ 선택" if sel.get("hash") == r["hash"] else ""
            cells = ([f"`{r['hash']}`", *[str(r["params"][k]) for k in keys], r["status"], check,
                      f"{act.get('calibration_method')}/{act.get('calibration_num')}/{act.get('opt_level')}",
                      _fmt(act.get("aggressive_flag_in_command")), _fmt(r.get("cpu_groups")),
                      _fmt(r.get("dxnn_size")), _fmt(r.get("compile_sec"))] + fp32
                     + [_fmt(r.get("tuning_map50"), True), _fmt(r.get("tuning_map50_95"), True),
                        _fmt(r.get("delta_pp")), _fmt(r.get("fps")), mark])
            lines.append("| " + " | ".join(cells) + " |")
        bad = [r for r in rows if r["status"] in ("excluded", "failed")]
        if bad:
            lines += ["", "제외/실패:"]
            for r in bad:
                why = r.get("error") or "; ".join((r.get("settings") or {}).get("mismatches") or [])
                lines.append(f"- `{r['hash']}` {r['status']}: {why}")
        lines.append("")
    path = store.root / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_pending_eval(store: SweepStore, stage: str, hashes: list[str]) -> Path:
    """평가가 필요한 조합 목록 (import-eval 입력 템플릿)."""
    items = []
    for h in hashes:
        r = store.get(h) or {}
        items.append({"hash": h, "dxnn_path": r.get("dxnn_path"), "dxnn_sha256": r.get("dxnn_sha256"),
                      "map50": None, "map50_95": None, "fps": None})
    doc = {"stage": stage, "split": "tuning",
           "pt": {"map50": None, "map50_95": None},
           "note": "NPU 에서 Tuning 200 평가 후 값을 채워 `dxconv sweep import-eval` 로 넣는다. "
                   "pt 는 PT 기준선 (dxconv baseline 의 tuning PT 값). 모르면 null.",
           "results": items}
    path = store.root / "pending_eval.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 프리셋
# --------------------------------------------------------------------------- #
def save_preset(model: str, best: dict[str, Any], config_dir: Path, note: str = "",
                path: Path | None = None) -> Path:
    """선택된 파라미터({"compile.opt_level": 1, ...})를 presets/<model>.yaml 에 병합 (.bak 백업)."""
    from dxconv.config import deep_merge, load_yaml
    from dxconv.sweep.space import to_nested

    path = path or config_dir / "presets" / f"{model}.yaml"
    old = load_yaml(path) if path.exists() else {}
    comments: list[str] = []
    if path.exists():
        shutil.copy2(path, path.with_suffix(".yaml.bak"))
        # 기존 주석(예: 기존 변환의 calibration_num 기록)은 유지
        comments = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.startswith("#")]
    merged = deep_merge(old, to_nested(best))
    header = "\n".join([*comments, f"# sweep 결과 반영: {_now()}"]
                       + ([f"# {note}"] if note else [])) + "\n"
    path.write_text(header + yaml.safe_dump(merged, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path
