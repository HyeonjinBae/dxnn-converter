"""이미지/라벨 폴더 → 캘리브레이션 / 튜닝 / 평가 자동 분할.

배포에서는 사용자가 이미지 폴더와 (YOLO) 라벨 폴더만 지정한다. 이미지 수가 매번 다르므로
비율(SplitCfg)로 나누고, 결과를 데이터셋 설정(dataset.yaml + 목록 파일)으로 저장해
기존 데이터셋 경로(dxconv.dataset)를 그대로 쓴다.

규칙:
- 중복 제거: 내용(SHA-256)이 같은 이미지는 하나만 쓴다 (경로 순 첫 파일). 분할 간 누수 방지.
- 층화: 이미지마다 "대표 클래스" = 들어 있는 클래스 중 전체에서 가장 드문 클래스
  (객체가 없으면 background). 대표 클래스별로 시드 셔플 후, 비율이 가장 모자란 그룹부터
  한 장씩 뽑는 순서를 만든다 → 순서의 어느 앞부분·구간도 전체 비율과 비슷하다.
- 그 순서의 앞에서 캘리브레이션 풀, 다음 튜닝, 나머지 평가. 캘리브레이션 후보(N)는
  풀의 앞부분이라 작은 N ⊂ 큰 N (중첩).
- 라벨 파일이 없으면 객체 없음(YOLO 관례)으로 보고 개수를 기록한다.
- 같은 폴더·같은 설정이면 항상 같은 분할 (재현 가능). 결과 폴더 이름에 지문(hash)을 넣어 재사용.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dxconv import DxconvError
from dxconv.config import SplitCfg

BACKGROUND = "background"
MIN_SPLIT = 1  # 각 분할 최소 장수


@dataclass
class Item:
    rel: str            # images_dir 기준 상대 경로
    sha256: str
    classes: list[int]  # 라벨에 있는 클래스 (중복 포함)
    has_label: bool


@dataclass
class SplitPlan:
    calibration_pool: int
    calibration_candidates: list[int]
    tuning: int
    evaluation: int
    warnings: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_label_classes(path: Path) -> list[int] | None:
    if not path.is_file():
        return None
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            try:
                out.append(int(float(parts[0])))
            except ValueError:
                continue
    return out


def scan(images_dir: Path, labels_dir: Path | None) -> tuple[list[Item], dict[str, Any]]:
    """이미지 목록 + 라벨 클래스 + SHA-256. 라벨은 labels_dir/<stem>.txt (기준선 평가와 같은 규칙)."""
    from dxconv.compile.calib import scan_images

    if not images_dir.is_dir():
        raise DxconvError(f"이미지 폴더가 없습니다: {images_dir}")
    if labels_dir is not None and not labels_dir.is_dir():
        raise DxconvError(f"라벨 폴더가 없습니다: {labels_dir}")
    paths, ignored = scan_images(images_dir)
    if not paths:
        raise DxconvError(f"이미지가 없습니다 (jpg/jpeg/png/bmp): {images_dir}")
    stems = Counter(p.stem for p in paths)
    dup_stems = [s for s, n in stems.items() if n > 1]
    if labels_dir is not None and dup_stems:
        raise DxconvError(f"파일 이름(확장자 제외)이 같은 이미지가 있어 라벨을 구분할 수 없습니다: "
                          f"{dup_stems[:5]}")
    items: list[Item] = []
    seen: dict[str, str] = {}
    duplicates: list[list[str]] = []
    missing_labels = 0
    for p in paths:
        rel = p.relative_to(images_dir).as_posix()
        sha = _sha256(p)
        if sha in seen:
            duplicates.append([seen[sha], rel])
            continue
        seen[sha] = rel
        cls = read_label_classes(labels_dir / f"{p.stem}.txt") if labels_dir else None
        if labels_dir is not None and cls is None:
            missing_labels += 1
        items.append(Item(rel, sha, cls or [], cls is not None))
    info = {"images_found": len(paths), "ignored_files": ignored, "duplicates_removed": len(duplicates),
            "duplicate_examples": duplicates[:10], "missing_labels": missing_labels,
            "labels_dir": str(labels_dir) if labels_dir else None}
    return items, info


def plan_sizes(n: int, cfg: SplitCfg) -> SplitPlan:
    warnings: list[str] = []
    pool = round(n * cfg.calibration_ratio)
    if cfg.calibration_max is not None:
        pool = min(pool, cfg.calibration_max)
    tune = round(n * cfg.tuning_ratio)
    if cfg.tuning_max is not None:
        tune = min(tune, cfg.tuning_max)
    ev = n - pool - tune
    if min(pool, tune, ev) < MIN_SPLIT:
        raise DxconvError(f"이미지 {n}장으로는 캘리브레이션/튜닝/평가를 나눌 수 없습니다 "
                          f"(계산: {pool}/{tune}/{ev}). 이미지를 늘리거나 split 비율을 조정하세요.")
    cands = sorted({max(1, round(pool * f)) for f in cfg.calibration_fractions})
    if ev < cfg.evaluation_min:
        warnings.append(f"평가 분할이 {ev}장으로 evaluation_min({cfg.evaluation_min}) 보다 적습니다. "
                        "최종 정확도 수치의 신뢰도가 낮습니다.")
    if pool < 50:
        warnings.append(f"캘리브레이션 풀이 {pool}장입니다 (후보 {cands}). 양자화 품질이 불안정할 수 있습니다.")
    if tune < 50:
        warnings.append(f"튜닝 분할이 {tune}장입니다. 조합 간 mAP 차이를 구분하기 어려울 수 있습니다.")
    return SplitPlan(pool, cands, tune, ev, warnings)


def group_key(item: Item, rarity: dict[int, int]) -> str:
    if not item.classes:
        return BACKGROUND
    c = min(set(item.classes), key=lambda k: (rarity[k], k))
    return f"class_{c}"


def stratified_order(items: list[Item], seed: int) -> tuple[list[Item], dict[str, str]]:
    """대표 클래스 비율이 어느 앞부분에서도 유지되는 순서."""
    rarity = Counter(c for it in items for c in set(it.classes))
    groups: dict[str, list[Item]] = {}
    keys: dict[str, str] = {}
    for it in sorted(items, key=lambda x: x.rel):
        k = group_key(it, rarity)
        groups.setdefault(k, []).append(it)
        keys[it.rel] = k
    for k, lst in groups.items():
        random.Random(f"{seed}:{k}").shuffle(lst)
    total = len(items)
    taken = dict.fromkeys(groups, 0)
    order: list[Item] = []
    for t in range(total):
        # 지금까지 t+1 장일 때 목표 대비 가장 모자란 그룹 (동률이면 이름순)
        k = max((k for k in sorted(groups) if taken[k] < len(groups[k])),
                key=lambda k: round((t + 1) * len(groups[k]) / total - taken[k], 9))
        order.append(groups[k][taken[k]])
        taken[k] += 1
    return order, keys


def fingerprint(items: list[Item], cfg: SplitCfg, images_dir: Path, labels_dir: Path | None) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode())
    h.update(str(images_dir.resolve()).encode())
    h.update(str(labels_dir.resolve() if labels_dir else "").encode())
    for it in items:
        h.update(f"{it.rel}\0{it.sha256}\0{','.join(map(str, it.classes))}\n".encode())
    return h.hexdigest()


def _class_counts(items: list[Item], keys: dict[str, str]) -> dict[str, Any]:
    return {"images": len(items),
            "group": dict(sorted(Counter(keys[it.rel] for it in items).items())),
            "objects_per_class": {str(k): v for k, v in
                                  sorted(Counter(c for it in items for c in it.classes).items())}}


def auto_split(images_dir: Path, labels_dir: Path | None, cfg: SplitCfg, out_root: Path,
               name: str) -> Path:
    """폴더를 분할해 <out_root>/<name>_<지문>/dataset.yaml 을 만든다 (이미 있으면 재사용)."""
    images_dir, labels_dir = Path(images_dir).expanduser(), (
        Path(labels_dir).expanduser() if labels_dir else None)
    items, scan_info = scan(images_dir, labels_dir)
    fp = fingerprint(items, cfg, images_dir, labels_dir)
    root = Path(out_root) / f"{name}_{fp[:12]}"
    ds_yaml = root / "dataset.yaml"
    if ds_yaml.is_file():
        return ds_yaml
    plan = plan_sizes(len(items), cfg)
    order, keys = stratified_order(items, cfg.seed)
    pool = order[:plan.calibration_pool]
    tune = order[plan.calibration_pool:plan.calibration_pool + plan.tuning]
    ev = order[plan.calibration_pool + plan.tuning:]
    lists = root / "lists"
    lists.mkdir(parents=True, exist_ok=True)

    def write(fname: str, its: list[Item]) -> str:
        (lists / fname).write_text("\n".join(it.rel for it in its) + "\n", encoding="utf-8")
        return f"lists/{fname}"

    cal = {n: write(f"calibration_{n}.txt", pool[:n]) for n in plan.calibration_candidates}
    info = {
        "fingerprint": fp,
        "images_dir": str(images_dir.resolve()),
        **scan_info,
        "split_cfg": cfg.model_dump(mode="json"),
        "used_images": len(items),
        "sizes": {"calibration_pool": plan.calibration_pool,
                  "calibration_candidates": plan.calibration_candidates,
                  "tuning": plan.tuning, "evaluation": plan.evaluation},
        "classes": {"all": _class_counts(items, keys),
                    "calibration": _class_counts(pool, keys),
                    "tuning": _class_counts(tune, keys),
                    "evaluation": _class_counts(ev, keys)},
        "warnings": plan.warnings
        + ([f"라벨 파일이 없는 이미지 {scan_info['missing_labels']}장 (객체 없음으로 처리)."]
           if scan_info["missing_labels"] else [])
        + ([f"내용이 같은 이미지 {scan_info['duplicates_removed']}장을 제외했습니다."]
           if scan_info["duplicates_removed"] else [])
        + ([] if labels_dir else ["라벨 폴더가 없어 클래스 층화 없이 나눴고, 튜닝/평가 mAP 는 계산할 수 없습니다."]),
    }
    (root / "split_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    ds = {
        "name": name,
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()) if labels_dir else None,
        "calibration": cal,
        "tuning": write("tuning.txt", tune),
        "evaluation": write("evaluation.txt", ev),
        "auto_split": {"fingerprint": fp[:12], "info": "split_info.json"},
        "note": "dxconv 자동 분할 (dxconv/datasplit.py). 직접 고치지 말고 split 설정으로 다시 만드세요.",
    }
    tmp = ds_yaml.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(ds, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(ds_yaml)  # dataset.yaml 은 마지막에 (있으면 분할 완료로 봄)
    return ds_yaml


def split_summary(ds_yaml: Path) -> dict[str, Any]:
    return json.loads((ds_yaml.parent / "split_info.json").read_text(encoding="utf-8"))


def resolve_dataset_input(
    model: str, config_dir: Path, *, dataset: str | None = None, images: Path | None = None,
    labels: Path | None = None, split_cfg: SplitCfg | None = None, out_root: Path = Path("runs"),
    required: bool = True, console: Any = None,
) -> Any:
    """데이터 입력 결정: --images(+--labels) 자동 분할 > --dataset > configs/datasets/<model>.local.yaml.

    반환: dxconv.dataset.Dataset 또는 None (required=False 이고 아무것도 없을 때).
    """
    from dxconv.dataset import find_default_dataset, load_dataset

    if images is not None and dataset is not None:
        raise DxconvError("--images 와 --dataset 은 함께 쓸 수 없습니다.")
    if labels is not None and images is None:
        raise DxconvError("--labels 는 --images 와 함께 지정합니다.")
    if images is not None:
        path = auto_split(Path(images), Path(labels) if labels else None, split_cfg or SplitCfg(),
                          Path(out_root) / "datasets", model)
        if console is not None:
            info = split_summary(path)
            sz = info["sizes"]
            console.print(f"[cyan]자동 분할:[/] {path.parent}  (이미지 {info['used_images']}장 → "
                          f"캘리브레이션 {sz['calibration_candidates']} / 튜닝 {sz['tuning']} / "
                          f"평가 {sz['evaluation']})", highlight=False)
            for w in info["warnings"]:
                console.print(f"[yellow]분할 경고:[/] {w}", highlight=False)
        return load_dataset(str(path), config_dir)
    if dataset is not None:
        return load_dataset(dataset, config_dir)
    default = find_default_dataset(model, config_dir)
    if default is not None:
        if console is not None:
            console.print(f"[cyan]데이터셋 설정 사용:[/] {default}", highlight=False)
        return load_dataset(str(default), config_dir)
    if required:
        raise DxconvError(
            "데이터 입력이 없습니다. --images <이미지 폴더> --labels <라벨 폴더> 로 지정하거나 "
            f"--dataset / configs/datasets/{model}.local.yaml 을 쓰세요.")
    return None


def fit_cfg_to_dataset(cfg: Any, ds: Any, *, explicit_num: bool, console: Any = None) -> Any:
    """자동 분할 데이터셋이면 calibration_num 을 가장 가까운 후보로 맞춘다 (명시한 값이면 그대로)."""
    from dxconv.dataset import fit_calibration_num

    if ds is None or not ds.cfg.is_auto or explicit_num:
        return cfg
    num = cfg.compile.calibration_num
    new = fit_calibration_num(num, ds)
    if new == num:
        return cfg
    if console is not None:
        console.print(f"[yellow]안내:[/] 자동 분할 후보 {sorted(ds.cfg.calibration)} 에 맞춰 "
                      f"calibration_num {num} → {new}", highlight=False)
    return cfg.model_copy(update={"compile": cfg.compile.model_copy(update={"calibration_num": new})})
