"""데이터셋 분할 설정 (캘리브레이션 / 튜닝 / 평가).

모델별 개인 설정 파일 `configs/datasets/<model>.local.yaml` (git 제외) 에 경로를 둔다.
예시: `configs/datasets/example.yaml`.

역할:
- calibration: 캘리브레이션 원본 이미지 목록. calibration_num(N) 별 목록 파일을 지정한다.
  dxconv 가 원본을 읽어 letterbox 한다 (compile.preprocess.letterbox_mode).
- tuning:      sweep 최적값 선택용 검증 목록
- evaluation:  최종 보고 수치용 검증 목록

목록 파일은 한 줄에 하나씩 images_dir 기준 파일 이름(상대 경로)을 적는다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dxconv import DxconvError
from dxconv.config import ConfigError, load_yaml

DATASET_DIR = "datasets"
LOCAL_SUFFIX = ".local.yaml"
SPLITS = ("tuning", "evaluation")


class DatasetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    images_dir: Path
    labels_dir: Path | None = None
    # calibration_num → 목록 파일 (예: {50: calibration_50.txt, 100: ..., 200: ...})
    calibration: dict[int, Path] = Field(default_factory=dict)
    tuning: Path | None = None
    evaluation: Path | None = None
    note: str | None = None
    auto_split: dict[str, Any] | None = None  # dxconv 자동 분할로 만든 설정이면 지문·정보 파일

    @property
    def is_auto(self) -> bool:
        return self.auto_split is not None


@dataclass
class Dataset:
    cfg: DatasetCfg
    source: Path  # 설정 파일 경로
    _cache: dict[Path, list[str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.cfg.name

    def read_list(self, path: Path) -> list[str]:
        if path not in self._cache:
            if not path.is_file():
                raise DxconvError(f"[{self.name}] 목록 파일이 없습니다: {path}")
            names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
            self._cache[path] = [n for n in names if n and not n.startswith("#")]
        return self._cache[path]

    def _images(self, names: list[str], what: str) -> list[Path]:
        paths = [self.cfg.images_dir / n for n in names]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise DxconvError(
                f"[{self.name}] {what}: 이미지 {len(missing)} 개가 없습니다 "
                f"(예: {missing[0]}). images_dir 를 확인하세요: {self.cfg.images_dir}"
            )
        return paths

    def calibration_list(self, num: int) -> Path:
        if num not in self.cfg.calibration:
            avail = ", ".join(str(n) for n in sorted(self.cfg.calibration)) or "(없음)"
            raise DxconvError(
                f"[{self.name}] calibration_num={num} 에 해당하는 캘리브레이션 목록이 없습니다. "
                f"사용 가능: {avail} (설정: {self.source})"
            )
        return self.cfg.calibration[num]

    def calibration_images(self, num: int) -> list[Path]:
        lst = self.calibration_list(num)
        names = self.read_list(lst)
        if len(names) != num:
            raise DxconvError(
                f"[{self.name}] {lst.name}: 이미지 {num} 장이어야 하는데 {len(names)} 장입니다."
            )
        return self._images(names, f"calibration {num}")

    def calibration_sample(self, num: int, repeat: int = 0) -> list[Path]:
        """repeat 0: calibration_images(num) 그대로. 1 이상: 가장 큰 캘리브레이션 목록(풀)에서
        repeat 를 시드로 num 장 무작위 추출 (풀이 num 장 이하면 풀 전체 = repeat 0 과 같은 사진)."""
        if repeat == 0:
            return self.calibration_images(num)
        import random

        pool_num = max(self.cfg.calibration)
        pool = self.read_list(self.calibration_list(pool_num))
        if len(pool) <= num:
            return self.calibration_images(num) if num in self.cfg.calibration else \
                self._images(sorted(pool), f"calibration pool {pool_num}")
        names = sorted(random.Random(f"calib-repeat-{repeat}").sample(sorted(pool), num))
        return self._images(names, f"calibration {num} (묶음 {repeat})")

    def split_images(self, split: str) -> list[Path]:
        if split not in SPLITS:
            raise ValueError(split)
        lst = getattr(self.cfg, split)
        if lst is None:
            raise DxconvError(f"[{self.name}] '{split}' 목록이 설정되지 않았습니다 ({self.source}).")
        return self._images(self.read_list(lst), split)

    def describe(self) -> dict[str, Any]:
        """report.json 에 남길 요약."""
        def lst(p: Path | None) -> dict[str, Any] | None:
            if p is None:
                return None
            return {"path": str(p), "sha256": _sha256(p) if p.is_file() else None}
        return {
            "name": self.name,
            "config": str(self.source),
            "images_dir": str(self.cfg.images_dir),
            "calibration": {str(n): lst(p) for n, p in sorted(self.cfg.calibration.items())},
            "tuning": lst(self.cfg.tuning),
            "evaluation": lst(self.cfg.evaluation),
            "auto_split": self.cfg.auto_split,
        }

    def check(self) -> dict[str, Any]:
        """분할 검증: 파일 존재, 목록 내 중복, 캘리브레이션 중첩, 분할 간 겹침(SHA-256)."""
        out: dict[str, Any] = {"counts": {}, "problems": [], "overlaps": {}, "nested": None}
        hashes: dict[str, set[str]] = {}
        lists: dict[str, list[str]] = {}
        named = {f"calibration_{n}": p for n, p in sorted(self.cfg.calibration.items())}
        named.update({s: getattr(self.cfg, s) for s in SPLITS if getattr(self.cfg, s)})
        for key, path in named.items():
            try:
                names = self.read_list(path)
                imgs = self._images(names, key)
            except DxconvError as e:
                out["problems"].append(str(e))
                continue
            lists[key] = names
            out["counts"][key] = len(names)
            if len(set(names)) != len(names):
                out["problems"].append(f"{key}: 목록 안에 중복된 이름이 있습니다.")
            hs = [_sha256(p) for p in imgs]
            hashes[key] = set(hs)
            if len(hashes[key]) != len(hs):
                out["problems"].append(f"{key}: 내용이 같은(SHA-256) 이미지가 목록 안에 있습니다.")
        cal_keys = [k for k in lists if k.startswith("calibration_")]
        cal_keys.sort(key=lambda k: int(k.split("_")[1]))
        if len(cal_keys) > 1:
            nested = all(lists[b][: len(lists[a])] == lists[a]
                         for a, b in zip(cal_keys, cal_keys[1:], strict=False))
            out["nested"] = nested
            if not nested:
                out["problems"].append("캘리브레이션 목록이 중첩(작은 N 이 큰 N 의 앞부분)이 아닙니다.")
        groups = {k: v for k, v in hashes.items() if k in SPLITS}
        if cal_keys:
            groups["calibration"] = set().union(*(hashes[k] for k in cal_keys if k in hashes))
        keys = sorted(groups)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                n = len(groups[a] & groups[b])
                out["overlaps"][f"{a}&{b}"] = n
                if n:
                    out["problems"].append(f"{a} 와 {b} 가 이미지 {n} 장을 공유합니다 (SHA-256).")
        out["ok"] = not out["problems"]
        return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_path(name_or_path: str, config_dir: Path) -> Path:
    """'fire' → configs/datasets/fire.local.yaml, 그 외에는 파일 경로로 본다."""
    p = Path(name_or_path).expanduser()
    if p.suffix in (".yaml", ".yml") or p.is_file():
        return p
    return config_dir / DATASET_DIR / f"{name_or_path}{LOCAL_SUFFIX}"


def load_dataset(name_or_path: str, config_dir: Path) -> Dataset:
    path = dataset_path(name_or_path, config_dir)
    if not path.is_file():
        raise ConfigError(
            f"데이터셋 설정 파일이 없습니다: {path}\n"
            f"  configs/{DATASET_DIR}/example.yaml 을 <model>{LOCAL_SUFFIX} 로 복사해 경로를 채우거나, "
            "--images/--labels 로 폴더를 주세요."
        )
    raw = load_yaml(path)
    base = path.parent

    def resolve(v: Any) -> Any:
        if v is None:
            return None
        q = Path(str(v)).expanduser()
        return q if q.is_absolute() else (base / q)

    try:
        cfg = DatasetCfg.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"데이터셋 설정 오류 ({path}):\n{e}") from e
    # 상대 경로는 설정 파일 위치 기준. 목록 파일 상대 경로는 lists_dir(=설정 파일 위치) 기준.
    cfg.images_dir = resolve(cfg.images_dir)
    cfg.labels_dir = resolve(cfg.labels_dir)
    cfg.calibration = {int(n): resolve(p) for n, p in cfg.calibration.items()}
    cfg.tuning = resolve(cfg.tuning)
    cfg.evaluation = resolve(cfg.evaluation)
    return Dataset(cfg=cfg, source=path)


def fit_calibration_num(num: int, ds: Dataset) -> int:
    """자동 분할 데이터셋에 요청 calibration_num 목록이 없으면 가장 가까운 후보 (같으면 작은 쪽)."""
    avail = sorted(ds.cfg.calibration)
    if num in avail or not avail:
        return num
    return min(avail, key=lambda n: (abs(n - num), n))


def find_default_dataset(model: str, config_dir: Path) -> Path | None:
    """configs/datasets/<model>.local.yaml 이 있으면 그 경로."""
    p = config_dir / DATASET_DIR / f"{model}{LOCAL_SUFFIX}"
    return p if p.is_file() else None
