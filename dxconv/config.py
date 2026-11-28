"""설정 로드 · 병합 · 검증.

우선순위 (위가 높음):
    1. CLI 인수 (명시적 옵션 > --set 오버라이드)
    2. --config 로 지정한 사용자 설정 파일
    3. configs/presets/<model>.yaml
    4. configs/defaults.yaml
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from dxconv import DxconvError


class ConfigError(DxconvError):
    """설정 파일/값 오류."""


# --------------------------------------------------------------------------- #
# 경로
# --------------------------------------------------------------------------- #
def find_config_dir() -> Path:
    """configs 디렉터리 위치를 찾는다.

    순서: 환경변수 DXCONV_CONFIG_DIR > 저장소 루트(editable 설치) > 현재 디렉터리의 configs/
    """
    env = os.environ.get("DXCONV_CONFIG_DIR")
    if env:
        p = Path(env).expanduser()
        if not p.is_dir():
            raise ConfigError(f"DXCONV_CONFIG_DIR 경로가 디렉터리가 아닙니다: {p}")
        return p
    repo = Path(__file__).resolve().parent.parent / "configs"
    if (repo / "defaults.yaml").is_file():
        return repo
    cwd = Path.cwd() / "configs"
    if (cwd / "defaults.yaml").is_file():
        return cwd
    raise ConfigError(
        "configs 디렉터리(defaults.yaml)를 찾을 수 없습니다. "
        "DXCONV_CONFIG_DIR 환경변수로 위치를 지정하세요."
    )


def list_presets(config_dir: Path) -> list[str]:
    """사용 가능한 모델 프리셋 이름 목록."""
    return sorted(p.stem for p in (config_dir / "presets").glob("*.yaml") if "@" not in p.stem)


REFERENCE_VERSION = "2.5.0"


def preset_overlay_path(config_dir: Path, model: str, dxcom_version: str | None) -> Path | None:
    """기준(2.5.0)이 아닌 DX-COM 버전용 프리셋: presets/<model>@dxcom<버전>.yaml (sweep 결과 저장 위치)."""
    if not dxcom_version or str(dxcom_version) == REFERENCE_VERSION:
        return None
    return config_dir / "presets" / f"{model}@dxcom{dxcom_version}.yaml"


# --------------------------------------------------------------------------- #
# YAML / 병합
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAML 파일을 dict로 읽는다. 빈 파일은 {}."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"설정 파일을 읽을 수 없습니다: {path} ({e})") from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 문법 오류: {path}\n{e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"설정 파일의 최상위는 매핑(dict)이어야 합니다: {path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """override를 base 위에 깊은 병합한 새 dict를 반환한다 (원본 불변).

    둘 다 dict인 키는 재귀 병합, 그 외에는 override 값으로 교체.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_set_items(items: Iterable[str]) -> dict[str, Any]:
    """`--set a.b.c=value` 목록을 중첩 dict로 변환한다.

    값은 YAML 스칼라로 해석한다 (200 → int, false → bool, ema → str).
    """
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(
                f"--set 형식 오류: '{item}' (예: --set compile.calibration_num=200)"
            )
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        if not key.strip() or any(not p for p in parts):
            raise ConfigError(f"--set 키 형식 오류: '{item}'")
        try:
            value = yaml.safe_load(raw) if raw.strip() else ""
        except yaml.YAMLError as e:
            raise ConfigError(f"--set 값을 해석할 수 없습니다: '{item}' ({e})") from e
        cur = out
        for p in parts[:-1]:
            nxt = cur.setdefault(p, {})
            if not isinstance(nxt, dict):
                raise ConfigError(f"--set 키 충돌: '{item}'")
            cur = nxt
        cur[parts[-1]] = value
    return out


def drop_none(d: Mapping[str, Any]) -> dict[str, Any]:
    """값이 None인 키를 재귀적으로 제거 (지정되지 않은 CLI 옵션 제거용)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            sub = drop_none(v)
            if sub:
                out[k] = sub
        elif v is not None:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# 스키마
# --------------------------------------------------------------------------- #
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


LetterboxMode = Literal["host", "dxcom", "none"]


class PreprocessCfg(_Strict):
    """캘리브레이션 전처리. Ultralytics 학습/추론 전처리와 일치해야 한다.

    letterbox_mode:
      host  — Ultralytics LetterBox 로 캘리브레이션 이미지를 미리 처리 (DX-COM 에는 resize 없음)
      dxcom — DX-COM 전처리(resize, mode=pad)로 letterbox (DX-COM 2.5.0 에서는 center 미지원으로 실패)
      none  — letterbox 없이 DX-COM resize (비율 무시, 학습 전처리와 다름)
    """

    letterbox_mode: LetterboxMode = "host"
    pad_value: int = Field(114, ge=0, le=255)
    color: Literal["RGB", "BGR"] = "RGB"
    scale: float = Field(255.0, gt=0)
    layout: Literal["NCHW", "NHWC"] = "NCHW"

    @model_validator(mode="before")
    @classmethod
    def _legacy_letterbox(cls, data: Any) -> Any:
        # 이전 설정 키 letterbox: true/false → letterbox_mode.
        # defaults.yaml 에 letterbox_mode 가 항상 있으므로 legacy 값이 우선한다:
        # false → none, true → (letterbox_mode 가 none 이면) host, 아니면 기존 값 유지.
        if isinstance(data, dict) and isinstance(data.get("letterbox"), bool):
            data = dict(data)
            legacy = data.pop("letterbox")
            mode = str(data.get("letterbox_mode", "host")).strip().lower()
            data["letterbox_mode"] = ("host" if mode == "none" else mode) if legacy else "none"
        return data

    @field_validator("letterbox_mode", mode="before")
    @classmethod
    def _lower_mode(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v


class ExportCfg(_Strict):
    opset: int = 17
    simplify: bool = True
    dynamic: bool = False
    half: bool = False
    nms: bool = False
    end2end: bool = True
    max_det: int = Field(300, gt=0)
    imgsz: int = Field(640, gt=0)
    batch: int = Field(1, gt=0)


#: DX-COM 2.5.0 `dxcom --help` 의 observer 목록(--recalibration_method: minmax/ema/iqr)과 같은 표기.
CalibMethod = Literal["minmax", "ema", "iqr"]


class CompileCfg(_Strict):
    dxcom_version: str | None = "2.5.0"
    opt_level: Literal[0, 1] = 1
    aggressive_partitioning: bool = True
    calibration_method: CalibMethod = "ema"
    calibration_num: int = Field(100, gt=0)
    calib_seed: int = 0
    # 데이터셋 캘리브레이션 사진 묶음 번호: 0 = 분할 목록 그대로, 1 이상 = 캘리브레이션 풀에서 이 번호를 시드로
    # calibration_num 장을 무작위 추출 (sweep 이 방식별로 여러 묶음을 평균 낼 때 사용)
    calib_repeat: int = Field(0, ge=0)
    # stratified: calib-dir 바로 아래 하위 폴더(클래스) 비율대로 샘플링, random: 전체에서 무작위
    calib_sampling: Literal["random", "stratified"] = "stratified"
    # 이미지 입력이 없을 때 받아 쓸 기본 캘리브레이션 사진 묶음 (zip, 안의 jpg/png 전부 사용)
    default_calib_url: str = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    # dxcom --export_html: 컴파일 요약 HTML 을 만들어 run 폴더에 복사
    export_html: bool = False
    preprocess: PreprocessCfg = Field(default_factory=PreprocessCfg)

    @field_validator("calibration_method", mode="before")
    @classmethod
    def _normalize_method(cls, v: Any) -> Any:
        # EMA / MinMax / IQR 등 대소문자 표기도 허용
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("dxcom_version", mode="before")
    @classmethod
    def _version_str(cls, v: Any) -> Any:
        return None if v is None else str(v)


class PostCfg(_Strict):
    conf: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.7, ge=0.0, le=1.0)
    # 지정하지 않으면 export.max_det 를 그대로 사용한다.
    max_det: int | None = Field(None, gt=0)


class EvalCfg(_Strict):
    """mAP 평가 설정. 배포용 postprocess 와 분리한다."""

    # mAP 계산용 최소 점수 (COCO/Ultralytics 관례 0.001). 배포 임계값은 postprocess.conf.
    conf: float = Field(0.001, ge=0.0, le=1.0)
    # 이미지당 최대 예측 수. 지정하지 않으면 export.max_det.
    max_det: int | None = Field(None, gt=0)
    # NPU FPS 측정 시간(초) (원격 평가 bench_fps / run_model -b)
    fps_seconds: float = Field(10.0, gt=0)


class SplitCfg(_Strict):
    """이미지/라벨 폴더 자동 분할 비율 (dxconv/datasplit.py).

    캘리브레이션 풀 = 전체 × calibration_ratio (최대 calibration_max),
    캘리브레이션 후보 = 풀 × calibration_fractions (중첩, B 단계 calibration_num 탐색값),
    튜닝 = 전체 × tuning_ratio (최대 tuning_max), 평가 = 나머지.
    """
    seed: int = 0
    calibration_ratio: float = Field(0.2, gt=0.0, lt=1.0)
    calibration_max: int | None = Field(200, gt=0)
    calibration_fractions: list[float] = Field(default_factory=lambda: [0.25, 0.5, 1.0])
    tuning_ratio: float = Field(0.2, gt=0.0, lt=1.0)
    tuning_max: int | None = Field(500, gt=0)
    evaluation_min: int = Field(100, ge=0)  # 이보다 적으면 경고

    @model_validator(mode="after")
    def _check(self) -> SplitCfg:
        if self.calibration_ratio + self.tuning_ratio >= 1.0:
            raise ValueError("calibration_ratio + tuning_ratio 는 1 보다 작아야 합니다 (평가 분할 필요).")
        fr = self.calibration_fractions
        if not fr or any(not 0.0 < f <= 1.0 for f in fr):
            raise ValueError("calibration_fractions 는 (0, 1] 범위의 값 목록이어야 합니다.")
        self.calibration_fractions = sorted(set(fr))
        return self


class RunCfg(_Strict):
    model: str
    export: ExportCfg = Field(default_factory=ExportCfg)
    compile: CompileCfg = Field(default_factory=CompileCfg)
    postprocess: PostCfg = Field(default_factory=PostCfg)
    evaluate: EvalCfg = Field(default_factory=EvalCfg)
    split: SplitCfg = Field(default_factory=SplitCfg)


def _format_validation_error(err: ValidationError) -> str:
    lines = ["설정값 검증 실패:"]
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"])
        lines.append(f"  - {loc}: {e['msg']} (입력값: {e.get('input')!r})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 해석
# --------------------------------------------------------------------------- #
def merge_layers(
    model: str,
    *,
    user_config: str | Path | None = None,
    set_items: Iterable[str] = (),
    cli_overrides: Mapping[str, Any] | None = None,
    config_dir: Path | None = None,
    no_data: bool = False,
) -> dict[str, Any]:
    """4단 우선순위로 병합된 원시 dict를 반환한다 (검증 전).

    no_data: 사용자 이미지 없이 기본 캘리브레이션 사진을 쓸 때 → 추천값의 settings_no_data 를 얹는다.
    """
    config_dir = config_dir or find_config_dir()
    defaults_path = config_dir / "defaults.yaml"
    if not defaults_path.is_file():
        raise ConfigError(f"defaults.yaml 이 없습니다: {defaults_path}")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", model or ""):
        raise ConfigError(f"모델 이름은 영문/숫자/_ . - 만 쓸 수 있습니다: {model!r}")
    # 프리셋(presets/<model>.yaml)은 선택 사항: 없으면 defaults.yaml 만으로 진행한다.

    def _merge(overlay: Path | None, rec: dict[str, Any] | None = None) -> dict[str, Any]:
        m = load_yaml(defaults_path)
        if rec:
            m = deep_merge(m, rec)
        preset = config_dir / "presets" / f"{model}.yaml"
        if preset.is_file():
            m = deep_merge(m, load_yaml(preset))
        if overlay is not None and overlay.is_file():
            m = deep_merge(m, load_yaml(overlay))
        if user_config is not None:
            m = deep_merge(m, load_yaml(user_config))
        m = deep_merge(m, parse_set_items(set_items))
        if cli_overrides:
            m = deep_merge(m, drop_none(cli_overrides))
        m["model"] = model
        return m

    merged = _merge(None)
    version = (merged.get("compile") or {}).get("dxcom_version")
    # 목표 DX-COM 버전의 dxnn 포맷 추천값 (recommend.yaml) 을 defaults 위에 얹는다
    rec = recommend_settings(config_dir, version, no_data=no_data)
    # 목표 DX-COM 버전 전용 프리셋 (예: presets/fire@dxcom2.4.1.yaml) 이 있으면 모델 프리셋 위에 얹는다
    overlay = preset_overlay_path(config_dir, model, version)
    if rec or (overlay is not None and overlay.is_file()):
        merged = _merge(overlay, rec)
    return merged


def recommend_format(config_dir: Path, dxcom_version: str | None) -> int | None:
    """DX-COM 버전 → dxnn 포맷 (compat.yaml)."""
    if not dxcom_version:
        return None
    compat = load_yaml(config_dir / "compat.yaml") if (config_dir / "compat.yaml").is_file() else {}
    for key, entry in (compat.get("dxcom") or {}).items():
        if str(key).split("-")[0] == str(dxcom_version).split("-")[0]:
            return (entry or {}).get("dxnn_format")
    return None


def recommend_settings(config_dir: Path, dxcom_version: str | None, no_data: bool = False) -> dict[str, Any]:
    """recommend.yaml 에서 목표 포맷의 추천 설정 (없으면 빈 dict).

    no_data=True 면 settings 위에 settings_no_data (기본 캘리브레이션 사진용 값) 를 얹는다.
    """
    path = config_dir / "recommend.yaml"
    fmt = recommend_format(config_dir, dxcom_version)
    if fmt is None or not path.is_file():
        return {}
    entry = ((load_yaml(path) or {}).get("formats") or {}).get(fmt) or {}
    settings = entry.get("settings") or {}
    if no_data and entry.get("settings_no_data"):
        settings = deep_merge(settings, entry["settings_no_data"])
    return settings


def resolve_config(
    model: str,
    *,
    user_config: str | Path | None = None,
    set_items: Iterable[str] = (),
    cli_overrides: Mapping[str, Any] | None = None,
    config_dir: Path | None = None,
    no_data: bool = False,
) -> RunCfg:
    """병합 + pydantic 검증된 최종 설정."""
    raw = merge_layers(
        model,
        user_config=user_config,
        set_items=set_items,
        cli_overrides=cli_overrides,
        config_dir=config_dir,
        no_data=no_data,
    )
    try:
        return RunCfg.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(e)) from e


def save_resolved(cfg: RunCfg, path: str | Path) -> Path:
    """병합된 최종 설정을 resolved.yaml 로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
    return path
