from __future__ import annotations

from pathlib import Path

import pytest

from dxconv.config import (
    ConfigError,
    deep_merge,
    merge_layers,
    parse_set_items,
    resolve_config,
    save_resolved,
)


def test_deep_merge_is_recursive_and_pure():
    base = {"a": {"b": 1, "c": 2}, "x": 1}
    over = {"a": {"b": 10}, "y": 2}
    out = deep_merge(base, over)
    assert out == {"a": {"b": 10, "c": 2}, "x": 1, "y": 2}
    assert base == {"a": {"b": 1, "c": 2}, "x": 1}


def test_defaults_only(config_dir: Path):
    cfg = resolve_config("fall", config_dir=config_dir)
    assert cfg.compile.calibration_num == 50  # recommend.yaml v9 (사용자 이미지 있을 때)
    assert cfg.compile.calibration_method == "minmax"
    assert cfg.export.opset == 17
    assert cfg.compile.preprocess.pad_value == 114


def test_preset_overrides_defaults(config_dir: Path):
    cfg = resolve_config("fire", config_dir=config_dir)
    assert cfg.compile.calibration_num == 50
    assert cfg.compile.opt_level == 1  # defaults 유지


def test_user_config_overrides_preset(config_dir: Path, fixtures: Path):
    cfg = resolve_config("fire", user_config=fixtures / "user_config.yaml", config_dir=config_dir)
    assert cfg.compile.calibration_num == 77
    assert cfg.compile.opt_level == 0
    assert cfg.export.opset == 13
    assert cfg.export.imgsz == 640


def test_set_overrides_user_config(config_dir: Path, fixtures: Path):
    cfg = resolve_config(
        "fire",
        user_config=fixtures / "user_config.yaml",
        set_items=["compile.calibration_num=200", "export.simplify=false"],
        config_dir=config_dir,
    )
    assert cfg.compile.calibration_num == 200
    assert cfg.export.simplify is False
    assert cfg.compile.opt_level == 0


def test_cli_flag_overrides_everything(config_dir: Path, fixtures: Path):
    cfg = resolve_config(
        "fire",
        user_config=fixtures / "user_config.yaml",
        set_items=["compile.calibration_num=200", "export.opset=19"],
        cli_overrides={"compile": {"calibration_num": 30, "opt_level": None},
                       "export": {"opset": 18}},
        config_dir=config_dir,
    )
    assert cfg.compile.calibration_num == 30
    assert cfg.export.opset == 18
    assert cfg.compile.opt_level == 0  # None 은 무시 → user config 값 유지


def test_parse_set_items_types():
    out = parse_set_items(["a.b=200", "a.c=false", "d=ema", "e.f.g=1.5"])
    assert out == {"a": {"b": 200, "c": False}, "d": "ema", "e": {"f": {"g": 1.5}}}


@pytest.mark.parametrize("bad", ["noequals", "=1", "a..b=1"])
def test_parse_set_items_errors(bad: str):
    with pytest.raises(ConfigError):
        parse_set_items([bad])


def test_unknown_key_rejected(config_dir: Path):
    with pytest.raises(ConfigError, match="compile.calib_numm"):
        resolve_config("fire", set_items=["compile.calib_numm=3"], config_dir=config_dir)


def test_invalid_value_rejected(config_dir: Path):
    with pytest.raises(ConfigError, match="calibration_method"):
        resolve_config("fire", set_items=["compile.calibration_method=percentile"], config_dir=config_dir)


@pytest.mark.parametrize("raw, expected", [
    ("minmax", "minmax"), ("MinMax", "minmax"), ("EMA", "ema"), ("iqr", "iqr"), ("IQR", "iqr"),
])
def test_calib_method_case_insensitive(config_dir: Path, raw: str, expected: str):
    cfg = resolve_config("fire", set_items=[f"compile.calibration_method={raw}"],
                         config_dir=config_dir)
    assert cfg.compile.calibration_method == expected


def test_defaults_compile(config_dir: Path):
    cfg = resolve_config("fire", config_dir=config_dir)
    assert cfg.compile.export_html is False
    assert cfg.compile.calib_sampling == "stratified"


def test_model_without_preset_uses_defaults(config_dir: Path):
    """프리셋이 없는 모델 이름도 defaults.yaml 로 동작한다 (어떤 pt 든 받는다)."""
    cfg = resolve_config("my_model", config_dir=config_dir)
    assert cfg.model == "my_model" and cfg.compile.calibration_num == 50
    with pytest.raises(ConfigError):
        resolve_config("bad name/x", config_dir=config_dir)


def test_merge_layers_sets_model(config_dir: Path):
    raw = merge_layers("person", config_dir=config_dir)
    assert raw["model"] == "person"


def test_save_resolved_roundtrip(config_dir: Path, tmp_path: Path):
    import yaml

    cfg = resolve_config("fire", config_dir=config_dir)
    p = save_resolved(cfg, tmp_path / "resolved.yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["compile"]["calibration_num"] == 50
    assert data["model"] == "fire"


def test_version_overlay_preset_and_format(config_dir: Path, tmp_path: Path):
    """presets/<model>@dxcom<버전>.yaml 은 그 버전을 목표로 할 때만 얹힌다."""
    import shutil

    from dxconv.config import list_presets
    from dxconv.env.compat import load_compat, version_for_format

    config_dir = Path(shutil.copytree(config_dir, tmp_path / "configs",
                                      ignore=shutil.ignore_patterns("*.local.yaml", "remote.yaml")))
    (config_dir / "presets" / "fire@dxcom2.4.1.yaml").write_text(
        "compile:\n  dxcom_version: '2.4.1'\n  calibration_num: 77\n")
    assert "fire@dxcom2.4.1" not in list_presets(config_dir)
    assert resolve_config("fire", config_dir=config_dir).compile.calibration_num == 50  # 기준(v9)
    cfg8 = resolve_config("fire", set_items=["compile.dxcom_version=2.4.1"], config_dir=config_dir)
    assert cfg8.compile.calibration_num == 77
    # CLI 값은 여전히 overlay 보다 우선
    cfg8b = resolve_config("fire", set_items=["compile.dxcom_version=2.4.1", "compile.calibration_num=5"],
                           config_dir=config_dir)
    assert cfg8b.compile.calibration_num == 5
    compat = load_compat(config_dir)
    assert version_for_format(8, compat) == "2.4.1" and version_for_format(9, compat) == "2.5.0"


def test_sweep_root_per_version(tmp_path: Path):
    from dxconv.sweep.stages import sweep_root

    assert sweep_root(tmp_path, "fire", "2.5.0").name == "fire"
    assert sweep_root(tmp_path, "fire", None).name == "fire"
    assert sweep_root(tmp_path, "fire", "2.4.1").name == "fire@dxcom2.4.1"


def test_no_data_uses_settings_no_data(config_dir: Path):
    """이미지 입력 없이 기본 캘리브레이션 사진을 쓸 때: recommend.yaml settings_no_data (ema·100)."""
    cfg = resolve_config("fall", config_dir=config_dir, no_data=True)
    assert (cfg.compile.calibration_method, cfg.compile.calibration_num) == ("ema", 100)
    assert cfg.compile.opt_level == 1 and cfg.compile.aggressive_partitioning is False
    # CLI 값이 우선
    cfg = resolve_config("fall", config_dir=config_dir, no_data=True,
                         cli_overrides={"compile": {"calibration_method": "minmax"}})
    assert cfg.compile.calibration_method == "minmax"
