"""이미지/라벨 폴더 자동 분할 (dxconv/datasplit.py) 테스트."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from dxconv import DxconvError
from dxconv.config import SplitCfg
from dxconv.dataset import fit_calibration_num, load_dataset
from dxconv.datasplit import auto_split, plan_sizes, split_summary


def _make(root: Path, n: int, *, smoke_every: int = 5, bg_every: int = 7, dup: int = 0,
          missing_label_every: int = 0) -> tuple[Path, Path]:
    """n 장 이미지 + YOLO 라벨. 0=fire 다수, 1=smoke 일부, 일부는 객체 없음."""
    img, lab = root / "images", root / "labels"
    img.mkdir(parents=True)
    lab.mkdir()
    for i in range(n):
        arr = np.random.default_rng(i).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img / f"im_{i:04d}.png")
        if missing_label_every and i % missing_label_every == 0:
            continue
        if bg_every and i % bg_every == 0:
            (lab / f"im_{i:04d}.txt").write_text("")
        elif smoke_every and i % smoke_every == 0:
            (lab / f"im_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.1 0.1\n")
        else:
            (lab / f"im_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    for j in range(dup):  # 내용이 같은 복사본
        (img / f"dup_{j}.png").write_bytes((img / f"im_{j:04d}.png").read_bytes())
    return img, lab


@pytest.mark.parametrize(("n", "pool", "cands", "tune", "ev"), [
    (1000, 200, [50, 100, 200], 200, 600),     # 기존 Fire 분할과 같은 크기
    (300, 60, [15, 30, 60], 60, 180),
    (10000, 200, [50, 100, 200], 500, 9300),   # 상한
])
def test_plan_sizes(n, pool, cands, tune, ev):
    p = plan_sizes(n, SplitCfg())
    assert (p.calibration_pool, p.calibration_candidates, p.tuning, p.evaluation) == (pool, cands, tune, ev)


def test_plan_sizes_small_warns_and_too_small_fails():
    p = plan_sizes(100, SplitCfg())
    assert p.evaluation == 60 and any("evaluation_min" in w for w in p.warnings)
    with pytest.raises(DxconvError):
        plan_sizes(2, SplitCfg())


def test_split_config_validation():
    with pytest.raises(ValueError):
        SplitCfg(calibration_ratio=0.6, tuning_ratio=0.5)
    with pytest.raises(ValueError):
        SplitCfg(calibration_fractions=[0.0, 0.5])
    assert SplitCfg(calibration_fractions=[1.0, 0.5, 0.5]).calibration_fractions == [0.5, 1.0]


def test_auto_split_properties(tmp_path: Path):
    img, lab = _make(tmp_path / "data", 300, dup=3)
    path = auto_split(img, lab, SplitCfg(), tmp_path / "out", "fire")
    ds = load_dataset(str(path), tmp_path)
    assert ds.cfg.is_auto and sorted(ds.cfg.calibration) == [15, 30, 60]
    # 중첩 / 겹침 없음 / 중복 제거 → 검사 통과
    res = ds.check()
    assert res["ok"], res["problems"]
    assert res["nested"] is True
    info = split_summary(path)
    assert info["duplicates_removed"] == 3 and info["used_images"] == 300
    # 층화: 각 분할의 대표 클래스 비율이 전체와 ±1장 수준
    allg = info["classes"]["all"]["group"]
    for split in ("calibration", "tuning", "evaluation"):
        g = info["classes"][split]["group"]
        n = info["classes"][split]["images"]
        for k, v in allg.items():
            assert abs(g.get(k, 0) - v * n / 300) <= 1.5, (split, k, g, allg)
    # 작은 후보도 비율 유지 (앞부분)
    c15 = ds.read_list(ds.cfg.calibration[15])
    keys = Counter("bg" if (lab / (Path(x).stem + ".txt")).read_text() == "" else "x" for x in c15)
    assert 1 <= keys["bg"] <= 4


def test_auto_split_deterministic_and_reused(tmp_path: Path):
    img, lab = _make(tmp_path / "data", 120)
    a = auto_split(img, lab, SplitCfg(), tmp_path / "out", "fire")
    b = auto_split(img, lab, SplitCfg(), tmp_path / "out", "fire")
    assert a == b
    c = auto_split(img, lab, SplitCfg(seed=1), tmp_path / "out", "fire")
    assert c != a
    la = (a.parent / "lists" / "evaluation.txt").read_text()
    lc = (c.parent / "lists" / "evaluation.txt").read_text()
    assert la != lc and sorted(la.split()) != [] and len(la.split()) == len(lc.split())


def test_missing_labels_and_no_labels(tmp_path: Path):
    img, lab = _make(tmp_path / "data", 120, missing_label_every=10)
    path = auto_split(img, lab, SplitCfg(), tmp_path / "out", "fire")
    info = split_summary(path)
    assert info["missing_labels"] == 12 and any("라벨 파일이 없는" in w for w in info["warnings"])
    path2 = auto_split(img, None, SplitCfg(), tmp_path / "out2", "fire")
    assert yaml.safe_load(path2.read_text())["labels_dir"] is None
    assert any("라벨 폴더가 없어" in w for w in split_summary(path2)["warnings"])


def test_fit_calibration_num(tmp_path: Path):
    img, lab = _make(tmp_path / "data", 300)
    ds = load_dataset(str(auto_split(img, lab, SplitCfg(), tmp_path / "out", "fire")), tmp_path)
    assert fit_calibration_num(50, ds) == 60      # 15/30/60 중 가장 가까운 값
    assert fit_calibration_num(100, ds) == 60
    assert fit_calibration_num(30, ds) == 30
    assert fit_calibration_num(22, ds) == 15      # 동률(7)이면 작은 쪽


def test_cli_dataset_split_and_convert_dry_run(tmp_path: Path):
    from typer.testing import CliRunner

    from dxconv.cli import app

    img, lab = _make(tmp_path / "data", 150)
    r = CliRunner().invoke(app, ["dataset", "split", "--images", str(img), "--labels", str(lab),
                                 "--model", "fire", "--out", str(tmp_path / "runs")])
    assert r.exit_code == 0, r.output
    assert "자동 분할" in r.output and len(list((tmp_path / "runs" / "datasets").glob("fire_*"))) == 1
    r = CliRunner().invoke(app, ["convert", "--pt", str(tmp_path / "none.pt"), "--model", "fire",
                                 "--images", str(img), "--dataset", "x"])
    assert r.exit_code == 1 and "함께 쓸 수 없습니다" in r.output
