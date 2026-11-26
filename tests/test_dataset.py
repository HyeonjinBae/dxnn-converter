"""데이터셋 분할 설정 (configs/datasets/<model>.local.yaml) 테스트."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import pytest
import yaml
from conftest import REPO, make_images, make_onnx
from typer.testing import CliRunner

from dxconv import DxconvError
from dxconv.cli import app
from dxconv.config import ConfigError
from dxconv.dataset import dataset_path, load_dataset

runner = CliRunner()


def _unique_images(root: Path, n: int) -> list[Path]:
    """내용(SHA-256)이 서로 다른 PNG 이미지 n 장."""
    import numpy as np
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        arr = np.random.default_rng(i).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        p = root / f"img_{i:04d}.png"
        Image.fromarray(arr).save(p)
        out.append(p)
    return out


def _make_dataset(root: Path, *, cal: int = 60, tune: int = 10, ev: int = 10,
                  nested: bool = True, overlap: bool = False) -> Path:
    """images/ 에 이미지를 만들고 lists/ 에 분할 목록, root/ds.yaml 에 설정을 쓴다."""
    imgs = _unique_images(root / "images", cal + tune + ev)
    names = [p.name for p in imgs]
    cal_names, tune_names, ev_names = names[:cal], names[cal:cal + tune], names[cal + tune:]
    if overlap:  # evaluation 첫 이미지를 tuning 첫 이미지와 같은 내용으로
        shutil.copy2(root / "images" / tune_names[0], root / "images" / ev_names[0])
    lists = root / "lists"
    lists.mkdir()
    c20 = cal_names[:20]
    c50 = cal_names[:50] if nested else cal_names[10:60]
    (lists / "cal_20.txt").write_text("\n".join(c20) + "\n")
    (lists / "cal_50.txt").write_text("\n".join(c50) + "\n")
    (lists / "tuning.txt").write_text("\n".join(tune_names) + "\n")
    (lists / "evaluation.txt").write_text("\n".join(ev_names) + "\n")
    cfg = {
        "name": "fire",
        "images_dir": "images",  # 상대 경로: YAML 위치 기준
        "calibration": {20: "lists/cal_20.txt", 50: "lists/cal_50.txt"},
        "tuning": "lists/tuning.txt",
        "evaluation": "lists/evaluation.txt",
    }
    path = root / "ds.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return path


def test_load_and_calibration_images(tmp_path: Path):
    ds = load_dataset(str(_make_dataset(tmp_path)), tmp_path)
    assert ds.cfg.images_dir == tmp_path / "images"
    imgs = ds.calibration_images(50)
    assert len(imgs) == 50 and all(p.is_file() for p in imgs)
    assert ds.calibration_images(20) == imgs[:20]
    assert len(ds.split_images("tuning")) == 10
    assert len(ds.split_images("evaluation")) == 10


def test_calibration_num_not_in_lists(tmp_path: Path):
    ds = load_dataset(str(_make_dataset(tmp_path)), tmp_path)
    with pytest.raises(DxconvError, match="사용 가능: 20, 50"):
        ds.calibration_images(100)


def test_list_count_mismatch_and_missing_image(tmp_path: Path):
    path = _make_dataset(tmp_path)
    (tmp_path / "lists/cal_20.txt").write_text("img_0000.png\n")
    ds = load_dataset(str(path), tmp_path)
    with pytest.raises(DxconvError, match="20 장이어야"):
        ds.calibration_images(20)
    (tmp_path / "images/img_0001.png").unlink()
    with pytest.raises(DxconvError, match="없습니다"):
        ds.calibration_images(50)


def test_check_ok(tmp_path: Path):
    res = load_dataset(str(_make_dataset(tmp_path)), tmp_path).check()
    assert res["ok"], res["problems"]
    assert res["nested"] is True
    assert res["counts"] == {"calibration_20": 20, "calibration_50": 50,
                             "tuning": 10, "evaluation": 10}
    assert set(res["overlaps"].values()) == {0}


def test_check_detects_sha_overlap_and_not_nested(tmp_path: Path):
    res = load_dataset(str(_make_dataset(tmp_path, nested=False, overlap=True)), tmp_path).check()
    assert not res["ok"]
    assert res["nested"] is False
    assert res["overlaps"]["evaluation&tuning"] == 1  # 이름은 달라도 내용(SHA-256)이 같음


def test_dataset_path_resolution(tmp_path: Path):
    assert dataset_path("fire", tmp_path) == tmp_path / "datasets" / "fire.local.yaml"
    assert dataset_path("x/my.yaml", tmp_path) == Path("x/my.yaml")
    with pytest.raises(ConfigError, match="example.yaml"):
        load_dataset("fire", tmp_path)


def test_example_config_is_valid_yaml():
    raw = yaml.safe_load((REPO / "configs/datasets/example.yaml").read_text())
    assert sorted(raw["calibration"]) == [50, 100, 200]
    assert raw["tuning"] and raw["evaluation"]


# --------------------------------------------------------------------------- #
# CLI / 파이프라인
# --------------------------------------------------------------------------- #
@pytest.fixture
def config_copy(config_dir: Path) -> Path:
    """configs 사본 (conftest config_dir: 개인 설정 제외, 테스트 프리셋 포함)."""
    return config_dir


def _fire_dataset(tmp_path: Path) -> Path:
    """fire 프리셋(calibration_num 50)에 맞는 데이터셋."""
    return _make_dataset(tmp_path / "data")


def test_compile_with_dataset_option(tmp_path: Path, config_copy: Path):
    ds = _fire_dataset(tmp_path)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--dataset", str(ds), "--out", str(tmp_path / "runs"), "--dry-run"])
    assert res.exit_code == 0, res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    rep = json.loads((run / "report.json").read_text())
    calib = rep["inputs"]["calib"]
    assert calib["source"] == "dataset" and calib["sampling"] == "list"
    assert calib["sampled"] == 50 and calib["seed"] is None
    assert calib["list"].endswith("cal_50.txt")
    assert rep["dataset"]["name"] == "fire" and rep["dataset"]["tuning"]["sha256"]
    samples = json.loads((run / "calib_samples.json").read_text())
    expected = (tmp_path / "data/lists/cal_50.txt").read_text().split()
    assert [Path(s).name for s in samples] == expected  # 목록 순서 그대로
    assert len(list((run / "calib_subset").iterdir())) == 50


def test_compile_uses_default_local_dataset(tmp_path: Path, config_copy: Path):
    ds = _fire_dataset(tmp_path)
    (config_copy / "datasets").mkdir(exist_ok=True)
    # 상대 경로가 원래 위치 기준이 되도록 절대 경로로 다시 쓴다
    raw = yaml.safe_load(ds.read_text())
    base = ds.parent
    raw["images_dir"] = str(base / raw["images_dir"])
    raw["calibration"] = {k: str(base / v) for k, v in raw["calibration"].items()}
    raw["tuning"], raw["evaluation"] = str(base / raw["tuning"]), str(base / raw["evaluation"])
    (config_copy / "datasets/fire.local.yaml").write_text(yaml.safe_dump(raw))
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--out", str(tmp_path / "runs"), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "데이터셋 설정 사용" in res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    assert json.loads((run / "report.json").read_text())["inputs"]["calib"]["source"] == "dataset"


def test_compile_without_images_uses_default_calib(tmp_path: Path, config_copy: Path):
    """이미지 입력이 없으면 기본 캘리브레이션 사진(캐시) + settings_no_data (ema·100)."""
    import numpy as np

    cache = tmp_path / "cache"
    calib = cache / "calib" / "coco128"
    calib.mkdir(parents=True)
    for i in range(120):
        cv2.imwrite(str(calib / f"{i:03d}.jpg"), np.full((48, 64, 3), i, np.uint8))
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "nodata",
                              "--out", str(tmp_path / "runs"), "--dry-run"],
                        env={"DXCONV_CACHE_DIR": str(cache)})
    assert res.exit_code == 0, res.output
    assert "기본 캘리브레이션 사진 사용" in res.output
    run = next((tmp_path / "runs" / "compile").iterdir())
    rep = json.loads((run / "report.json").read_text())
    assert rep["inputs"]["calib"]["source"] == "calib_dir"
    assert Path(rep["inputs"]["calib"]["dir"]) == calib.resolve()
    resolved = yaml.safe_load((run / "resolved.yaml").read_text())
    assert (resolved["compile"]["calibration_method"], resolved["compile"]["calibration_num"]) == ("ema", 100)


def test_default_calib_download_failure(tmp_path: Path, monkeypatch):
    from dxconv.compile import default_calib

    monkeypatch.setenv("DXCONV_CACHE_DIR", str(tmp_path))

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(default_calib.urllib.request, "urlretrieve", boom)
    with pytest.raises(DxconvError, match="--images / --calib-dir"):
        default_calib.default_calib_dir("https://example.invalid/x.zip")


def test_default_calib_unpacks_images_only(tmp_path: Path, monkeypatch):
    import zipfile

    from dxconv.compile import default_calib

    monkeypatch.setenv("DXCONV_CACHE_DIR", str(tmp_path / "c"))
    z = tmp_path / "src.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("set/images/a/1.jpg", b"x")
        zf.writestr("set/images/b/2.PNG", b"y")
        zf.writestr("set/labels/1.txt", b"0 0.5 0.5 1 1")
        zf.writestr("__MACOSX/set/._1.jpg", b"z")
    monkeypatch.setattr(default_calib.urllib.request, "urlretrieve",
                        lambda url, dst: Path(dst).write_bytes(z.read_bytes()))
    d = default_calib.default_calib_dir("https://example.invalid/set.zip")
    assert sorted(p.name for p in d.iterdir()) == ["1.jpg", "2.PNG"]
    assert default_calib.default_calib_dir("https://example.invalid/set.zip") == d  # 캐시 재사용


def test_calib_dir_and_dataset_conflict(tmp_path: Path, config_copy: Path):
    make_images(tmp_path / "calib", 60)
    onnx = make_onnx(tmp_path / "in.onnx")
    res = runner.invoke(app, ["compile", "--onnx", str(onnx), "--model", "fire",
                              "--calib-dir", str(tmp_path / "calib"),
                              "--dataset", str(_fire_dataset(tmp_path)),
                              "--out", str(tmp_path / "runs"), "--dry-run"])
    assert res.exit_code == 1
    assert "함께 쓸 수 없습니다" in res.output


def test_dataset_check_cli(tmp_path: Path, config_copy: Path):
    ok = runner.invoke(app, ["dataset", "check", str(_fire_dataset(tmp_path)), "--json"])
    assert ok.exit_code == 0, ok.output
    bad_root = tmp_path / "bad"
    bad = runner.invoke(app, ["dataset", "check",
                              str(_make_dataset(bad_root, overlap=True))])
    assert bad.exit_code == 1
    assert "공유" in bad.output
