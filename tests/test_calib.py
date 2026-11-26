from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_images

from dxconv import DxconvError
from dxconv.compile.calib import make_subset, prepare_calib, sample_images, validate_calib_dir


def test_not_enough_images(tmp_path: Path):
    make_images(tmp_path / "c", 10)
    with pytest.raises(DxconvError, match="부족"):
        validate_calib_dir(tmp_path / "c", 20)


def test_missing_dir(tmp_path: Path):
    with pytest.raises(DxconvError, match="없습니다"):
        validate_calib_dir(tmp_path / "none", 1)


def test_broken_images_detected(tmp_path: Path):
    make_images(tmp_path / "c", 10, broken=2)
    info = validate_calib_dir(tmp_path / "c", 10)
    assert len(info.images) == 10
    assert len(info.broken) == 2
    assert all(p.name.startswith("broken_") for p in info.broken)


def test_broken_images_do_not_count(tmp_path: Path):
    make_images(tmp_path / "c", 9, broken=3)
    with pytest.raises(DxconvError, match="열리지 않는 파일"):
        validate_calib_dir(tmp_path / "c", 10)


def test_non_image_extensions_ignored(tmp_path: Path):
    make_images(tmp_path / "c", 5)
    (tmp_path / "c" / "labels.txt").write_text("x")
    make_images(tmp_path / "c" / "sub", 2, ext=".png")
    info = validate_calib_dir(tmp_path / "c", 7)
    assert len(info.images) == 7
    assert info.ignored == 1


def test_subset_property(tmp_path: Path):
    imgs = [tmp_path / f"{i:04d}.jpg" for i in range(300)]
    s50 = sample_images(imgs, 50, seed=0)
    s100 = sample_images(imgs, 100, seed=0)
    assert set(s50) <= set(s100)
    assert s100[:50] == s50


def test_sampling_is_order_independent_and_seeded(tmp_path: Path):
    imgs = [tmp_path / f"{i:04d}.jpg" for i in range(100)]
    a = sample_images(imgs, 20, seed=3)
    b = sample_images(list(reversed(imgs)), 20, seed=3)
    c = sample_images(imgs, 20, seed=4)
    assert a == b
    assert a != c


def test_make_subset_symlinks(tmp_path: Path):
    imgs = make_images(tmp_path / "c", 5)
    dest = make_subset(imgs[:3], tmp_path / "subset")
    files = sorted(dest.iterdir())
    assert len(files) == 3
    assert all(f.is_symlink() or f.is_file() for f in files)
    # 다시 만들면 기존 내용이 교체된다
    dest = make_subset(imgs[:2], tmp_path / "subset")
    assert len(list(dest.iterdir())) == 2


def test_prepare_calib(tmp_path: Path):
    make_images(tmp_path / "c", 30, broken=1)
    subset, info, samples = prepare_calib(tmp_path / "c", 20, 0, tmp_path / "subset")
    assert len(samples) == 20
    assert len(list(subset.iterdir())) == 20
    assert len(info.broken) == 1


# --------------------------------------------------------------------------- #
# host letterbox: Ultralytics 추론 전처리와 픽셀 단위 일치
# --------------------------------------------------------------------------- #
def _noise_image(path: Path, w: int, h: int, mode: str = "RGB", seed: int = 0) -> Path:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    ch = 3 if mode == "RGB" else 1
    arr = rng.integers(0, 256, size=(h, w, ch), dtype=np.uint8)
    Image.fromarray(arr.squeeze() if ch == 1 else arr, mode).save(path)
    return path


def _ultralytics_reference(src: Path, imgsz: tuple[int, int]):  # noqa: ANN202
    """Ultralytics BasePredictor.pre_transform 과 같은 방식 (정적 모델: auto=False).

    dxconv 구현과 독립적으로 LetterBox 를 기본 인자로 만들어 비교한다.
    """
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.patches import imread

    return LetterBox(imgsz, auto=False, scale_fill=False, stride=32)(image=imread(str(src)))


@pytest.mark.parametrize("w, h, imgsz, ext, mode", [
    (1280, 720, (640, 640), ".jpg", "RGB"),   # 가로로 긴 이미지 → 위아래 패딩
    (480, 800, (640, 640), ".png", "RGB"),    # 세로로 긴 이미지 → 좌우 패딩
    (333, 251, (640, 640), ".bmp", "RGB"),    # 작은 이미지 → 확대(scaleup) + 홀수 패딩
    (640, 640, (640, 640), ".jpg", "RGB"),    # 그대로
    (1000, 600, (384, 640), ".png", "RGB"),   # 정사각형이 아닌 입력 크기 (h=384, w=640)
    (500, 300, (640, 640), ".png", "L"),      # 흑백 → BGR 3채널로 읽힘
])
def test_letterbox_matches_ultralytics_pixelwise(
    tmp_path: Path, w: int, h: int, imgsz: tuple[int, int], ext: str, mode: str
):
    import cv2
    import numpy as np

    from dxconv.compile.calib import LetterboxSpec, make_letterboxed_subset

    src = _noise_image(tmp_path / f"src{ext}", w, h, mode)
    dest = make_letterboxed_subset([src], tmp_path / "subset", LetterboxSpec(*imgsz, 114))
    files = list(dest.iterdir())
    assert [f.suffix for f in files] == [".png"]

    ours = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    ref = _ultralytics_reference(src, imgsz)
    assert ours.shape == ref.shape == (imgsz[0], imgsz[1], 3)
    assert ours.dtype == ref.dtype == np.uint8
    assert np.array_equal(ours, ref)


def test_letterbox_padding_centered_114(tmp_path: Path):
    import cv2
    import numpy as np

    from dxconv.compile.calib import LetterboxSpec, make_letterboxed_subset

    src = tmp_path / "wide.png"
    cv2.imwrite(str(src), np.full((320, 640, 3), (10, 20, 30), np.uint8))  # BGR
    dest = make_letterboxed_subset([src], tmp_path / "subset", LetterboxSpec(640, 640, 114))
    out = cv2.imread(str(next(dest.iterdir())))
    # 640x320 → 640x320 그대로, 위아래 160 px 씩 114 패딩 (가운데 정렬)
    assert (out[:160] == 114).all() and (out[-160:] == 114).all()
    assert (out[160:480] == (10, 20, 30)).all()  # 채널 순서(BGR) 유지


def test_prepare_calib_host_letterbox(tmp_path: Path):
    from dxconv.compile.calib import LetterboxSpec

    make_images(tmp_path / "c", 12)
    subset, _, samples = prepare_calib(tmp_path / "c", 10, 0, tmp_path / "subset",
                                       letterbox=LetterboxSpec(64, 64))
    files = sorted(subset.iterdir())
    assert len(files) == 10 and all(f.suffix == ".png" and not f.is_symlink() for f in files)
    assert files[0].name == f"00000_{samples[0].stem}.png"


# --------------------------------------------------------------------------- #
# stratified 샘플링 (calib-dir 바로 아래 하위 폴더 = 클래스)
# --------------------------------------------------------------------------- #
def _fake_tree(root: Path, sizes: dict[str, int]) -> list[Path]:
    """파일을 만들지 않고 경로만 만든다 (sample_stratified 는 경로만 본다)."""
    out = []
    for cls, n in sizes.items():
        base = root if cls == "(root)" else root / cls
        out += [base / f"{cls}_{i:04d}.jpg" for i in range(n)]
    return out


def _counts(samples: list[Path], root: Path) -> dict[str, int]:
    from dxconv.compile.calib import class_of

    out: dict[str, int] = {}
    for p in samples:
        out[class_of(p, root)] = out.get(class_of(p, root), 0) + 1
    return out


def test_stratified_proportional(tmp_path: Path):
    from dxconv.compile.calib import sample_stratified

    imgs = _fake_tree(tmp_path, {"fire": 60, "normal": 20})
    s, groups = sample_stratified(imgs, tmp_path, 20, seed=0)
    assert _counts(s, tmp_path) == {"fire": 15, "normal": 5}
    assert set(groups) == {"fire", "normal"}
    # 50:50
    imgs = _fake_tree(tmp_path, {"fire": 50, "normal": 50})
    assert _counts(sample_stratified(imgs, tmp_path, 30, seed=1)[0], tmp_path) == {
        "fire": 15, "normal": 15}


@pytest.mark.parametrize("sizes", [
    {"fire": 50, "normal": 50},
    {"a": 7, "b": 130, "c": 41},
    {"fire": 383, "normal": 1},
])
def test_stratified_every_prefix_within_one(tmp_path: Path, sizes: dict[str, int]):
    """어떤 N 에서도 클래스별 개수가 비율 목표(N * n_c / 전체)와 1 개 이내."""
    from dxconv.compile.calib import sample_stratified

    imgs = _fake_tree(tmp_path, sizes)
    total = sum(sizes.values())
    full, _ = sample_stratified(imgs, tmp_path, total, seed=0)
    assert sorted(full) == sorted(imgs)  # 전체를 뽑으면 빠짐/중복 없음
    for n in range(1, total + 1):
        got = _counts(full[:n], tmp_path)
        for cls, size in sizes.items():
            assert abs(got.get(cls, 0) - n * size / total) < 1, (n, cls, got)


def test_stratified_subset_property(tmp_path: Path):
    from dxconv.compile.calib import sample_stratified

    imgs = _fake_tree(tmp_path, {"fire": 120, "normal": 80, "smoke": 37})
    s50, _ = sample_stratified(imgs, tmp_path, 50, seed=0)
    s100, _ = sample_stratified(imgs, tmp_path, 100, seed=0)
    assert set(s50) <= set(s100)
    assert s100[:50] == s50


def test_stratified_seeded_and_order_independent(tmp_path: Path):
    from dxconv.compile.calib import sample_stratified

    imgs = _fake_tree(tmp_path, {"fire": 50, "normal": 50})
    a, _ = sample_stratified(imgs, tmp_path, 20, seed=3)
    b, _ = sample_stratified(list(reversed(imgs)), tmp_path, 20, seed=3)
    c, _ = sample_stratified(imgs, tmp_path, 20, seed=4)
    assert a == b
    assert a != c


def test_stratified_nested_folders_use_top_level_class(tmp_path: Path):
    from dxconv.compile.calib import class_of

    assert class_of(tmp_path / "fire" / "cam1" / "x.jpg", tmp_path) == "fire"
    assert class_of(tmp_path / "x.jpg", tmp_path) == "(root)"


def test_prepare_calib_stratified_records_counts(tmp_path: Path):
    make_images(tmp_path / "c" / "fire", 30)
    make_images(tmp_path / "c" / "normal", 10)
    _, info, samples = prepare_calib(tmp_path / "c", 20, 0, tmp_path / "subset",
                                     sampling="stratified")
    assert info.sampling == info.sampling_requested == "stratified"
    assert info.class_totals == {"fire": 30, "normal": 10}
    assert info.class_counts == {"fire": 15, "normal": 5}
    assert len(samples) == 20 and not info.warnings


def test_prepare_calib_stratified_falls_back_to_random(tmp_path: Path):
    make_images(tmp_path / "c", 30)
    _, info, samples = prepare_calib(tmp_path / "c", 20, 0, tmp_path / "subset",
                                     sampling="stratified")
    assert info.sampling_requested == "stratified" and info.sampling == "random"
    assert any("하위 폴더가 없어" in w for w in info.warnings)
    assert samples == sample_images(info.images, 20, 0)
    assert info.class_counts == {"(root)": 20}


def test_prepare_calib_stratified_root_images_warn(tmp_path: Path):
    make_images(tmp_path / "c", 10)
    make_images(tmp_path / "c" / "fire", 10)
    _, info, _ = prepare_calib(tmp_path / "c", 10, 0, tmp_path / "subset",
                               sampling="stratified")
    assert info.sampling == "stratified"
    assert info.class_counts == {"(root)": 5, "fire": 5}
    assert any("(root)" in w for w in info.warnings)


def test_prepare_calib_random_mode(tmp_path: Path):
    make_images(tmp_path / "c" / "fire", 20)
    make_images(tmp_path / "c" / "normal", 20)
    _, info, samples = prepare_calib(tmp_path / "c", 10, 0, tmp_path / "subset",
                                     sampling="random")
    assert info.sampling == "random" and not info.warnings
    assert samples == sample_images(info.images, 10, 0)
    assert sum(info.class_counts.values()) == 10


def test_prepare_calib_unknown_sampling(tmp_path: Path):
    make_images(tmp_path / "c", 5)
    with pytest.raises(DxconvError, match="calib_sampling"):
        prepare_calib(tmp_path / "c", 5, 0, tmp_path / "subset", sampling="kmeans")
