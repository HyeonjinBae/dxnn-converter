"""캘리브레이션 데이터셋 검증 · 시드 고정 샘플링 · 부분집합 생성.

부분집합은 두 가지 방식으로 만든다.
- 심볼릭 링크 (make_subset): 원본 그대로. DX-COM 이 resize/letterbox 까지 처리할 때.
- host letterbox (make_letterboxed_subset): Ultralytics LetterBox 로 미리 처리한 PNG.
  DX-COM 2.5.0 의 Resize_Pad 가 가운데 패딩(pad_location: center)을 지원하지 않아서
  학습/추론과 같은 letterbox 를 호스트에서 적용한다.
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dxconv import DxconvError

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


SamplingMode = str  # "random" | "stratified"

#: calib-dir 바로 아래(하위 폴더 밖)에 있는 이미지의 클래스 이름 (stratified)
ROOT_CLASS = "(root)"


@dataclass
class CalibInfo:
    root: Path
    images: list[Path] = field(default_factory=list)  # 정상 이미지 (정렬됨)
    broken: list[Path] = field(default_factory=list)  # 열리지 않는 이미지
    ignored: int = 0  # 이미지 확장자가 아닌 파일 수
    # 샘플링 결과 (prepare_calib 이 채움)
    sampling_requested: str | None = None
    sampling: str | None = None  # 실제로 쓴 방식 (stratified 요청이어도 클래스가 없으면 random)
    class_totals: dict[str, int] = field(default_factory=dict)  # 클래스별 정상 이미지 수
    class_counts: dict[str, int] = field(default_factory=dict)  # 클래스별 샘플 수
    warnings: list[str] = field(default_factory=list)


def scan_images(root: Path) -> tuple[list[Path], int]:
    images: list[Path] = []
    ignored = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
        else:
            ignored += 1
    return sorted(images, key=lambda x: x.relative_to(root).as_posix()), ignored


def is_readable_image(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify 후에는 재오픈 필요
            im.load()
        return True
    except Exception:  # noqa: BLE001 - PIL 예외 종류 다양
        return False


def validate_calib_dir(root: str | Path, required: int, check_open: bool = True) -> CalibInfo:
    """캘리브레이션 폴더를 검증한다.

    - 이미지 확장자(jpg/jpeg/png/bmp) 파일만 사용 (하위 폴더 포함)
    - 열리지 않는 파일은 broken 목록으로 분리
    - 정상 이미지 수 < required 이면 DxconvError
    """
    root = Path(root)
    if not root.is_dir():
        raise DxconvError(f"캘리브레이션 폴더가 없습니다: {root}")
    images, ignored = scan_images(root)
    info = CalibInfo(root=root, ignored=ignored)
    for p in images:
        if not check_open or is_readable_image(p):
            info.images.append(p)
        else:
            info.broken.append(p)
    if len(info.images) < required:
        msg = (
            f"캘리브레이션 이미지가 부족합니다: 정상 {len(info.images)} 장 < "
            f"calibration_num {required} 장 (폴더: {root})"
        )
        if info.broken:
            msg += f"\n  열리지 않는 파일 {len(info.broken)} 개: " + ", ".join(
                str(p) for p in info.broken[:10]
            )
        raise DxconvError(msg)
    return info


def sample_images(images: list[Path], n: int, seed: int) -> list[Path]:
    """시드 고정 샘플링. 전체 정렬 → 시드로 셔플 → 앞에서 n 개.

    같은 시드라면 n=50 결과는 항상 n=100 결과의 앞부분(부분집합)이 된다.
    """
    if n > len(images):
        raise DxconvError(f"샘플 수({n}) 가 이미지 수({len(images)}) 보다 많습니다.")
    ordered = sorted(images, key=lambda p: p.as_posix())
    random.Random(seed).shuffle(ordered)
    return ordered[:n]


def class_of(path: Path, root: Path) -> str:
    """calib-dir 바로 아래 하위 폴더 이름을 클래스로 본다. 바로 아래 파일은 ROOT_CLASS."""
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else ROOT_CLASS


def group_by_class(images: list[Path], root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in images:
        groups.setdefault(class_of(p, root), []).append(p)
    return groups


def stratified_order(groups: dict[str, list[Path]], seed: int) -> list[Path]:
    """클래스 비율을 유지하는 전체 샘플 순서.

    - 클래스마다 (seed, 클래스 이름)으로 고정 셔플한다.
    - k 번째 자리에는 '목표 개수(n_c * k / 전체) - 이미 뽑은 개수'가 가장 큰 클래스를 놓는다
      (동점이면 이미지가 많은 클래스 → 이름 순).
    순서가 N 과 무관하게 정해지므로, 앞에서 N 개를 자르면 어떤 N 에서도 클래스별 개수가
    비율 목표와 1 개 이내로 맞고, N=50 샘플은 항상 N=100 샘플의 앞부분이 된다.
    """
    shuffled: dict[str, list[Path]] = {}
    for cls in sorted(groups):
        items = sorted(groups[cls], key=lambda p: p.as_posix())
        random.Random(f"{seed}:{cls}").shuffle(items)
        shuffled[cls] = items
    sizes = {c: len(v) for c, v in shuffled.items()}
    total = sum(sizes.values())
    taken = dict.fromkeys(shuffled, 0)
    order: list[Path] = []
    for k in range(1, total + 1):
        # 정수 비교: n_c * k / total - taken_c  →  n_c * k - taken_c * total
        best = max(
            (c for c in shuffled if taken[c] < sizes[c]),
            key=lambda c: (sizes[c] * k - taken[c] * total, sizes[c], [-ord(ch) for ch in c]),
        )
        order.append(shuffled[best][taken[best]])
        taken[best] += 1
    return order


def sample_stratified(
    images: list[Path], root: Path, n: int, seed: int
) -> tuple[list[Path], dict[str, list[Path]]]:
    """하위 폴더(클래스) 비율대로 n 개를 뽑는다. (샘플, 클래스별 이미지) 반환."""
    if n > len(images):
        raise DxconvError(f"샘플 수({n}) 가 이미지 수({len(images)}) 보다 많습니다.")
    groups = group_by_class(images, root)
    return stratified_order(groups, seed)[:n], groups


def make_subset(samples: list[Path], dest: str | Path) -> Path:
    """샘플을 dest 폴더에 심볼릭 링크로 만든다 (실패 시 복사). dest 경로 반환."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for i, src in enumerate(samples):
        link = dest / f"{i:05d}_{src.name}"
        try:
            os.symlink(src.resolve(), link)
        except OSError:
            shutil.copy2(src, link)
    return dest


@dataclass
class LetterboxSpec:
    """host letterbox 대상 크기와 패딩 값."""

    height: int
    width: int
    pad_value: int = 114


def ultralytics_letterbox(spec: LetterboxSpec) -> Any:
    """Ultralytics 추론(정적 shape 모델)과 같은 설정의 LetterBox.

    BasePredictor.pre_transform 에서 ONNX 등 정적 모델은 auto=False, scale_fill=False 이고
    나머지는 기본값(scaleup=True, center=True, INTER_LINEAR)을 쓴다.
    """
    from ultralytics.data.augment import LetterBox

    return LetterBox(
        new_shape=(spec.height, spec.width),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        padding_value=spec.pad_value,
    )


def read_bgr(path: Path) -> Any:
    """Ultralytics 와 같은 방식으로 이미지를 읽는다 (OpenCV, BGR, HWC, uint8)."""
    from ultralytics.utils.patches import imread

    img = imread(str(path))
    if img is None:
        raise DxconvError(f"캘리브레이션 이미지를 읽을 수 없습니다: {path}")
    return img


def make_letterboxed_subset(samples: list[Path], dest: str | Path, spec: LetterboxSpec) -> Path:
    """샘플을 Ultralytics LetterBox 로 처리해 dest 폴더에 PNG(무손실, BGR)로 저장한다."""
    import cv2

    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    lb = ultralytics_letterbox(spec)
    for i, src in enumerate(samples):
        out = lb(image=read_bgr(src))
        target = dest / f"{i:05d}_{src.stem}.png"
        if not cv2.imwrite(str(target), out):
            raise DxconvError(f"letterbox 이미지를 저장하지 못했습니다: {target}")
    return dest


def letterbox_impl_info() -> dict[str, Any]:
    """report 에 남길 host letterbox 구현 정보."""
    try:
        import ultralytics

        ver = ultralytics.__version__
    except Exception:  # noqa: BLE001
        ver = None
    return {
        "impl": "ultralytics.data.augment.LetterBox",
        "ultralytics_version": ver,
        "args": {"auto": False, "scale_fill": False, "scaleup": True, "center": True},
        "reader": "ultralytics.utils.patches.imread (BGR)",
        "format": "png",
    }


def prepare_calib(
    root: str | Path, num: int, seed: int, dest: str | Path, check_open: bool = True,
    letterbox: LetterboxSpec | None = None, sampling: SamplingMode = "random",
) -> tuple[Path, CalibInfo, list[Path]]:
    """검증 → 샘플링 → 부분집합 생성. letterbox 가 있으면 host letterbox PNG 로 만든다.

    sampling:
      random     — 전체 정렬 → seed 셔플 → 앞에서 num 개 (sample_images)
      stratified — calib-dir 바로 아래 하위 폴더를 클래스로 보고 비율대로 (sample_stratified).
                   하위 폴더가 없으면 random 으로 동작하고 info.warnings 에 경고를 남긴다.
    샘플링 방식과 클래스별 개수는 info.sampling / class_totals / class_counts 에 기록된다.
    """
    if sampling not in ("random", "stratified"):
        raise DxconvError(f"알 수 없는 calib_sampling: {sampling} (random | stratified)")
    info = validate_calib_dir(root, num, check_open=check_open)
    root = Path(root)
    groups = group_by_class(info.images, root)
    info.sampling_requested = sampling
    if sampling == "stratified" and set(groups) == {ROOT_CLASS}:
        info.warnings.append(
            f"calib_sampling=stratified 이지만 {root} 아래에 클래스 하위 폴더가 없어 "
            "random 샘플링으로 진행합니다."
        )
        sampling = "random"
    if sampling == "stratified":
        if ROOT_CLASS in groups:
            info.warnings.append(
                f"{root} 바로 아래 이미지 {len(groups[ROOT_CLASS])} 장은 "
                f"'{ROOT_CLASS}' 클래스로 따로 샘플링합니다."
            )
        samples, _ = sample_stratified(info.images, root, num, seed)
    else:
        samples = sample_images(info.images, num, seed)
    info.sampling = sampling
    info.class_totals = {c: len(v) for c, v in sorted(groups.items())}
    sampled = group_by_class(samples, root)
    info.class_counts = {c: len(sampled.get(c, [])) for c in info.class_totals}
    if letterbox is not None:
        return make_letterboxed_subset(samples, dest, letterbox), info, samples
    return make_subset(samples, dest), info, samples


def prepare_calib_from_list(
    images: list[Path], dest: str | Path, check_open: bool = True,
    letterbox: LetterboxSpec | None = None, root: Path | None = None,
) -> tuple[Path, CalibInfo, list[Path]]:
    """고정 목록(데이터셋 설정의 calibration 목록)으로 부분집합을 만든다. 샘플링하지 않는다."""
    root = Path(root) if root else (images[0].parent if images else Path("."))
    info = CalibInfo(root=root)
    for p in images:
        if not check_open or is_readable_image(p):
            info.images.append(p)
        else:
            info.broken.append(p)
    if info.broken:
        raise DxconvError(
            f"캘리브레이션 목록에 열리지 않는 이미지 {len(info.broken)} 개가 있습니다: "
            + ", ".join(str(p) for p in info.broken[:10])
        )
    samples = list(info.images)
    info.sampling_requested = info.sampling = "list"
    groups = group_by_class(samples, root)
    info.class_totals = {c: len(v) for c, v in sorted(groups.items())}
    info.class_counts = dict(info.class_totals)
    subset = (make_letterboxed_subset(samples, dest, letterbox) if letterbox is not None
              else make_subset(samples, dest))
    return subset, info, samples

