"""기본 캘리브레이션 사진: 사용자가 이미지를 주지 않을 때 (pt 만 입력).

INT8 양자화에는 실제 사진이 반드시 필요하다. 실측(YOLO26s 2개 모델)에서 학습 데이터와 무관한 일반 사진
(coco128)으로 캘리브레이션해도 ema·100 기준 자기 데이터와 차이가 0~0.1%p 였다. 합성(노이즈·단색) 이미지는
-1%p ~ 붕괴라 쓰지 않는다.

받은 사진은 ~/.cache/dxconv/calib/<zip 이름>/ 에 두고 재사용한다 (DXCONV_CACHE_DIR 로 변경).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from dxconv import DxconvError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def cache_root() -> Path:
    return Path(os.environ.get("DXCONV_CACHE_DIR") or Path.home() / ".cache" / "dxconv")


def default_calib_dir(url: str, console: Any = None) -> Path:
    """url 의 zip 을 받아 이미지만 평평한 폴더로 풀고 그 경로를 반환 (이미 있으면 그대로)."""
    name = Path(url.split("?")[0]).stem or "default"
    dest = cache_root() / "calib" / name
    if dest.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in dest.iterdir()):
        return dest
    if console is not None:
        console.print(f"[cyan]기본 캘리브레이션 사진 받는 중:[/] {url}", highlight=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        zpath = Path(tmp) / "calib.zip"
        try:
            urllib.request.urlretrieve(url, zpath)  # noqa: S310 - 설정된 https 주소
        except Exception as e:  # noqa: BLE001
            raise DxconvError(
                f"기본 캘리브레이션 사진을 받지 못했습니다 ({e}). 인터넷 연결을 확인하거나 "
                "--images / --calib-dir 로 실제 사진 폴더를 지정하세요.") from e
        out = Path(tmp) / "images"
        out.mkdir()
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                p = Path(info.filename)
                if info.is_dir() or p.suffix.lower() not in IMAGE_EXTS or p.name.startswith("."):
                    continue
                with zf.open(info) as src, (out / p.name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        if not any(out.iterdir()):
            raise DxconvError(f"기본 캘리브레이션 zip 에 이미지가 없습니다: {url}")
        if dest.exists():
            shutil.rmtree(dest)
        out.rename(dest)
    return dest
