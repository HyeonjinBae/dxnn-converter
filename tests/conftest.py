from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


# 테스트용 모델 프리셋: 저장소에는 모델 프리셋을 두지 않으므로, 프리셋 병합을 검증할 값을 여기서 만든다.
TEST_PRESETS = {"fire": "compile:\n  calibration_num: 50\n"}


@pytest.fixture(autouse=True)
def config_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """저장소 configs/ 사본 + 테스트 프리셋. CLI 도 DXCONV_CONFIG_DIR 로 이 사본을 쓴다.

    개인 설정(remote.yaml, datasets/*.local.yaml)은 복사하지 않는다.
    """
    import shutil

    dst = tmp_path_factory.mktemp("configs")
    shutil.copytree(REPO / "configs", dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("remote.yaml", "*.local.yaml", "*.bak"))
    (dst / "presets").mkdir(exist_ok=True)
    for name, body in TEST_PRESETS.items():
        (dst / "presets" / f"{name}.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("DXCONV_CONFIG_DIR", str(dst))
    return dst


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


def make_images(root: Path, n: int, broken: int = 0, ext: str = ".jpg") -> list[Path]:
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = root / f"img_{i:04d}{ext}"
        Image.new("RGB", (32, 24), (i % 256, 10, 20)).save(p)
        paths.append(p)
    for i in range(broken):
        p = root / f"broken_{i}.jpg"
        p.write_bytes(b"not an image")
        paths.append(p)
    return paths


def make_onnx(path: Path, shape: list[int | str] | None = None, opset: int = 17) -> Path:
    import onnx
    from onnx import TensorProto, helper

    shape = shape or [1, 3, 640, 640]
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, shape)
    node = helper.make_node("Identity", ["images"], ["output0"])
    graph = helper.make_graph([node], "g", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path
