from __future__ import annotations

from dxconv.env.compat import check_compat, select_dxcom_version
from dxconv.env.detect import EnvInfo, parse_dxrt_status, parse_version

COMPAT = {"dxcom": {"2.5.0": {"dxnn_format": 9, "min_dxrt": "3.5.0"},
                    "2.4.1-rc.1": {"dxnn_format": None, "min_dxrt": None}}}


def test_parse_version():
    assert parse_version("DX-COM v2.5.0 (build 123)") == "2.5.0"
    assert parse_version("dx_com 2.4.1-rc.1") == "2.4.1-rc.1"
    assert parse_version("nothing") is None


def test_select_version_priority():
    env = EnvInfo(dxcom=[{"source": "pip", "version": "2.4.1-rc.1", "path": None},
                         {"source": "path", "version": "2.5.0", "path": "x"}])
    assert select_dxcom_version("2.2.0", env, "2.5.0") == ("2.2.0", "cli")
    assert select_dxcom_version(None, env, "2.2.0") == ("2.2.0", "config")  # 목표 버전 우선
    assert select_dxcom_version(None, env, None) == ("2.5.0", "detected")
    assert select_dxcom_version(None, EnvInfo(), "2.5.0") == ("2.5.0", "config")


def test_non_reference_version_warns():
    res = check_compat("2.4.1-rc.1", EnvInfo(), COMPAT)
    assert any("기준 버전" in w for w in res.warnings)


def test_old_dxrt_warns():
    env = EnvInfo(dxcom=[{"source": "pip", "version": "2.5.0", "path": None}],
                  dxrt={"version": "3.0.0"}, npu_present=True)
    res = check_compat(None, env, COMPAT)
    assert res.version == "2.5.0"
    assert any("DX-RT" in w and "업그레이드" in w for w in res.warnings)


def test_new_dxrt_ok():
    env = EnvInfo(dxcom=[{"source": "pip", "version": "2.5.0", "path": None}],
                  dxrt={"version": "3.5.1"}, npu_present=True)
    assert check_compat(None, env, COMPAT).warnings == []


def test_parse_dxrt_status():
    text = "DXRT v3.5.0\nDevice 0: driver version v1.7.1\nFirmware version: v2.1.0\n"
    info = parse_dxrt_status(text)
    assert info == {"version": "3.5.0", "driver": "1.7.1", "firmware": "2.1.0"}


def test_version_for_dxrt(config_dir):
    from dxconv.env.compat import load_compat, version_for_dxrt

    compat = load_compat(config_dir)
    assert version_for_dxrt("3.5.0", compat) == ("2.5.0", 9)
    assert version_for_dxrt("3.6.1", compat) == ("2.5.0", 9)
    assert version_for_dxrt("3.2.0", compat) == ("2.4.1", 8)
    assert version_for_dxrt("3.1.0", compat) == ("2.4.1", 8)
    import pytest

    from dxconv import DxconvError
    with pytest.raises(DxconvError):
        version_for_dxrt("3.0.0", compat)
