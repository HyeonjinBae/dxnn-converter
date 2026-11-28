import io
import tarfile

from dxconv.remote.client import HostCfg, RemoteClient


class FakeClient(RemoteClient):
    def run(self, script, *, timeout=None, check=True, stdin=None):
        if stdin is not None and script.startswith("tar xf"):
            self.sent = stdin
        return ""


def test_sync_images_sends_symlink_targets(tmp_path):
    real = tmp_path / "real.jpg"
    real.write_bytes(b"jpegdata")
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "a.jpg").symlink_to(real)
    c = FakeClient("dev", HostCfg(hostname="h", username="u", workdir="/w"))
    c.sync_images("k", imgs, ["a.jpg"], "tuning")
    with tarfile.open(fileobj=io.BytesIO(c.sent)) as tar:
        m = tar.getmember("a.jpg")
        assert m.isfile()
        assert tar.extractfile(m).read() == b"jpegdata"


def test_ssh_hint_by_cause():
    from dxconv.remote.client import HostCfg, RemoteClient

    c = RemoteClient("npu", HostCfg(hostname="10.0.0.5", port=2222, username="user"))
    assert "ssh-copy-id -p 2222 user@10.0.0.5" in c.ssh_hint("user@10.0.0.5: Permission denied (publickey).")
    assert "port(2222" in c.ssh_hint("ssh: connect to host 10.0.0.5 port 2222: Operation timed out")
    assert "포트 2222" in c.ssh_hint("ssh: connect to host 10.0.0.5 port 2222: Connection refused")
    assert "yes" in c.ssh_hint("Host key verification failed.")
    assert c.ssh_hint("Segmentation fault") is None  # 장비 명령 자체의 실패는 ssh 안내 대상 아님


def test_run_raises_ssh_hint_on_255(monkeypatch):
    import subprocess

    import pytest

    from dxconv import DxconvError
    from dxconv.remote.client import HostCfg, RemoteClient

    def fake(*a, **k):
        return subprocess.CompletedProcess(a, 255, b"", b"user@h: Permission denied (publickey,password).")

    monkeypatch.setattr(subprocess, "run", fake)
    c = RemoteClient("npu", HostCfg(hostname="h", username="user"))
    with pytest.raises(DxconvError, match="ssh-copy-id"):
        c.check()
