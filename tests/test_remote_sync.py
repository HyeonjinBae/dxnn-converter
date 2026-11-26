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
