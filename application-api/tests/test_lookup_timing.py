import httpx
import pytest

from app.config import Settings
from app.stresser_client import StresserClient, StresserUnavailable


def client(settings, error=None):
    def handler(request):
        if error:
            raise error
        return httpx.Response(200, json={"stress_level": 0, "cpu_workers": 0})

    return StresserClient(settings, httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_timing_is_off_by_default(tmp_path):
    stresser = client(Settings())
    assert stresser.timing is None
    assert await stresser.get_level() == 0
    await stresser.aclose()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_timing_file_records_each_lookup(tmp_path):
    path = tmp_path / "lookups.csv"
    ok = client(Settings(lookup_timing_file=str(path)))
    assert await ok.get_level() == 0
    await ok.aclose()
    failing = client(Settings(lookup_timing_file=str(path)), httpx.ReadTimeout("secret-host"))
    with pytest.raises(StresserUnavailable):
        await failing.get_level()
    await failing.aclose()
    lines = [line.split(",") for line in path.read_text(encoding="ascii").splitlines()]
    assert [outcome for _, _, outcome in lines] == ["ok", "ReadTimeout"]
    assert all(int(end) > 0 and int(duration) >= 0 for end, duration, _ in lines)
    assert "secret" not in path.read_text(encoding="ascii")
