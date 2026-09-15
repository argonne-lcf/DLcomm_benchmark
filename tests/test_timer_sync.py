"""Timer synchronization and barrier placement (fix #5).

See ``docs/fixes/05-timing-and-statistics.md``.
"""

import importlib
import time

import pytest

from dl_comm.timer import timer, reset_times, set_sync_device, TIMES

# `dl_comm.timer.timer` as an attribute resolves to the re-exported FUNCTION,
# not the submodule, because dl_comm/timer/__init__.py does
# `from .timer import timer`. Load the submodule explicitly so monkeypatching
# targets the right object.
timer_mod = importlib.import_module("dl_comm.timer.timer")


@pytest.fixture(autouse=True)
def clean_times():
    reset_times()
    set_sync_device(None, enabled=True)
    yield
    reset_times()
    set_sync_device(None, enabled=True)


class FakeDevice:
    def __init__(self, dev_type="xpu"):
        self.type = dev_type


def test_timer_records_elapsed():
    with timer("unit"):
        time.sleep(0.01)
    assert len(TIMES["unit"]) == 1
    assert TIMES["unit"][0] >= 0.01


def test_timer_records_even_when_body_raises():
    """A collective that throws must not silently drop its sample."""
    with pytest.raises(ValueError):
        with timer("boom"):
            raise ValueError("collective failed")
    assert len(TIMES["boom"]) == 1


def test_sync_is_invoked_at_both_ends(monkeypatch):
    """The queue must drain before the start stamp and before the stop stamp.

    Without this the timed region measures kernel *enqueue*, not completion.
    """
    calls = []
    monkeypatch.setattr(timer_mod, "sync_device", lambda: calls.append(len(calls)))
    with timer_mod.timer("synced"):
        pass
    assert len(calls) == 2, f"expected sync before and after, got {len(calls)}"


def test_sync_can_be_disabled_for_cpu_regions(monkeypatch):
    calls = []
    monkeypatch.setattr(timer_mod, "sync_device", lambda: calls.append(1))
    with timer_mod.timer("host-only", sync=False):
        pass
    assert calls == [], "sync=False must not touch the device"


def test_sync_device_is_a_noop_without_a_device():
    """No registered device must not raise on a CPU-only machine."""
    set_sync_device(None)
    timer_mod.sync_device()  # must not raise


def test_sync_device_noop_when_disabled():
    set_sync_device(FakeDevice("xpu"), enabled=False)
    timer_mod.sync_device()  # must not raise and must not try to touch xpu


def test_multiple_iterations_accumulate():
    for _ in range(5):
        with timer("(Within-Group-0)"):
            pass
    assert len(TIMES["(Within-Group-0)"]) == 5


def test_reset_clears_all_labels():
    with timer("a"):
        pass
    with timer("b"):
        pass
    reset_times()
    assert len(TIMES) == 0
