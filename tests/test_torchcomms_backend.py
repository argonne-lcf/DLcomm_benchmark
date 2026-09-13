"""Static conformance check: does TorchCommsDist match what DLcomm calls?

Runs without torchcomms installed -- it compares the adapter's method names
against the dist.* surface the collectives actually use, so a missing entry
point is caught in CPU CI rather than 20 minutes into an Aurora queue.
"""

import re
from pathlib import Path

import pytest

from dl_comm.comm import torchcomms_backend as tcb

SRC = Path(__file__).resolve().parents[1] / "dl_comm" / "comm" / "collectives.py"
MAIN = Path(__file__).resolve().parents[1] / "dl_comm" / "dl_comm_main.py"


def _dist_calls_used():
    text = SRC.read_text()
    return sorted(set(re.findall(r"\bdist\.([a-z_]+)", text)))


def _main_dist_calls_used():
    text = MAIN.read_text()
    return sorted(set(re.findall(r"\bdist\.([a-z_]+)", text)))


def test_adapter_covers_every_dist_call_the_main_loop_makes():
    """The driver calls dist.* too -- notably destroy_process_group at shutdown.

    Missing this crashes only at the very end of a run, after all the
    measurements, which is the most expensive possible place to fail.
    """
    used = _main_dist_calls_used()
    assert used, "found no dist.* calls in the main loop -- the grep is wrong"
    missing = [name for name in used if not hasattr(tcb.TorchCommsDist, name)]
    assert not missing, (
        f"TorchCommsDist is missing {missing}; dl_comm_main.py calls these. "
        f"Full surface used: {used}"
    )


def test_adapter_covers_every_dist_call_the_collectives_make():
    """Every dist.X used by a collective must exist on TorchCommsDist."""
    used = _dist_calls_used()
    assert used, "found no dist.* calls -- the grep is wrong"
    missing = [name for name in used if not hasattr(tcb.TorchCommsDist, name)]
    assert not missing, (
        f"TorchCommsDist is missing {missing}; collectives.py calls these via "
        f"dist. Full surface used: {used}"
    )


def test_module_imports_without_torchcomms_installed():
    """Importing the adapter must not require torchcomms to be present."""
    assert isinstance(tcb.is_available(), bool)


def test_requesting_backend_without_library_raises_actionable_error():
    if tcb.is_available():
        pytest.skip("torchcomms is installed here; the failure path needs it absent")
    with pytest.raises(RuntimeError) as e:
        tcb._require()
    msg = str(e.value)
    assert "torchcomms" in msg
    assert "ccl_backend" in msg, "error should name the config field to change"


def test_transport_defaults_by_device_type():
    assert tcb.resolve_transport("gpu") == "xccl"
    assert tcb.resolve_transport("xpu") == "xccl"
    assert tcb.resolve_transport("cpu") == "gloo"


def test_explicit_transport_is_validated_not_trusted():
    assert tcb.resolve_transport("gpu", "nccl") == "nccl"
    with pytest.raises(ValueError) as e:
        tcb.resolve_transport("gpu", "nccl-typo")
    assert "nccl-typo" in str(e.value)


def test_every_supported_transport_resolves():
    for t in tcb.SUPPORTED_TRANSPORTS:
        assert tcb.resolve_transport("gpu", t) == t


def test_group_records_its_global_ranks():
    g = tcb.TorchCommsGroup(object(), [4, 5, 6])
    assert g.ranks == [4, 5, 6]
    assert "4" in repr(g)


def test_work_wrapper_is_idempotent_and_safe_on_none():
    w = tcb._Work(None)
    assert w.wait() is None
    assert w.is_completed() is True


def test_recv_without_src_is_rejected():
    """torch.distributed allows a wildcard src; torchcomms does not.

    Silently receiving from the wrong peer would corrupt a correctness check,
    so the adapter must refuse rather than guess.
    """
    d = tcb.TorchCommsDist(comm=object())
    with pytest.raises(ValueError) as e:
        d.recv(tensor=object(), src=None)
    assert "src" in str(e.value)
    with pytest.raises(ValueError):
        d.irecv(tensor=object(), src=None)
