"""Transport selection for the torchcomms backend.

Regression cover for the Aurora-only assumption: resolve_transport() mapped
every GPU to xccl, and nothing in the config could override it, so
ccl_backend: torchcomms was unreachable on NVIDIA and AMD hardware.

These tests import the backend module directly by path rather than through
``dl_comm.comm``, whose package __init__ pulls in mpi4py. Transport selection
is pure logic with no MPI involvement, so it stays runnable in CPU CI.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SPEC = json.loads((ROOT / "dl_comm" / "config" / "config_spec.json").read_text())


def _load_backend():
    """Import torchcomms_backend without triggering dl_comm.comm.__init__."""
    path = ROOT / "dl_comm" / "comm" / "torchcomms_backend.py"
    spec = importlib.util.spec_from_file_location("_tcb_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tcb = _load_backend()


class TestResolveTransportDefaults:
    """Defaults are chosen from the REGISTERED transports, not guessed.

    These tests used to assert a hardcoded ``xccl`` for gpu/xpu. That was
    the Aurora-only behaviour the function was later fixed to remove: on an
    NVIDIA machine xccl is not registered, and returning it made every
    torchcomms run die inside ``tc.new_comm``. The tests kept asserting the
    old contract, so they failed on any machine that is not Aurora --
    including a plain CPU box, where NO transports are registered at all.

    They now pin the registration state explicitly instead of depending on
    whatever the host happens to have installed, so the behaviour under
    test is the resolution LOGIC rather than the machine.
    """

    def test_xpu_prefers_xccl_when_registered(self, monkeypatch):
        """Aurora's contract, stated as a controlled condition."""
        monkeypatch.setattr(tcb, "enumerate_transports",
                            lambda: ("ok", ("xccl", "gloo")))
        monkeypatch.setattr(tcb, "_accelerator_vendor", lambda: "xpu")
        assert tcb.resolve_transport("xpu") == "xccl"

    def test_gpu_prefers_nccl_when_registered(self, monkeypatch):
        """On NVIDIA, nccl -- not xccl -- is the correct default."""
        monkeypatch.setattr(tcb, "enumerate_transports",
                            lambda: ("ok", ("nccl", "gloo")))
        monkeypatch.setattr(tcb, "_accelerator_vendor", lambda: "cuda")
        assert tcb.resolve_transport("gpu") == "nccl"

    def test_cpu_defaults_to_gloo_when_registered(self, monkeypatch):
        monkeypatch.setattr(tcb, "enumerate_transports",
                            lambda: ("ok", ("nccl", "gloo")))
        assert tcb.resolve_transport("cpu") == "gloo"

    def test_accelerator_with_no_vendor_transport_fails_closed(
            self, monkeypatch):
        """The anti-regression this function exists for.

        Silently returning gloo on an accelerator would run the whole
        benchmark over the host network and report it as a normal result.
        """
        monkeypatch.setattr(tcb, "enumerate_transports",
                            lambda: ("ok", ("gloo",)))
        monkeypatch.setattr(tcb, "_accelerator_vendor", lambda: "cuda")
        with pytest.raises((RuntimeError, ValueError)):
            tcb.resolve_transport("gpu")

    def test_empty_registration_fails_closed(self, monkeypatch):
        """No registered backends is an incomplete install, not a default."""
        monkeypatch.setattr(tcb, "enumerate_transports", lambda: ("ok", ()))
        with pytest.raises(RuntimeError):
            tcb.resolve_transport("gpu")


class TestResolveTransportOverride:
    """An explicit request wins over the device-based guess."""

    @pytest.mark.parametrize("transport", ["nccl", "rccl", "gloo", "ncclx"])
    def test_request_overrides_device_default(self, transport):
        assert tcb.resolve_transport("gpu", transport) == transport

    def test_request_is_case_insensitive(self):
        assert tcb.resolve_transport("gpu", "NCCL") == "nccl"

    def test_nccl_reachable_on_gpu(self):
        # The exact combination that was impossible before: a GPU device_type
        # carrying NCCL rather than XCCL.
        assert tcb.resolve_transport("gpu", "nccl") == "nccl"

    def test_unknown_transport_raises(self):
        with pytest.raises(ValueError, match="Unknown torchcomms transport"):
            tcb.resolve_transport("gpu", "not-a-transport")

    def test_error_names_the_valid_options(self):
        with pytest.raises(ValueError) as excinfo:
            tcb.resolve_transport("gpu", "nccl2")
        for name in ("nccl", "xccl", "gloo"):
            assert name in str(excinfo.value)

    def test_empty_string_falls_back_to_default(self, monkeypatch):
        """"" is falsy: treated as unset, not as an invalid transport.

        The assertion was "" -> xccl, which encoded the old Aurora-only
        default and failed anywhere xccl is not registered. What actually
        matters is that an empty string takes the DEFAULT path rather than
        the explicit-request path, so pin the registration state and check
        it resolves to the vendor default.
        """
        monkeypatch.setattr(tcb, "enumerate_transports",
                            lambda: ("ok", ("nccl", "gloo")))
        monkeypatch.setattr(tcb, "_accelerator_vendor", lambda: "cuda")
        assert tcb.resolve_transport("gpu", "") == "nccl"

    def test_empty_string_is_not_treated_as_an_unknown_transport(
            self, monkeypatch):
        """The failure mode this guards: "" raising "Unknown transport"."""
        monkeypatch.setattr(tcb, "enumerate_transports",
                            lambda: ("ok", ("xccl", "gloo")))
        monkeypatch.setattr(tcb, "_accelerator_vendor", lambda: "xpu")
        assert tcb.resolve_transport("xpu", "") == "xccl"


class TestConfigSpec:
    """The validator vocabulary must match what the backend accepts."""

    def test_spec_declares_transport(self):
        assert "transport" in SPEC, "config_spec.json is missing 'transport'"

    def test_spec_matches_backend_supported_transports(self):
        assert sorted(SPEC["transport"]) == sorted(tcb.SUPPORTED_TRANSPORTS)

    def test_every_spec_transport_resolves(self):
        for transport in SPEC["transport"]:
            assert tcb.resolve_transport("gpu", transport) == transport

    def test_torchcomms_is_a_valid_pytorch_backend(self):
        assert "torchcomms" in SPEC["backend"]["pytorch"]
