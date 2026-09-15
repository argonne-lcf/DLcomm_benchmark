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
    """The unconfigured behaviour must not change."""

    def test_gpu_still_defaults_to_xccl(self):
        # Aurora relies on this default; changing it would silently
        # repoint every existing GPU run at another transport.
        assert tcb.resolve_transport("gpu") == "xccl"

    def test_xpu_still_defaults_to_xccl(self):
        assert tcb.resolve_transport("xpu") == "xccl"

    def test_cpu_still_defaults_to_gloo(self):
        assert tcb.resolve_transport("cpu") == "gloo"


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

    def test_empty_string_falls_back_to_default(self):
        # "" is falsy: treated as unset, not as an invalid transport.
        assert tcb.resolve_transport("gpu", "") == "xccl"


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
