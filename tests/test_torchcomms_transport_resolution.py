"""Regression cover for resolve_transport() backend resolution.

DEFECT (found on Tara, jobs 8478 / 8479, 2026-10-24)
----------------------------------------------------
resolve_transport() returned the literal string "xccl" for any device_type in
(gpu, xpu). That is correct only on Aurora, where "gpu" means Intel XPU. On an
NVIDIA machine the xccl backend is not registered in the torchcomms
entry-point group, so tc.new_comm() raised

    ModuleNotFoundError: failed to find backend xccl,
    is it registered via entry_points.txt?

Layer 5 was therefore unrunnable on Tara at both scales. At 1 node it
presented as "rank 0 exited with code 1"; at 2 nodes one rank exited 1 and the
launcher then sat until the harness timeout, so the job log said rc=143 --
a hardcoded vendor string presenting as a hang, which is the expensive kind of
failure to diagnose.

The tests below pin the behaviour that the fix must keep:
  * resolution follows what is REGISTERED, not what the device is called
  * an explicit request for an unregistered backend fails here, with a
    message naming the registered set, rather than inside new_comm
  * device_type cpu never returns an accelerator transport
  * when entry points cannot be enumerated at all, the historical Aurora
    guess is preserved, so this change cannot regress Aurora

Run: python3 tests/test_torchcomms_transport_resolution.py
(plain unittest; the Tara dlcomm-env has no pytest)
"""

import unittest
from unittest import mock

import dl_comm.comm.torchcomms_backend as tcb


class _EP:
    """Stands in for an importlib.metadata EntryPoint (only .name is read)."""

    def __init__(self, name):
        self.name = name


def _with(registered, vendor):
    """Patch the two inputs resolve_transport() consults."""
    return (
        mock.patch.object(tcb, "registered_transports",
                          return_value=tuple(registered)),
        mock.patch.object(tcb, "_accelerator_vendor", return_value=vendor),
    )


class TestCudaBuild(unittest.TestCase):
    """A stock CUDA torchcomms wheel: this is exactly what Tara has."""

    REGISTERED = ("fake", "gloo", "nccl", "nccl-lazy")

    def test_gpu_resolves_to_nccl_not_xccl(self):
        # THE REGRESSION. Pre-fix this returned "xccl" and the run died.
        a, b = _with(self.REGISTERED, "cuda")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "nccl")

    def test_cpu_does_not_get_an_accelerator_transport(self):
        # A GPU machine can still be asked for a host-memory run. Returning
        # nccl there would silently move the traffic onto the device and the
        # reported "cpu" bandwidth would be a GPU number.
        a, b = _with(self.REGISTERED, "cuda")
        with a, b:
            self.assertEqual(tcb.resolve_transport("cpu"), "gloo")

    def test_explicit_xccl_fails_early_and_names_what_is_available(self):
        a, b = _with(self.REGISTERED, "cuda")
        with a, b:
            with self.assertRaises(ValueError) as cm:
                tcb.resolve_transport("gpu", "xccl")
        msg = str(cm.exception)
        self.assertIn("xccl", msg)
        self.assertIn("nccl", msg)  # tells the user what they CAN use

    def test_explicit_registered_request_is_honoured(self):
        a, b = _with(self.REGISTERED, "cuda")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu", "nccl"), "nccl")

    def test_unknown_name_still_rejected(self):
        a, b = _with(self.REGISTERED, "cuda")
        with a, b:
            with self.assertRaises(ValueError):
                tcb.resolve_transport("gpu", "not-a-transport")


class TestAuroraUnchanged(unittest.TestCase):
    """The fix must not change what Aurora does."""

    def test_xpu_with_xccl_registered_picks_xccl(self):
        a, b = _with(("xccl", "gloo"), "xpu")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "xccl")
            self.assertEqual(tcb.resolve_transport("xpu"), "xccl")

    def test_unenumerable_entry_points_fall_back_to_historical_guess(self):
        # If introspection fails we must not invent a different answer than
        # the code shipped with; () means "unknown", not "none registered".
        a, b = _with((), "xpu")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "xccl")
            self.assertEqual(tcb.resolve_transport("cpu"), "gloo")


class TestRocm(unittest.TestCase):
    def test_rocm_prefers_rccl(self):
        a, b = _with(("rccl", "gloo"), "rocm")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "rccl")

    def test_rocm_without_rccl_falls_through_to_gloo(self):
        a, b = _with(("gloo",), "rocm")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "gloo")


class TestNoAccelerator(unittest.TestCase):
    def test_cpu_only_machine(self):
        a, b = _with(("gloo", "fake"), "none")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "gloo")


class TestEnumeration(unittest.TestCase):
    """registered_transports() must never raise -- it is called on every run."""

    def test_returns_names(self):
        with mock.patch("importlib.metadata.entry_points",
                        return_value=[_EP("nccl"), _EP("gloo")]):
            self.assertEqual(tcb.registered_transports(), ("gloo", "nccl"))

    def test_enumeration_failure_is_unknown_not_empty_crash(self):
        with mock.patch("importlib.metadata.entry_points",
                        side_effect=RuntimeError("boom")):
            self.assertEqual(tcb.registered_transports(), ())

    def test_real_environment_is_self_consistent(self):
        # Not a mock: whatever this machine actually has must resolve to
        # something that machine can actually load.
        reg = tcb.registered_transports()
        got = tcb.resolve_transport("gpu")
        if reg:
            self.assertIn(got, reg,
                          f"resolved {got!r} which is not registered: {reg}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
