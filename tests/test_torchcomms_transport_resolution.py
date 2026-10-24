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

REVIEW FIX (20261024T071020Z-dlcomm3c)
--------------------------------------
The independent review found that the first version of this fix was not
fail-closed: on an XPU machine whose torchcomms build did not register xccl,
resolution fell through to gloo and the benchmark silently ran over the host
network, reporting host bandwidth as Aurora fabric bandwidth. The tests in
TestFailClosedOnAccelerator pin the corrected behaviour, and
TestEnumerationStatus pins the "enumeration failed" vs "enumeration found
nothing" distinction those two cases now require.

Run: python3 tests/test_torchcomms_transport_resolution.py
(plain unittest; the Tara dlcomm-env has no pytest)
"""

import unittest
from unittest import mock

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_backend():
    """Import torchcomms_backend without triggering dl_comm.comm.__init__.

    Matches the loader in tests/test_torchcomms_transport.py. Importing
    ``dl_comm.comm.torchcomms_backend`` normally would execute the package
    __init__, which pulls in omegaconf and the rest of the runtime stack --
    none of which this module needs. Loading the file directly keeps the
    suite runnable on a bare interpreter (Tara's login nodes have only
    cray-python, with no omegaconf).
    """
    path = ROOT / "dl_comm" / "comm" / "torchcomms_backend.py"
    spec = importlib.util.spec_from_file_location("_tcb_resolution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tcb = _load_backend()


class _EP:
    """Stands in for an importlib.metadata EntryPoint (only .name is read)."""

    def __init__(self, name):
        self.name = name


def _with(registered, vendor, status="ok"):
    """Patch the two inputs resolve_transport() consults.

    resolve_transport() reads enumerate_transports(), which returns
    (status, names). Patching that -- rather than registered_transports() --
    is what makes these tests exercise the real code path.
    """
    return (
        mock.patch.object(tcb, "enumerate_transports",
                          return_value=(status, tuple(registered))),
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


class TestPolicyConsistency(unittest.TestCase):
    """Auto-selection must never return a name explicit validation rejects.

    REVIEW FIX: nccl-lazy was in _VENDOR_PREFERENCE['cuda'] but missing from
    SUPPORTED_TRANSPORTS, so a machine registering only nccl-lazy would
    auto-select a transport that `transport='nccl-lazy'` refused as unknown.
    """

    def test_every_preference_entry_is_supported(self):
        for vendor, prefs in tcb._VENDOR_PREFERENCE.items():
            for name in prefs:
                self.assertIn(
                    name, tcb.SUPPORTED_TRANSPORTS,
                    f"_VENDOR_PREFERENCE[{vendor!r}] offers {name!r} but "
                    f"SUPPORTED_TRANSPORTS rejects it")

    def test_every_native_transport_is_supported(self):
        for vendor, names in tcb._NATIVE_TRANSPORTS.items():
            for name in names:
                self.assertIn(name, tcb.SUPPORTED_TRANSPORTS,
                              f"_NATIVE_TRANSPORTS[{vendor!r}] -> {name!r}")

    def test_autoselected_nccl_lazy_would_be_accepted_explicitly(self):
        a, b = _with(("nccl-lazy", "gloo"), "cuda")
        with a, b:
            got = tcb.resolve_transport("gpu")
            self.assertEqual(got, "nccl-lazy")
        # The same name, passed explicitly, must not be rejected.
        a, b = _with(("nccl-lazy", "gloo"), "cuda")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu", got), got)


class TestFailClosedOnAccelerator(unittest.TestCase):
    """An accelerator run must never silently downgrade to the host network.

    REVIEW FIX: this is the Aurora regression the reviewer identified. If a
    packaging change drops the xccl entry point, the old code returned gloo
    and produced plausible-looking numbers measured over the wrong fabric.
    """

    def test_xpu_without_xccl_raises_instead_of_returning_gloo(self):
        a, b = _with(("gloo", "fake"), "xpu")
        with a, b:
            with self.assertRaises(RuntimeError) as cm:
                tcb.resolve_transport("gpu")
        msg = str(cm.exception)
        self.assertIn("xccl", msg)       # what was expected
        self.assertIn("gloo", msg)       # what was actually found
        self.assertNotEqual(msg.strip(), "")

    def test_cuda_without_any_nccl_raises(self):
        a, b = _with(("gloo", "fake"), "cuda")
        with a, b:
            with self.assertRaises(RuntimeError):
                tcb.resolve_transport("gpu")

    def test_rocm_without_rccl_raises(self):
        a, b = _with(("gloo",), "rocm")
        with a, b:
            with self.assertRaises(RuntimeError):
                tcb.resolve_transport("gpu")

    def test_explicit_gloo_on_accelerator_is_still_allowed(self):
        # Opting in is fine; it is the SILENT downgrade that is not.
        a, b = _with(("gloo", "fake"), "xpu")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu", "gloo"), "gloo")

    def test_cpu_run_on_accelerator_machine_still_gets_gloo(self):
        # vendor is forced to "none" by device_type=cpu, so fail-closed must
        # not fire: a host-memory run over gloo is exactly right.
        a, b = _with(("gloo", "fake"), "xpu")
        with a, b:
            self.assertEqual(tcb.resolve_transport("cpu"), "gloo")


class TestAuroraUnchanged(unittest.TestCase):
    """The fix must not change what Aurora does."""

    def test_xpu_with_xccl_registered_picks_xccl(self):
        a, b = _with(("xccl", "gloo"), "xpu")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "xccl")
            self.assertEqual(tcb.resolve_transport("xpu"), "xccl")

    def test_unenumerable_entry_points_fall_back_to_historical_guess(self):
        # If introspection fails we must not invent a different answer than
        # the code shipped with. status="unavailable" means "unknown".
        a, b = _with((), "xpu", status="unavailable")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "xccl")
            self.assertEqual(tcb.resolve_transport("cpu"), "gloo")

    def test_unenumerable_keeps_explicit_request(self):
        a, b = _with((), "xpu", status="unavailable")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu", "xccl"), "xccl")


class TestRocm(unittest.TestCase):
    def test_rocm_prefers_rccl(self):
        a, b = _with(("rccl", "gloo"), "rocm")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "rccl")


class TestNoAccelerator(unittest.TestCase):
    def test_cpu_only_machine(self):
        a, b = _with(("gloo", "fake"), "none")
        with a, b:
            self.assertEqual(tcb.resolve_transport("gpu"), "gloo")


class TestEnumerationStatus(unittest.TestCase):
    """"could not enumerate" and "enumerated, found nothing" differ.

    REVIEW FIX: both used to collapse to (), so a broken install took the
    Aurora fallback path and failed later inside new_comm with a worse
    message.
    """

    def test_failure_reports_unavailable(self):
        with mock.patch("importlib.metadata.entry_points",
                        side_effect=RuntimeError("boom")):
            status, names = tcb.enumerate_transports()
        self.assertEqual(status, "unavailable")
        self.assertEqual(names, ())

    def test_success_reports_ok(self):
        with mock.patch("importlib.metadata.entry_points",
                        return_value=[_EP("nccl"), _EP("gloo")]):
            status, names = tcb.enumerate_transports()
        self.assertEqual(status, "ok")
        self.assertEqual(names, ("gloo", "nccl"))

    def test_empty_but_successful_enumeration_is_an_error_not_a_guess(self):
        a, b = _with((), "xpu", status="ok")
        with a, b:
            with self.assertRaises(RuntimeError) as cm:
                tcb.resolve_transport("gpu")
        self.assertIn("entry-point", str(cm.exception))


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
        status, reg = tcb.enumerate_transports()
        if status != "ok" or not reg:
            self.skipTest("no enumerable torchcomms backends on this host")
        try:
            got = tcb.resolve_transport("gpu")
        except RuntimeError:
            # Fail-closed refusal is a legitimate outcome on a host whose
            # accelerator has no native transport registered.
            return
        self.assertIn(got, reg,
                      f"resolved {got!r} which is not registered: {reg}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
