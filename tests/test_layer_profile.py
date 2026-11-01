#!/usr/bin/env python3
"""Offline tests for layer_profile.py.

The phase timing needs real GPUs, but the parsing and transport-detection
logic does not -- and that logic is exactly where a silent wrong answer
would be most damaging, because a mis-parsed NCCL transport line would let
me attribute the 2-node cliff to the wrong cause.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "21_cuda_layer_sweeps"))

import layer_profile as lp  # noqa: E402


class TestSizeParsing(unittest.TestCase):
    def test_units(self):
        self.assertEqual(lp.parse_size("4096"), 4096)
        self.assertEqual(lp.parse_size("1KiB"), 1024)
        self.assertEqual(lp.parse_size("16MiB"), 16 * 2**20)
        self.assertEqual(lp.parse_size("1GiB"), 2**30)

    def test_bad(self):
        for bad in ("", "12MB", "abc", "-4KiB"):
            with self.assertRaises(ValueError):
                lp.parse_size(bad)

    def test_human_roundtrip(self):
        for s in ("1KiB", "16MiB", "1GiB"):
            self.assertEqual(lp.human(lp.parse_size(s)), s)


class TestNcclTransportParsing(unittest.TestCase):
    """The single most important parser in the profiler."""

    def test_detects_socket_fallback(self):
        # TCP fallback: this is the outcome that would explain a 51x cliff.
        text = (
            "node0:1:1 [0] NCCL INFO Channel 00 : 0[0] -> 1[0] "
            "[send] via NET/Socket/0\n"
        )
        r = lp.parse_nccl_transport(text)
        self.assertEqual(r["net"], "Socket")

    def test_detects_ofi(self):
        text = ("node0:1:1 [0] NCCL INFO NET/OFI Using provider cxi\n"
                "node0:1:1 [0] NCCL INFO Channel 01 via NET/OFI/0\n")
        r = lp.parse_nccl_transport(text)
        self.assertEqual(r["net"], "OFI")
        self.assertEqual(r["provider"], "cxi")

    def test_detects_ib(self):
        text = "n:1:1 [0] NCCL INFO Channel 00 : 0[0] -> 1[0] via NET/IB/0\n"
        self.assertEqual(lp.parse_nccl_transport(text)["net"], "IB")

    def test_no_evidence_is_none_not_guess(self):
        """If NCCL printed nothing, we must NOT invent a transport."""
        r = lp.parse_nccl_transport("some unrelated output\n")
        self.assertIsNone(r["net"])
        self.assertIsNone(r["provider"])
        self.assertEqual(r["evidence"], [])

    def test_keeps_evidence_lines(self):
        text = "x NCCL INFO Channel 00 via NET/Socket/0\n"
        r = lp.parse_nccl_transport(text)
        self.assertEqual(len(r["evidence"]), 1)
        self.assertIn("NET/Socket", r["evidence"][0])


class TestProvenance(unittest.TestCase):
    def test_reports_cxi_absence_truthfully(self):
        p = lp.capture_provenance()
        self.assertIn("cxi_devices", p)
        self.assertIsInstance(p["cxi_devices"], list)
        self.assertIn("host", p)
        self.assertIn("nccl_net_plugins", p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
