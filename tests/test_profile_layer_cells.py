#!/usr/bin/env python3
"""Regression tests for the profile layer's CSV rows.

Both bugs these cover were real and silent:

  * `grep -c ... || echo 0` emitted "0\\n0", which printf turned into an
    EXTRA broken CSV row with layer='0'. The sweep still exited 0.
  * DLCOMM_PROF_SIZES contains commas, so embedding it raw shifted every
    later field and the status column ended up holding "0".

Neither changed an exit code, so only a structural assertion catches them.
"""
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sweep_layers import SweepHarness  # noqa: E402

HEADER = ("layer,cell,axis,value,collective,dtype,extra,"
          "nodes,ranks,exit,rows,status,detail").split(",")
VALID_STATUS = {"ok", "failed", "unsupported", "unavailable"}


class TestProfileLayerRows(SweepHarness):
    def runTest(self):
        pass

    def _rows(self, nodes):
        shutil.rmtree(self.run, ignore_errors=True)
        _, rows = self.sweep("profile", nodes=nodes)
        return rows

    def test_emits_four_cells_each_scale(self):
        for nodes in ("1", "2"):
            rows = self._rows(nodes)
            self.assertEqual(len(rows), 4, f"{nodes}-node rows: {rows}")

    def test_every_row_is_the_profile_layer(self):
        """Guards the 'layer=0' corruption from the doubled grep count."""
        for nodes in ("1", "2"):
            for r in self._rows(nodes):
                self.assertEqual(r["layer"], "profile", f"stray row: {r}")

    def test_status_column_is_a_real_status(self):
        """Guards field-shift from commas in the value column."""
        for nodes in ("1", "2"):
            for r in self._rows(nodes):
                self.assertIn(r["status"], VALID_STATUS,
                              f"status={r['status']!r} in {r}")

    def test_no_value_contains_a_comma(self):
        for nodes in ("1", "2"):
            for r in self._rows(nodes):
                for k, v in r.items():
                    if v is not None:
                        self.assertNotIn(",", str(v),
                                         f"comma in {k}={v!r} would shift CSV")

    def test_expected_cell_names(self):
        want = {"provenance", "nccl_transport", "phases", "isolation_control"}
        for nodes in ("1", "2"):
            got = {r["cell"] for r in self._rows(nodes)}
            self.assertEqual(got, want)

    def test_transport_cell_never_invents_a_value(self):
        """With no NCCL output the cell must say so, not name a transport."""
        for nodes in ("1", "2"):
            for r in self._rows(nodes):
                if r["cell"] == "nccl_transport":
                    self.assertIn("no_NET_line", str(r["value"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
