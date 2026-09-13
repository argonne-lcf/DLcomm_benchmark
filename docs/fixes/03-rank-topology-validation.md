# Fix 3 — Orphaned ranks silently excluded from all groups

**Severity:** high — a misconfigured launch produced a clean, plausible, meaningless result
**Files:** `dl_comm/config/topology.py` (new), `dl_comm/config/mpi_utils.py`, `dl_comm/config/validation.py`, `dl_comm/dl_comm_main.py`
**Tests:** `tests/test_topology_validation.py`

## The defect

Group construction in `setup_communication_groups()` strides by the ranks the
launcher actually placed per node, but iterates only over the configured
device list:

```python
ranks_per_physical_node = mpi_size // num_compute_nodes
for gpu_idx, gpu_id in enumerate(device_ids_per_node):
    rank = node * ranks_per_physical_node + gpu_idx
```

When `-ppn` exceeds `len(device_ids_per_node)`, the surplus ranks on each node
match no `gpu_idx` and are never placed into any group. They participate in no
collective, contribute no timing, and exit normally.

Concretely, a 2-node × 4-device configuration launched with `-ppn 12`:

```
within group 0: ranks [0, 1, 2, 3]
within group 1: ranks [12, 13, 14, 15]
placed:   8 of 24
orphaned: [4..11, 16..23]
```

16 of 24 ranks did nothing. The job exited 0 and printed a clean timing table.

## Why nothing caught it

Two guards existed and neither was active.

1. The assertion that would have caught the mismatch was commented out at
   `validation.py:350-354`.
2. `validate_mpi_configuration()` computed a `has_errors` flag and then
   discarded it — it was never returned, and the caller never branched on it.

## The fix

`dl_comm/config/topology.py` provides `validate_rank_topology()`, which
reproduces the exact group-construction arithmetic, computes the set of ranks
that will actually be placed, and reports the difference against `mpi_size`.

It rejects:

- more ranks launched than the configuration can place (the orphan case);
- fewer ranks launched than the configuration requires;
- `device_ids_per_node` inconsistent with `num_devices_per_node`;
- `mpi_size` not an exact multiple of `num_compute_nodes`;
- a launcher `PALS_LOCAL_SIZE` / `PMI_LOCAL_SIZE` that disagrees with the
  configured devices per node.

The error names the orphaned ranks rather than reporting a bare count, and the
commented-out assertion in `validation.py` is replaced with a live call.
`validate_mpi_configuration()` now returns its error flag and
`dl_comm_main.py` exits non-zero on it.

`strict=False` downgrades the failure to a warning for the deliberate
oversubscription case, but the default is to refuse to run.

## Verification

```
$ python -m pytest tests/test_topology_validation.py -q
```

`test_the_original_orphaned_rank_scenario_is_rejected` reproduces the 2×4 /
`-ppn 12` launch and asserts it is now rejected.
`test_orphaned_rank_list_is_accurate` pins the placed set to exactly
`{0,1,2,3,12,13,14,15}` and the orphan list to the remaining 16 ranks, so the
helper cannot drift from the group-construction code it mirrors.

That pinning caught a real error during development: the first version of
`covered_ranks()` strided by device count rather than
`mpi_size // num_compute_nodes` and produced the wrong orphan set. The test
failed and the helper was corrected to match the shipped arithmetic.
