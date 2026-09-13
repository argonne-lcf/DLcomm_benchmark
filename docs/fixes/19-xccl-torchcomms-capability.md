# Finding 19 — torchcomms/XCCL capability on Aurora

**Artefacts:** `probe_xccl_support.py`, `run_probe.sh`, job `8824688`
(`/lus/flare/projects/datascience/kaushik/DLcomm/validation/probe_20260913_220852`)

## Result

The torchcomms XCCL backend on Aurora implements **one** of the fourteen
operations DLcomm requires.

```
XCCL SUPPORT MATRIX (torch 2.10.0a0+git449b176, world=24)
  [yes] all_reduce
  [NO ] barrier                  [NO ] broadcast
  [NO ] reduce                   [NO ] all_gather
  [NO ] all_gather_single        [NO ] reduce_scatter_single
  [NO ] all_to_all_single        [NO ] all_to_all_v_single
  [NO ] scatter                  [NO ] gather
  [NO ] send/recv                [NO ] send/recv async
  [NO ] split
```

Every failure carries the same upstream message:

```
RuntimeError: XCCL <op> is not supported now and will be added later
```

The binding exists as a dispatch skeleton with a single real implementation
behind it.

## Consequence

`ccl_backend: torchcomms` is **not usable on Aurora** at
`frameworks/2025.3.1`. This is an upstream gap, not a DLcomm or adapter
defect: the adapter completed environment bootstrap, created both 12-rank
within-node subcommunicators, and entered the timing loop before reaching an
unimplemented op.

The adapter (docs/fixes/17) is retained and tested so the backend becomes
usable as Intel fills in the XCCL operations. Re-run `run_probe.sh` after any
`frameworks` module update to re-measure.

## Method: probe by calling, never by introspection

`torchcomms present: True` and `xccl available: True` are both accurate and
both misleading. All twenty methods exist and are introspectable; `hasattr`
returns `True` for every one of the thirteen that throw at call time.

`probe_xccl_support.py` therefore judges capability by **invoking** each
operation on real device tensors and recording the outcome:

```python
def check(name, fn):
    try:
        fn(); torch.xpu.synchronize(); results[name] = "OK"
    except Exception as e:
        results[name] = f"FAIL: {str(e).splitlines()[0][:90]}"
```

The same trap applies to gloo, whose `all_to_all` is present as a symbol and
refuses at call time.

## Cost of the alternative

Three PBS jobs were spent discovering three unimplemented ops one at a time
(`8824643` signature bug, `8824653` `split`, `8824676` `barrier`), at roughly
twenty minutes of queue wait each. A single job exercising all fourteen
operations produced the complete boundary. When a dependency's capability is
unknown, enumerate it in one job rather than discovering it incrementally.

## Reuse

```bash
qsub run_probe.sh        # 2 nodes, 24 ranks, ~7 s of compute
```

The matrix is printed by rank 0 and the unsupported list is emitted as a
single line for scripted comparison across module versions.
