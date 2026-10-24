"""NIXL transfer measurement for DLcomm.

Purpose
-------
``dl_comm.transfer.h2d`` measures the host-device link through the framework's
own copy path (``torch.Tensor.copy_``). That answers "how fast is this PCIe /
NVLink-C2C link under CUDA", which is the right question for a training job
whose tensors already live in torch.

It is the wrong question for a *disaggregated* job -- KV-cache offload,
prefill/decode split, tiered storage -- where the transfer is issued by an
accelerator-agnostic data-movement library against registered memory, not by
the framework. NIXL (NVIDIA Inference Xfer Library) is that layer: one API over
UCX/GDS/POSIX/object backends, with explicit memory registration and
asynchronous transfer handles.

The cost model differs from a plain ``copy_`` in ways that matter and that the
h2d module cannot surface:

* **Registration is not free.** ``register_memory`` pins and describes the
  buffer to the backend. A benchmark that registers inside the timed loop
  measures registration, not transfer. We register once, outside.
* **Transfers are asynchronous.** ``initialize_xfer`` returns a handle;
  ``transfer`` starts it; the state machine runs until ``DONE``. Timing must
  bracket the *completion*, not the submission.
* **The backend is a variable.** UCX over a local device is a different path
  from POSIX, and reporting a number without naming the backend is not a
  measurement.

Patterns measured
-----------------
``h2d``, ``d2h``, ``d2d`` -- same three directions as the CUDA layer, so the
two can be compared directly on one plot. ``d2d`` here is intra-process device
to device (two distinct device buffers), matching ``h2d.measure``'s definition.

Methodology mirrors ``dl_comm.transfer.h2d`` deliberately -- shuffled iota
fill, best-of-N, optional barrier before each iteration -- so that a NIXL
number and a CUDA number for the same direction differ only by the transfer
mechanism under test.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NixlTransferResult",
    "NixlUnavailable",
    "available",
    "measure_nixl",
    "format_nixl_table",
]

_VALID_DIRECTIONS = ("h2d", "d2h", "d2d")

# NIXL spells memory types with these tags in descriptor tuples.
_DRAM = "DRAM"
_VRAM = "VRAM"


class NixlUnavailable(RuntimeError):
    """Raised when the NIXL runtime or a usable backend is absent.

    Carries the underlying cause rather than a bare "not available", because
    the three failure modes need different fixes: the package is missing, the
    package imports but no plugin is registered, or a plugin exists but cannot
    serve the requested memory type.
    """


def available() -> bool:
    """True when ``nixl`` imports and exposes the agent API.

    Deliberately does not instantiate an agent: construction allocates
    resources and emits log lines, which is too heavy for a capability probe
    used to decide whether to skip a layer.
    """
    try:
        from nixl._api import nixl_agent  # noqa: F401
    except Exception:
        return False
    return True


def plugins(agent_name: str = "dlcomm_probe") -> list[str]:
    """Return the backend plugins this NIXL build registered.

    Needed because "nixl is installed" and "nixl can move bytes on this node"
    are different claims -- a build with only object/file plugins cannot serve
    a VRAM transfer, and the failure would otherwise surface as an opaque
    error deep inside ``initialize_xfer``.
    """
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except Exception as exc:  # pragma: no cover - import guarded by available()
        raise NixlUnavailable(f"nixl import failed: {exc}") from exc
    agent = nixl_agent(agent_name, nixl_agent_config(backends=[]))
    return list(agent.get_plugin_list())


@dataclass
class NixlTransferResult:
    """One measured NIXL transfer pattern.

    Field names match :class:`dl_comm.transfer.h2d.TransferResult` so both can
    feed one table; ``backend`` is the addition, because a NIXL bandwidth
    figure is meaningless without naming the plugin that carried it.
    """

    direction: str
    backend: str
    nbytes: int
    world_size: int
    times_s: list[float] = field(default_factory=list)
    registration_s: float = 0.0

    @property
    def total_bytes(self) -> int:
        """Bytes this rank moved in one iteration.

        Deliberately NOT ``nbytes * world_size``. Each rank registers its own
        buffers and times only its own copy, so scaling by the world size
        makes a single rank's local measurement claim job-wide throughput:
        the printed rate then rises with node count while the measured time
        is unchanged (observed: identical best_s of ~0.137 s reported as
        1.92 GB/s at 4 ranks and 7.87 GB/s at 16). Aggregate throughput is
        the caller's business -- sum ``gbytes_per_s`` over ranks, which is
        only correct because this value is per-rank.
        """
        return self.nbytes

    @property
    def best_s(self) -> float:
        if not self.times_s:
            raise ValueError(
                f"no timings recorded for {self.direction}; "
                "measure_nixl() did not complete an iteration"
            )
        return min(self.times_s)

    @property
    def gbytes_per_s(self) -> float:
        """This rank's GB/s (10^9) using its best iteration."""
        best = self.best_s
        if best <= 0:
            raise ValueError(
                f"non-positive best time {best!r} for {self.direction}; "
                "the timer resolution is too coarse for this buffer size"
            )
        return self.total_bytes / best / 1e9


def _await_done(agent: Any, handle: Any, direction: str, timeout_s: float) -> None:
    """Block until a transfer handle reports DONE.

    A bounded wait, not ``while True``: a transfer that never completes is a
    real failure mode (wrong backend for the memory type, peer never posted
    the matching side) and must surface as a diagnosable error instead of a
    hung job that the batch system eventually kills with no explanation.
    """
    deadline = time.monotonic() + timeout_s
    state = agent.check_xfer_state(handle)
    while state != "DONE":
        if state == "ERR":
            raise RuntimeError(
                f"NIXL reported ERR for a {direction} transfer; "
                "the backend rejected the descriptor list"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"NIXL {direction} transfer did not reach DONE within "
                f"{timeout_s:.1f}s (last state {state!r})"
            )
        state = agent.check_xfer_state(handle)


def resolve_backend(
    requested: str | None = None,
    *,
    preference: tuple[str, ...] = ("UCX", "GDS", "POSIX"),
    agent_name: str = "dlcomm_resolve",
) -> str:
    """Pick a NIXL backend, preferring an explicit request.

    Mirrors ``torchcomms_backend.resolve_transport``: an explicit value is
    validated against what this build actually registered rather than being
    trusted, and the default is resolved from the machine instead of being a
    hardcoded vendor string. Hardcoding ``"UCX"`` would reproduce the exact
    defect that made layer 5 unrunnable on Tara -- a backend name baked in on
    one machine and unregistered on the next.
    """
    registered = plugins(agent_name)
    if requested is not None:
        want = str(requested).upper()
        if want not in registered:
            raise NixlUnavailable(
                f"NIXL backend {requested!r} is not registered in this build; "
                f"available: {registered}"
            )
        return want
    for candidate in preference:
        if candidate in registered:
            return candidate
    raise NixlUnavailable(
        f"none of the preferred NIXL backends {preference} are registered; "
        f"available: {registered}"
    )


def measure_nixl(
    torch_mod: Any,
    device: Any,
    *,
    nbytes: int = 1 << 26,
    iterations: int = 10,
    warmup: int = 3,
    dist: Any = None,
    world_size: int = 1,
    directions: tuple[str, ...] = _VALID_DIRECTIONS,
    backend: str | None = None,
    backend_preference: tuple[str, ...] = ("UCX", "GDS", "POSIX"),
    backend_params: dict[str, Any] | None = None,
    backend_env: dict[str, str] | None = None,
    agent_name: str = "dlcomm",
    timeout_s: float = 60.0,
) -> list[NixlTransferResult]:
    """Measure h2d / d2h / d2d bandwidth through NIXL.

    ``dist`` is optional; when supplied its ``barrier`` runs before each timed
    iteration so ranks start together, matching ``h2d.measure``.

    All backend selection and tuning is optional:

    * ``backend`` -- when ``None``, resolved from what this NIXL build
      registered, in ``backend_preference`` order.
    * ``backend_params`` -- backend-specific keys forwarded to
      ``nixl_agent_config``.
    * ``backend_env`` -- environment variables (``UCX_TLS``,
      ``UCX_IB_GDA_RETAIN_INACTIVE_CTX``, ...) applied **before** the agent is
      constructed and restored afterwards. UCX reads its configuration once at
      agent creation, so setting these after the fact is silently ignored;
      applying them here is the only point at which they take effect.

    Registration happens once per buffer, before timing, and is reported
    separately in ``registration_s`` -- it is a real per-job cost but not part
    of steady-state transfer bandwidth.
    """
    for d in directions:
        if d not in _VALID_DIRECTIONS:
            raise ValueError(
                f"unknown direction {d!r}; expected one of {_VALID_DIRECTIONS}"
            )
    if nbytes <= 0:
        raise ValueError(f"nbytes must be positive, got {nbytes}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except Exception as exc:
        raise NixlUnavailable(
            "transfer layer 'nixl' requires the nixl package; "
            "install it with `pip install nixl` or drop 'nixl' from "
            "transfer_layers"
        ) from exc

    resolved_backend = resolve_backend(backend, preference=backend_preference)

    cfg_kwargs: dict[str, Any] = {"backends": [resolved_backend]}
    if backend_params:
        cfg_kwargs.update(backend_params)

    saved_env: dict[str, str | None] = {}
    if backend_env:
        for k, v in backend_env.items():
            saved_env[k] = os.environ.get(k)
            os.environ[k] = str(v)
    try:
        try:
            agent = nixl_agent(agent_name, nixl_agent_config(**cfg_kwargs))
        except TypeError as exc:
            # An unsupported tuning key must name itself, not surface as a
            # bare TypeError from inside the binding.
            raise NixlUnavailable(
                f"nixl_agent_config rejected backend_params "
                f"{backend_params!r}: {exc}"
            ) from exc
        except Exception as exc:
            # A restrictive UCX_TLS is the common cause: the backend cannot
            # build at all and NIXL reports an opaque NIXL_ERR_BACKEND.
            raise NixlUnavailable(
                f"NIXL could not create backend {resolved_backend!r} "
                f"(backend_env={backend_env!r}): {exc}"
            ) from exc
    finally:
        for k, old in saved_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    backend = resolved_backend

    itemsize = 4
    n = nbytes // itemsize
    if n == 0:
        raise ValueError(f"nbytes={nbytes} is smaller than one int32 element")

    i32 = torch_mod.int32
    results: list[NixlTransferResult] = []

    # Shuffled iota, never a constant: a zero/constant buffer can be served
    # from a zero page or compressed in transit, inflating apparent bandwidth.
    src_host = torch_mod.randperm(n, dtype=torch_mod.int64).to(i32)
    dst_host = torch_mod.empty(n, dtype=i32)
    src_dev = torch_mod.empty(n, dtype=i32, device=device)
    dst_dev = torch_mod.empty(n, dtype=i32, device=device)
    src_dev.copy_(src_host)

    def _reg(tensor: Any) -> tuple[Any, float]:
        t0 = time.monotonic()
        descs = agent.register_memory(tensor)
        if descs is None:
            raise NixlUnavailable(
                "NIXL register_memory returned no descriptors for a "
                f"{'device' if tensor.is_cuda else 'host'} buffer; backend "
                f"{backend!r} cannot serve this memory type"
            )
        return descs, time.monotonic() - t0

    reg_src_host, t_rsh = _reg(src_host)
    reg_dst_host, t_rdh = _reg(dst_host)
    reg_src_dev, t_rsd = _reg(src_dev)
    reg_dst_dev, t_rdd = _reg(dst_dev)

    pairs = {
        "h2d": (reg_src_host, reg_dst_dev, t_rsh + t_rdd),
        "d2h": (reg_src_dev, reg_dst_host, t_rsd + t_rdh),
        "d2d": (reg_src_dev, reg_dst_dev, t_rsd + t_rdd),
    }

    try:
        for direction in directions:
            src_descs, dst_descs, reg_cost = pairs[direction]
            res = NixlTransferResult(
                direction=direction,
                backend=backend,
                nbytes=nbytes,
                world_size=world_size,
                registration_s=reg_cost,
            )

            # Local transfer: both sides belong to this agent, so the
            # descriptor lists are exchanged with ourselves.
            handle = agent.initialize_xfer(
                "WRITE", src_descs.trim(), dst_descs.trim(), agent.name
            )

            for i in range(warmup + iterations):
                if dist is not None:
                    dist.barrier()
                t0 = time.monotonic()
                state = agent.transfer(handle)
                if state == "ERR":
                    raise RuntimeError(
                        f"NIXL refused to start a {direction} transfer"
                    )
                _await_done(agent, handle, direction, timeout_s)
                elapsed = time.monotonic() - t0
                if i >= warmup:
                    res.times_s.append(elapsed)

            agent.release_xfer_handle(handle)
            results.append(res)
    finally:
        for descs in (reg_src_host, reg_dst_host, reg_src_dev, reg_dst_dev):
            try:
                agent.deregister_memory(descs)
            except Exception:
                # Teardown must not mask a real measurement error.
                pass
        # Drop the agent while the CUDA context is still alive. Left to
        # interpreter shutdown, the UCX worker is destroyed after CUDA has
        # already torn down its context and the rank dies with SIGSEGV --
        # *after* printing correct results, so the crash looks unrelated to
        # this function and the job reports exit 139 on a successful run.
        try:
            del agent
        except Exception:
            pass
        gc.collect()

    return results


def format_nixl_table(results: list[NixlTransferResult]) -> str:
    """Render results as fixed-width text, one row per pattern."""
    if not results:
        return "(no NIXL results)"
    head = (
        f"{'direction':<14}{'backend':<10}{'bytes':>14}"
        f"{'best_s':>12}{'GB/s':>12}{'reg_s':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.direction:<14}{r.backend:<10}{r.nbytes:>14}"
            f"{r.best_s:>12.6f}{r.gbytes_per_s:>12.2f}{r.registration_s:>10.4f}"
        )
    return "\n".join(lines)
