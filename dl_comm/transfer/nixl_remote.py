"""Cross-node NIXL RDMA transfer measurement for DLcomm.

``nixl_transfer.measure_nixl`` moves bytes between buffers that both belong to
the local agent -- h2d/d2h/d2d on one node. That is a real path, but it is not
the one disaggregated serving depends on: a prefill node writing KV cache into
a decode node's VRAM crosses the fabric, and the fabric is what the benchmark
exists to characterize. This module measures that path.

Why a separate module rather than a flag on ``measure_nixl``:

* The local path needs no peer discovery. This one requires an out-of-band
  exchange of agent metadata and descriptor lists before a transfer can be
  initialized at all, so it needs a communicator and a pairing policy.
* Completion means something different. Locally, ``DONE`` on the initiating
  handle means the bytes landed. Remotely, ``DONE`` means *my* write was
  delivered to the peer -- it says nothing about whether the peer's write to
  *me* has arrived. Reading my own destination buffer on local ``DONE`` is a
  race, and it is the kind that passes on a quiet machine. Inbound completion
  is established with the peer's notification, which is why every transfer
  here carries one.
* Timing has to exclude that notification wait. A notification round-trip is
  latency, not bandwidth; folding it into the timed region produces a number
  that falls with message size and looks like a slow fabric. The timed region
  below is the local WRITE only.

Correctness is not optional here. A WRITE to a wrong-but-valid remote address
reports ``DONE`` and moves bytes nowhere useful, so every measurement is
followed by a byte-level check against a rank-identifiable payload: element
*i* of rank *r*'s buffer is ``r * _TAG_STRIDE + i``. A buffer that arrives
from the wrong peer, or is left untouched, cannot satisfy that.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# Large enough that no two ranks' payloads overlap within any buffer size this
# benchmark uses, so a received buffer identifies its sender unambiguously.
_TAG_STRIDE = 1_000_003

_VALID_PAIRINGS = ("split", "ring")


class NixlRemoteUnavailable(RuntimeError):
    """Raised when a cross-node NIXL transfer cannot be set up.

    Distinct from ``nixl_transfer.NixlUnavailable`` because the remedies
    differ: that one means NIXL cannot move bytes at all, this one means the
    local path works but peer setup failed (metadata exchange, connection,
    or descriptor deserialization).
    """


class NixlVerificationError(RuntimeError):
    """Raised when a transfer completed but delivered the wrong bytes.

    A hard error rather than a logged warning. A cross-node bandwidth figure
    computed over a transfer that did not deliver the sender's data is not a
    slow result, it is a meaningless one, and must not reach a results table.
    """


@dataclass
class RemoteTransferResult:
    """One measured cross-node NIXL pattern.

    ``verified`` is part of the record, not a side channel, so a table cannot
    show a bandwidth number without showing whether the bytes were checked.
    """

    direction: str
    backend: str
    nbytes: int
    world_size: int
    local_rank_id: int
    peer_rank_id: int
    cross_node: bool
    times_s: list[float] = field(default_factory=list)
    registration_s: float = 0.0
    verified: bool = False
    mismatched_elements: int = 0

    @property
    def total_bytes(self) -> int:
        """Bytes this rank moved in one iteration.

        Per-rank, never scaled by world size -- see the identical note in
        ``nixl_transfer.NixlTransferResult.total_bytes``, where multiplying by
        ``world_size`` made a single rank's local copy report job-wide
        throughput and inflated the 16-rank figure fourfold.
        """
        return self.nbytes

    @property
    def best_s(self) -> float:
        if not self.times_s:
            raise ValueError(
                f"no timings recorded for {self.direction}; the measurement "
                "loop did not complete an iteration"
            )
        return min(self.times_s)

    @property
    def median_s(self) -> float:
        """Median iteration time.

        Reported alongside the best because a fabric shared with other jobs
        has a tail: a best-of-N can be a quiet-moment outlier, and the gap
        between the two is the honest indicator of how contended the run was.
        """
        if not self.times_s:
            raise ValueError(f"no timings recorded for {self.direction}")
        ordered = sorted(self.times_s)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @property
    def gbytes_per_s(self) -> float:
        best = self.best_s
        if best <= 0:
            raise ValueError(
                f"non-positive best time {best!r} for {self.direction}; the "
                "timer resolution is too coarse for this buffer size"
            )
        return self.total_bytes / best / 1e9

    @property
    def median_gbytes_per_s(self) -> float:
        med = self.median_s
        if med <= 0:
            raise ValueError(f"non-positive median time {med!r}")
        return self.total_bytes / med / 1e9


def pair_ranks(
    rank: int,
    world_size: int,
    *,
    pairing: str = "split",
    ranks_per_node: int | None = None,
) -> int:
    """Return the rank this rank exchanges with.

    ``split`` pairs rank *r* with ``r + world/2``. With a block rank
    distribution (``mpiexec -ppn``) that guarantees both halves sit on
    different nodes, which is the point -- it is the only pairing here whose
    cross-node property can be reasoned about without knowing the launcher's
    placement.

    ``ring`` pairs *r* with ``r + ranks_per_node`` (mod world), giving every
    rank a partner one node away. Useful at larger node counts where ``split``
    would send traffic between the two most distant halves only.

    Neither pairing is *assumed* to be cross-node by the measurement code --
    ``measure_remote_nixl`` compares the actual hostnames and records the
    answer, because a pairing that silently degenerates to same-node traffic
    would otherwise be reported as a fabric measurement.
    """
    if world_size < 2:
        raise ValueError(
            f"cross-node transfer needs at least 2 ranks, got {world_size}"
        )
    if world_size % 2 != 0:
        raise ValueError(
            f"world_size must be even to pair every rank, got {world_size}; "
            "launch an even number of ranks or drop one from the measurement"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    if pairing == "split":
        return (rank + world_size // 2) % world_size
    if pairing == "ring":
        if not ranks_per_node or ranks_per_node < 1:
            raise ValueError(
                "pairing='ring' requires ranks_per_node >= 1 to know how far "
                "to step; pass it explicitly rather than guessing"
            )
        if world_size % ranks_per_node != 0:
            raise ValueError(
                f"world_size {world_size} is not a multiple of ranks_per_node "
                f"{ranks_per_node}; ring pairing would be asymmetric"
            )
        if world_size == ranks_per_node:
            raise ValueError(
                "ring pairing on a single node has no cross-node partner; "
                "run on 2+ nodes or use pairing='split'"
            )
        return (rank + ranks_per_node) % world_size
    raise ValueError(f"unknown pairing {pairing!r}; expected one of {_VALID_PAIRINGS}")


def expected_payload(torch_mod: Any, n_elements: int, sender_rank: int) -> Any:
    """The int32 payload rank ``sender_rank`` sends.

    Separate from the measurement so a test can assert the contract without a
    GPU, a fabric, or NIXL installed.
    """
    if n_elements <= 0:
        raise ValueError(f"n_elements must be positive, got {n_elements}")
    base = torch_mod.arange(n_elements, dtype=torch_mod.int64)
    return (base + sender_rank * _TAG_STRIDE).to(torch_mod.int32)


def _await_state(
    agent: Any,
    handle: Any,
    label: str,
    timeout_s: float,
    poll_sleep_s: float = 0.0,
) -> None:
    """Block until the handle reports DONE, bounded.

    Bounded rather than ``while True`` because the common cross-node failure
    -- a peer that never posted its side, or a descriptor list pointing at
    memory the remote agent never registered -- otherwise hangs until the
    batch system kills the job with no diagnosis in the log.
    """
    deadline = time.monotonic() + timeout_s
    state = agent.check_xfer_state(handle)
    while state != "DONE":
        if state == "ERR":
            raise RuntimeError(
                f"NIXL reported ERR during {label}; the backend rejected the "
                "descriptor list or the remote agent went away"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"NIXL {label} did not reach DONE within {timeout_s:.1f}s "
                f"(last state {state!r})"
            )
        if poll_sleep_s:
            time.sleep(poll_sleep_s)
        state = agent.check_xfer_state(handle)


def _await_notif(
    agent: Any,
    prefix: bytes,
    timeout_s: float,
    label: str,
    poll_sleep_s: float = 0.0,
) -> bytes:
    """Wait for an inbound notification whose message starts with ``prefix``.

    This is what establishes that the *peer's* write into this rank's buffer
    has landed. The local handle reaching DONE only proves the outbound
    direction; reading the destination buffer on that signal alone is a race
    that passes whenever the peer happens to be slightly ahead.

    Notifications are drained into a local list because ``get_new_notifs``
    returns everything that has arrived since the last call, including
    messages for an iteration this rank has not reached yet.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        batch = agent.get_new_notifs() or {}
        for _sender, messages in batch.items():
            for msg in messages:
                if msg.startswith(prefix):
                    return msg
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no inbound NIXL notification matching {prefix!r} within "
                f"{timeout_s:.1f}s during {label}; the peer's write to this "
                "rank never completed"
            )
        if poll_sleep_s:
            time.sleep(poll_sleep_s)


def measure_remote_nixl(
    torch_mod: Any,
    device: Any,
    *,
    rank: int,
    world_size: int,
    allgather: Callable[[Any], Sequence[Any]],
    barrier: Callable[[], None],
    hostname: str,
    nbytes: int = 1 << 26,
    iterations: int = 10,
    warmup: int = 3,
    pairing: str = "split",
    ranks_per_node: int | None = None,
    backend: str | None = None,
    backend_preference: tuple[str, ...] = ("UCX", "GDS", "POSIX"),
    backend_params: dict[str, Any] | None = None,
    backend_env: dict[str, str] | None = None,
    agent_name: str | None = None,
    timeout_s: float = 120.0,
    verify: bool = True,
) -> RemoteTransferResult:
    """Measure one rank's cross-node device-to-device RDMA write through NIXL.

    ``allgather`` and ``barrier`` are injected rather than imported from
    mpi4py so the caller owns the communicator and so the whole function is
    testable with plain Python stand-ins -- the peer-exchange logic is where
    the bugs live, and it should not need a cluster to exercise.

    The timed region is the local WRITE to DONE and nothing else. Barrier,
    notification wait, and verification all sit outside it; including them
    reports latency as bandwidth, which at 1 MiB understates the fabric by
    more than an order of magnitude.
    """
    if nbytes <= 0:
        raise ValueError(f"nbytes must be positive, got {nbytes}")
    if nbytes % 4 != 0:
        raise ValueError(
            f"nbytes must be a multiple of 4 (int32 payload), got {nbytes}"
        )
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except Exception as exc:
        raise NixlRemoteUnavailable(
            "cross-node NIXL transfer requires the nixl package; install it "
            "with `pip install nixl` or drop 'nixl_remote' from the layer list"
        ) from exc

    from dl_comm.transfer.nixl_transfer import NixlUnavailable, resolve_backend

    try:
        resolved = resolve_backend(backend, preference=backend_preference)
    except NixlUnavailable as exc:
        raise NixlRemoteUnavailable(str(exc)) from exc

    peer_rank = pair_ranks(
        rank, world_size, pairing=pairing, ranks_per_node=ranks_per_node
    )

    # Hostnames are compared, not assumed. A pairing that lands both ranks on
    # one node still produces a plausible-looking bandwidth number; recording
    # cross_node=False is what stops it being read as a fabric measurement.
    hosts = list(allgather(hostname))
    if len(hosts) != world_size:
        raise NixlRemoteUnavailable(
            f"allgather returned {len(hosts)} hostnames for world_size "
            f"{world_size}; the communicator does not match the declared world"
        )
    cross_node = hosts[rank] != hosts[peer_rank]

    name = agent_name or f"dlcomm_remote_r{rank}"
    cfg_kwargs: dict[str, Any] = {"backends": [resolved]}
    if backend_params:
        cfg_kwargs.update(backend_params)

    saved_env: dict[str, str | None] = {}
    if backend_env:
        # UCX reads its configuration once, at agent construction. Exporting
        # UCX_* after this point is silently ignored -- a fact that cost
        # several confusing runs during bring-up of the local path.
        for key, value in backend_env.items():
            saved_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
    try:
        try:
            agent = nixl_agent(name, nixl_agent_config(**cfg_kwargs))
        except Exception as exc:
            raise NixlRemoteUnavailable(
                f"NIXL could not create backend {resolved!r} "
                f"(backend_env={backend_env!r}): {exc}"
            ) from exc
    finally:
        for key, old in saved_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    n = nbytes // 4
    result = RemoteTransferResult(
        direction="remote_d2d",
        backend=resolved,
        nbytes=nbytes,
        world_size=world_size,
        local_rank_id=rank,
        peer_rank_id=peer_rank,
        cross_node=cross_node,
    )

    src = expected_payload(torch_mod, n, rank).to(device)
    # Sentinel fill, not zeros: a destination that is never written must fail
    # verification, and zero is a value a real payload could legitimately hold.
    dst = torch_mod.full((n,), -1, dtype=torch_mod.int32, device=device)
    expect = expected_payload(torch_mod, n, peer_rank).to(device)
    if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()

    reg_src = reg_dst = None
    handle = None
    try:
        t_reg = time.monotonic()
        reg_src = agent.register_memory(src)
        reg_dst = agent.register_memory(dst)
        result.registration_s = time.monotonic() - t_reg
        if reg_src is None or reg_dst is None:
            raise NixlRemoteUnavailable(
                f"NIXL register_memory returned no descriptors for VRAM; "
                f"backend {resolved!r} cannot serve device memory on this node"
            )

        # Peer discovery. Both exchanges are collective, so every rank must
        # reach them -- which is why a rank that fails earlier raises rather
        # than returning a partial result and deadlocking its partner here.
        metadata = list(allgather(agent.get_agent_metadata()))
        peer_name = agent.add_remote_agent(metadata[peer_rank])
        if isinstance(peer_name, (bytes, bytearray)):
            peer_name = peer_name.decode()

        # Each rank needs its PARTNER's destination descriptors: a WRITE
        # targets remote memory, so the local side of initialize_xfer is my
        # source and the remote side is their destination.
        serialized = list(allgather(agent.get_serialized_descs(reg_dst.trim())))
        remote_dst = agent.deserialize_descs(serialized[peer_rank])
        if remote_dst is None:
            raise NixlRemoteUnavailable(
                f"could not deserialize destination descriptors from rank "
                f"{peer_rank}; the peer registered memory the local backend "
                "cannot address"
            )

        agent.make_connection(peer_name)
        barrier()

        notif_prefix = b"dlcomm_xfer_"
        handle = agent.initialize_xfer(
            "WRITE",
            reg_src.trim(),
            remote_dst,
            peer_name,
            notif_prefix + str(rank).encode(),
        )
        if handle is None:
            raise NixlRemoteUnavailable(
                f"initialize_xfer returned no handle for peer {peer_name!r}; "
                "the backend refused the remote descriptor list"
            )

        for i in range(warmup + iterations):
            barrier()
            start = time.monotonic()
            state = agent.transfer(handle)
            if state == "ERR":
                raise RuntimeError(
                    f"NIXL refused to start the remote write to rank {peer_rank}"
                )
            _await_state(agent, handle, f"remote write to rank {peer_rank}", timeout_s)
            elapsed = time.monotonic() - start
            if i >= warmup:
                result.times_s.append(elapsed)
            # Outside the timed region: this is the inbound direction's
            # completion, and it is a round-trip latency, not bandwidth.
            _await_notif(
                agent,
                notif_prefix,
                timeout_s,
                f"inbound write from rank {peer_rank}",
            )

        barrier()
        if verify:
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                torch_mod.cuda.synchronize()
            mismatched = int((dst != expect).sum().item())
            result.mismatched_elements = mismatched
            result.verified = mismatched == 0
            if mismatched:
                bad = (dst != expect).nonzero()
                first = int(bad[0].item()) if bad.numel() else -1
                raise NixlVerificationError(
                    f"rank {rank} received {mismatched} of {n} elements that "
                    f"do not match rank {peer_rank}'s payload (first bad index "
                    f"{first}); the transfer completed but delivered the wrong "
                    "bytes, so the bandwidth figure is meaningless"
                )
    finally:
        if handle is not None:
            try:
                agent.release_xfer_handle(handle)
            except Exception:
                pass
        for descs in (reg_src, reg_dst):
            if descs is not None:
                try:
                    agent.deregister_memory(descs)
                except Exception:
                    # Teardown must never mask a real measurement error.
                    pass
        # Drop the agent while the CUDA context is still alive. Left to
        # interpreter shutdown, UCX tears down after CUDA and the rank dies
        # with SIGSEGV *after* printing correct results -- see the exit-139
        # note in nixl_transfer and run_nixl_transfer.
        try:
            del agent
        except Exception:
            pass
        gc.collect()

    return result


def format_remote_table(results: list[RemoteTransferResult]) -> str:
    """Render cross-node results as fixed-width text.

    ``verified`` and ``cross`` are columns rather than footnotes: an unchecked
    or same-node number must not be readable as a verified fabric measurement
    at a glance.
    """
    if not results:
        return "(no cross-node NIXL results)"
    head = (
        f"{'pair':<12}{'backend':<9}{'bytes':>12}{'best_s':>11}"
        f"{'GB/s':>10}{'medGB/s':>10}{'cross':>7}{'verified':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        pair = f"r{r.local_rank_id}->r{r.peer_rank_id}"
        lines.append(
            f"{pair:<12}{r.backend:<9}{r.nbytes:>12}{r.best_s:>11.6f}"
            f"{r.gbytes_per_s:>10.2f}{r.median_gbytes_per_s:>10.2f}"
            f"{str(r.cross_node):>7}{str(r.verified):>10}"
        )
    return "\n".join(lines)
