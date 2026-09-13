"""torchcomms backend adapter for DLcomm.

``torchcomms`` is Meta's experimental replacement for the ``torch.distributed``
collectives API (open-sourced October 2025, native XCCL/NCCLX/RCCL backends).
It is a *transport*, not a framework: it moves the same ``torch.Tensor``
objects that ``torch.distributed`` does. That is why it is wired in here as a
``ccl_backend`` value rather than as a third ``framework`` alongside pytorch
and jax -- the framework axis costs 56 branch sites, this costs one.

The design constraint is that the 13 registered collectives in
``dl_comm/comm/collectives.py`` already receive their communication module as a
``dist=`` parameter. So rather than branch inside every collective, this module
exposes a ``TorchCommsDist`` object implementing the 17 ``dist.*`` functions
those collectives actually call. The collectives are unmodified.

Two differences from ``torch.distributed`` are handled here:

1. torchcomms is object-oriented -- ``comm.all_reduce(...)`` on a communicator
   object -- where torch.distributed is module-level with a ``group=`` kwarg.
   Subcommunicators come from ``comm.split(ranks, name)``, which this module
   maps onto DLcomm's ``group`` concept.
2. Every torchcomms op takes a mandatory positional ``async_op`` argument and
   returns a ``TorchWork``. Blocking semantics are ``async_op=False``; the
   non-blocking path returns the work handle so ``isend``/``irecv`` can expose
   a ``.wait()`` just as torch.distributed does.

Op coverage is complete for DLcomm's registry, including the vector variants
(``all_to_all_v_single``) and point-to-point (``send``/``recv`` with
``async_op``), which means ``sendrecv_async`` is natively expressible here.
See ``docs/fixes/17-torchcomms-backend.md``.
"""

from __future__ import annotations

from typing import Any, Sequence

# torchcomms is optional: importing this module must not fail on a machine
# without it. Resolution is deferred to is_available() / build().
try:  # pragma: no cover - exercised on Aurora, not in CPU CI
    import torchcomms as _tc
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _tc = None


BACKEND_NAME = "torchcomms"

#: torchcomms backends that can carry DLcomm traffic, by accelerator vendor.
SUPPORTED_TRANSPORTS = ("ncclx", "nccl", "rcclx", "rccl", "xccl", "gloo")


def is_available() -> bool:
    """True when the torchcomms module imported successfully."""
    return _tc is not None


def _require():
    if _tc is None:
        raise RuntimeError(
            "ccl_backend 'torchcomms' was requested but the torchcomms module "
            "is not importable. torchcomms needs PyTorch >= 2.8; on Aurora it "
            "ships with frameworks/2025.3.1. Install it with "
            "`pip install torchcomms`, or choose a different ccl_backend."
        )
    return _tc


def resolve_transport(device_type: str, requested: str | None = None) -> str:
    """Pick the torchcomms transport for a device type.

    DLcomm's config names a *device* (gpu/cpu) and Aurora means Intel XPU, so
    the natural transport there is xccl. An explicit request always wins, but
    is validated rather than trusted.
    """
    if requested:
        t = requested.lower()
        if t not in SUPPORTED_TRANSPORTS:
            raise ValueError(
                f"Unknown torchcomms transport '{requested}'. "
                f"Valid transports: {list(SUPPORTED_TRANSPORTS)}"
            )
        return t
    if device_type and device_type.lower() in ("gpu", "xpu"):
        return "xccl"
    return "gloo"


class _Work:
    """Wraps a ``TorchWork`` so callers can use torch.distributed's ``.wait()``.

    A blocking torchcomms call still returns a work handle; waiting on it again
    is harmless, which keeps one code path for both cases.
    """

    __slots__ = ("_work",)

    def __init__(self, work: Any):
        self._work = work

    def wait(self, timeout=None):  # noqa: ARG002 - signature parity
        if self._work is not None and hasattr(self._work, "wait"):
            return self._work.wait()
        return None

    def is_completed(self) -> bool:
        if self._work is not None and hasattr(self._work, "is_completed"):
            return bool(self._work.is_completed())
        return True


class TorchCommsDist:
    """A ``torch.distributed``-shaped facade over a torchcomms communicator.

    Only the 17 entry points DLcomm's collectives actually call are
    implemented. Anything else raises ``AttributeError`` naturally, which is
    preferable to silently degrading to a different transport.
    """

    def __init__(self, comm: Any, op_map: dict | None = None):
        self._comm = comm
        self._splits: dict[tuple, Any] = {}
        self._op_map = op_map or {}

    # -- communicator plumbing ------------------------------------------
    @property
    def comm(self):
        return self._comm

    def _c(self, group):
        """Resolve DLcomm's ``group`` to a torchcomms communicator."""
        if group is None:
            return self._comm
        if isinstance(group, TorchCommsGroup):
            return group.comm
        return group

    def _op(self, op):
        """Map a DLcomm op (str or torch ReduceOp) to a torchcomms ReduceOp."""
        tc = _require()
        if op is None:
            return tc.ReduceOp.SUM
        if isinstance(op, str):
            name = op.upper()
            name = {"PRODUCT": "PRODUCT", "PROD": "PRODUCT", "MEAN": "AVG"}.get(name, name)
            if not hasattr(tc.ReduceOp, name):
                raise ValueError(
                    f"torchcomms has no reduce op '{op}'. Available: "
                    f"{[o for o in dir(tc.ReduceOp) if o.isupper()]}"
                )
            return getattr(tc.ReduceOp, name)
        # torch.distributed.ReduceOp -> match by name
        text = str(op).rsplit(".", 1)[-1].upper()
        text = {"PROD": "PRODUCT", "MEAN": "AVG"}.get(text, text)
        return getattr(tc.ReduceOp, text, tc.ReduceOp.SUM)

    # -- rank / topology -------------------------------------------------
    def get_rank(self, group=None) -> int:
        return int(self._c(group).get_rank())

    def get_world_size(self, group=None) -> int:
        return int(self._c(group).get_size())

    def get_process_group_ranks(self, group=None):
        if isinstance(group, TorchCommsGroup):
            return list(group.ranks)
        return list(range(self.get_world_size(group)))

    def new_group(self, ranks: Sequence[int], name: str | None = None,
                  backend=None, timeout=None,
                  use_local_synchronization: bool = False,
                  group_desc: str | None = None, pg_options=None):
        """Create a subcommunicator. Mirrors ``dist.new_group``.

        The torch.distributed-only keywords (``backend``, ``timeout``,
        ``use_local_synchronization``, ``group_desc``, ``pg_options``) are
        accepted and ignored: torchcomms' ``split`` has no equivalent knobs,
        and callers such as ``comm_setup.setup_communication_groups`` pass
        ``use_local_synchronization=True`` unconditionally. Rejecting them
        crashed job 8824643 on all 24 ranks at group-creation time.
        """
        key = tuple(sorted(int(r) for r in ranks))
        if key not in self._splits:
            label = name or group_desc or (
                "dlcomm_" + "_".join(str(r) for r in key))
            self._splits[key] = TorchCommsGroup(
                self._comm.split(list(key), label), list(key)
            )
        return self._splits[key]

    # -- capability probes --------------------------------------------------
    def is_mpi_available(self) -> bool:
        """torchcomms has no MPI transport; report that honestly."""
        return False

    def is_nccl_available(self) -> bool:
        tc = _tc
        if tc is None:
            return False
        # torchcomms exposes nccl/ncclx as transports rather than as a
        # torch.distributed backend flag.
        return "nccl" in SUPPORTED_TRANSPORTS or "ncclx" in SUPPORTED_TRANSPORTS

    # -- dense collectives ------------------------------------------------
    def all_reduce(self, tensor, op=None, group=None, async_op=False):
        w = self._c(group).all_reduce(tensor, self._op(op), async_op)
        return _Work(w) if async_op else None

    def reduce(self, tensor, dst=0, op=None, group=None, async_op=False):
        w = self._c(group).reduce(tensor, int(dst), self._op(op), async_op)
        return _Work(w) if async_op else None

    def broadcast(self, tensor, src=0, group=None, async_op=False):
        w = self._c(group).broadcast(tensor, int(src), async_op)
        return _Work(w) if async_op else None

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        w = self._c(group).all_gather(tensor_list, tensor, async_op)
        return _Work(w) if async_op else None

    def gather(self, tensor, gather_list=None, dst=0, group=None, async_op=False):
        # torch.distributed passes gather_list=None on non-root ranks;
        # torchcomms wants a sequence, so hand it an empty one.
        w = self._c(group).gather(gather_list or [], tensor, int(dst), async_op)
        return _Work(w) if async_op else None

    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False):
        w = self._c(group).scatter(tensor, scatter_list or [], int(src), async_op)
        return _Work(w) if async_op else None

    def reduce_scatter(self, output, input_list, op=None, group=None, async_op=False):
        w = self._c(group).reduce_scatter(output, input_list, self._op(op), async_op)
        return _Work(w) if async_op else None

    def all_to_all(self, output_tensor_list, input_tensor_list, group=None, async_op=False):
        w = self._c(group).all_to_all(output_tensor_list, input_tensor_list, async_op)
        return _Work(w) if async_op else None

    def all_to_all_single(self, output, input, output_split_sizes=None,
                          input_split_sizes=None, group=None, async_op=False):
        """Even split uses all_to_all_single; uneven uses the _v variant.

        DLcomm's ``alltoallv`` relies on the split-size path, so routing here
        keeps one call site for both collectives.
        """
        c = self._c(group)
        if output_split_sizes is None and input_split_sizes is None:
            w = c.all_to_all_single(output, input, async_op)
        else:
            n = c.get_size()
            osz = list(output_split_sizes) if output_split_sizes is not None else [output.numel() // n] * n
            isz = list(input_split_sizes) if input_split_sizes is not None else [input.numel() // n] * n
            w = c.all_to_all_v_single(output, input, osz, isz, async_op)
        return _Work(w) if async_op else None

    def barrier(self, group=None, async_op=False, device_ids=None):  # noqa: ARG002
        """Barrier. ``device_ids`` is torch.distributed-only and ignored:
        the torchcomms communicator is already bound to its device."""
        w = self._c(group).barrier(async_op)
        return _Work(w) if async_op else None

    # -- point-to-point ----------------------------------------------------
    def send(self, tensor, dst, group=None, tag=0):  # noqa: ARG002 - no tags in torchcomms
        self._c(group).send(tensor, int(dst), False)

    def recv(self, tensor, src=None, group=None, tag=0):  # noqa: ARG002
        if src is None:
            raise ValueError(
                "torchcomms requires an explicit src for recv; "
                "wildcard receives are not supported by this backend."
            )
        self._c(group).recv(tensor, int(src), False)

    def isend(self, tensor, dst, group=None, tag=0):  # noqa: ARG002
        return _Work(self._c(group).send(tensor, int(dst), True))

    def irecv(self, tensor, src=None, group=None, tag=0):  # noqa: ARG002
        if src is None:
            raise ValueError(
                "torchcomms requires an explicit src for irecv; "
                "wildcard receives are not supported by this backend."
            )
        return _Work(self._c(group).recv(tensor, int(src), True))

    # -- lifecycle ---------------------------------------------------------
    def init_process_group(self, *args, **kwargs):  # noqa: ARG002
        """No-op: the communicator is built in ``build()``.

        Present so that any caller following the torch.distributed lifecycle
        does not crash on a missing attribute.
        """
        return None

    def destroy_process_group(self, group=None):
        """Tear down subcommunicators and then the world communicator.

        torchcomms calls this ``finalize()``. Split communicators are finalized
        first: releasing the parent while a child is live is undefined.
        """
        if group is not None:
            c = self._c(group)
            if hasattr(c, "finalize"):
                c.finalize()
            return
        for g in self._splits.values():
            try:
                if hasattr(g.comm, "finalize"):
                    g.comm.finalize()
            except Exception:  # noqa: BLE001 - shutdown must not mask results
                pass
        self._splits.clear()
        if hasattr(self._comm, "finalize"):
            self._comm.finalize()


class TorchCommsGroup:
    """A torchcomms subcommunicator plus the global ranks it covers."""

    __slots__ = ("comm", "ranks")

    def __init__(self, comm: Any, ranks: Sequence[int]):
        self.comm = comm
        self.ranks = list(ranks)

    def __repr__(self) -> str:
        return f"TorchCommsGroup(ranks={self.ranks})"


def build(device, device_type: str = "gpu", transport: str | None = None,
          name: str = "dlcomm", timeout=None) -> TorchCommsDist:
    """Create the world communicator and wrap it in the dist facade.

    torchcomms derives rank/world size from ``RANK``/``WORLD_SIZE`` and
    bootstraps via ``MASTER_ADDR``/``MASTER_PORT``, exactly as torchrun sets
    them. DLcomm launches under mpiexec, so the caller is responsible for
    exporting those from the MPI rank -- see ``dl_comm_main``.
    """
    tc = _require()
    kwargs = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    comm = tc.new_comm(
        resolve_transport(device_type, transport),
        device,
        name,
        **kwargs,
    )
    return TorchCommsDist(comm)
