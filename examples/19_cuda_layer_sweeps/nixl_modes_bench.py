#!/usr/bin/env python3
"""NIXL API-coverage benchmark: the paths nixl_putget_bench.py does not touch.

nixl_putget_bench.py walks the basic lifecycle (register -> describe ->
exchange -> initialize_xfer -> transfer -> check_xfer_state -> release) and
exercises 11 of the 37 nixl_agent methods. That proves the layer works; it
does not characterise how Dynamo actually drives it.

This tool adds the missing paths, one per --mode, so a slow or broken mode
names itself instead of hiding inside an aggregate:

  prepped     prep_xfer_dlist + make_prepped_xfer + release_dlist_handle.
              THE path vLLM/Dynamo uses for KV-cache movement: descriptor
              lists are prepared once and reused per transfer, so the
              per-transfer setup cost is paid at startup instead of on every
              request. Compared against the initialize_xfer path at the same
              size, this is the number that says whether the ~20 ms fixed
              cost seen in the basic bench is inherent or amortisable.

  batch       many descriptors in ONE transfer (scatter-gather). A KV cache
              is a set of per-layer blocks, not one contiguous slab, so
              single-descriptor numbers overstate what Dynamo will see.
              Reports per-descriptor cost so batching's benefit is explicit.

  notif       completion by notification (send_notif / get_new_notifs /
              update_notifs / check_remote_xfer_done) instead of polling
              check_xfer_state. Polling burns a core and cannot tell a
              receiver that its buffer is ready; notifications are how a real
              consumer learns a transfer landed.

  introspect  query_memory, estimate_xfer_cost, get_xfer_telemetry,
              query_xfer_backend, plugin/backend params. No fabric traffic --
              this asks NIXL what it thinks it is doing, and cross-checks
              NIXL's own telemetry against our wall-clock timing. If the two
              disagree, one of them is wrong and the report should say so.

Every mode verifies bytes actually moved. A transfer that reports DONE while
moving nothing is a failure, not a fast result.
"""
import argparse
import json
import os
import socket
import sys
import time

# Barrier/rendezvous timeout. Overridable because the default proved too
# tight: MODE_batch at 16 descriptors exceeded 300 s and died at the
# bxfer barrier while the transfers themselves were healthy (n=1 and n=4
# both byte-exact). A harness timeout must not be reported as a fabric
# failure, so make the limit explicit rather than baked in.
SYNC_TIMEOUT = float(os.environ.get("NIXL_SYNC_TIMEOUT", "300"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def size_label(n):
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            v = n / div
            return f"{v:g}{unit}"
    return f"{n}B"


def publish(d, name, payload):
    tmp = os.path.join(d, f".{name}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload if isinstance(payload, bytes) else str(payload).encode())
    os.replace(tmp, os.path.join(d, name))


def collect(d, name, timeout=SYNC_TIMEOUT):
    p = os.path.join(d, name)
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                return fh.read()
        time.sleep(0.05)
    raise TimeoutError(f"waiting for {name} timed out after {timeout}s")


def barrier(d, rank, tag, nranks=2, timeout=SYNC_TIMEOUT):
    publish(d, f"bar.{tag}.{rank}", b"1")
    end = time.time() + timeout
    while time.time() < end:
        if all(os.path.exists(os.path.join(d, f"bar.{tag}.{r}"))
               for r in range(nranks)):
            return
        time.sleep(0.05)
    raise TimeoutError(f"barrier {tag} timed out")


def make_buf(torch, nbytes, mem, gpu, fill=None):
    if mem == "VRAM":
        b = torch.zeros(nbytes, dtype=torch.uint8, device=f"cuda:{gpu}")
    else:
        b = torch.zeros(nbytes, dtype=torch.uint8).pin_memory()
    if fill is not None:
        b.fill_(fill)
    if mem == "VRAM":
        torch.cuda.synchronize()
    return b


def mem_type_for(agent, mem):
    """Ask the agent which spelling it accepts rather than guessing."""
    accepted = getattr(agent, "nixl_mems", None) or {}
    wanted = ("VRAM", "cuda") if mem == "VRAM" else ("DRAM", "cpu")
    return next((m for m in wanted if m in accepted), wanted[0])


def dev_for(mem, gpu):
    return gpu if mem == "VRAM" else 0


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def mode_prepped(A):
    """prep_xfer_dlist + make_prepped_xfer, vs initialize_xfer at same size.

    The comparison is the point: preparing descriptor lists once and reusing
    them is what Dynamo does, and the delta against the naive path is the
    cost Dynamo avoids.
    """
    agent, torch, args = A.agent, A.torch, A.args
    rows = []
    for nbytes in A.sizes:
        tag = size_label(nbytes)
        log(f"---- prepped {tag} ----")
        buf = make_buf(torch, nbytes, args.mem, A.gpu,
                       fill=0xAB if not A.is_init else None)
        reg = agent.register_memory(buf)
        if not reg:
            sys.exit("ERROR: register_memory failed")
        mt = mem_type_for(agent, args.mem)
        tup = [(buf.data_ptr(), nbytes, dev_for(args.mem, A.gpu))]
        local_descs = agent.get_xfer_descs(tup, mem_type=mt)

        publish(A.sync, f"pm.{tag}.{A.rank}", agent.get_agent_metadata())
        publish(A.sync, f"pd.{tag}.{A.rank}", agent.get_serialized_descs(local_descs))
        barrier(A.sync, A.rank, f"pmeta.{tag}")
        peer = agent.add_remote_agent(collect(A.sync, f"pm.{tag}.{A.peer}"))
        remote_descs = agent.deserialize_descs(collect(A.sync, f"pd.{tag}.{A.peer}"))

        row = dict(bytes=nbytes, label=tag, mode="prepped")

        if A.is_init:
            # --- prepare once -------------------------------------------
            t0 = time.perf_counter()
            l_side = agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs, mem_type=mt)
            r_side = agent.prep_xfer_dlist(peer, remote_descs, mem_type=mt)
            t_prep = time.perf_counter() - t0
            if not l_side or not r_side:
                sys.exit("ERROR: prep_xfer_dlist returned no handle")

            # --- per-transfer cost with lists already prepared ----------
            def prepped_once():
                h = agent.make_prepped_xfer(args.op, l_side, [0],
                                            r_side, [0], b"p")
                if not h:
                    sys.exit("ERROR: make_prepped_xfer returned no handle")
                if agent.transfer(h) == "ERR":
                    sys.exit("ERROR: prepped transfer() returned ERR")
                while True:
                    st = agent.check_xfer_state(h)
                    if st == "DONE":
                        break
                    if st == "ERR":
                        sys.exit("ERROR: prepped transfer entered ERR")
                agent.release_xfer_handle(h)

            for _ in range(args.warmup):
                prepped_once()
            s = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                prepped_once()
                s.append(time.perf_counter() - t0)
            p_best, p_mean = min(s), sum(s) / len(s)

            # --- same work through the naive path, for contrast ---------
            def naive_once():
                h = agent.initialize_xfer(args.op, local_descs, remote_descs,
                                          peer, b"n")
                if agent.transfer(h) == "ERR":
                    sys.exit("ERROR: naive transfer() returned ERR")
                while True:
                    st = agent.check_xfer_state(h)
                    if st == "DONE":
                        break
                    if st == "ERR":
                        sys.exit("ERROR: naive transfer entered ERR")
                agent.release_xfer_handle(h)

            for _ in range(args.warmup):
                naive_once()
            s = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                naive_once()
                s.append(time.perf_counter() - t0)
            n_best, n_mean = min(s), sum(s) / len(s)

            row.update(prep_s=t_prep,
                       prepped_best_us=p_best * 1e6, prepped_mean_us=p_mean * 1e6,
                       prepped_best_gbps=nbytes / p_best / 1e9,
                       naive_best_us=n_best * 1e6, naive_mean_us=n_mean * 1e6,
                       naive_best_gbps=nbytes / n_best / 1e9,
                       speedup=n_best / p_best if p_best else None)
            log(f"prep once {t_prep*1e3:8.3f} ms")
            log(f"prepped best {p_best*1e6:10.1f} us -> {nbytes/p_best/1e9:7.3f} GB/s")
            log(f"naive   best {n_best*1e6:10.1f} us -> {nbytes/n_best/1e9:7.3f} GB/s")
            log(f"speedup {n_best/p_best:6.2f}x" if p_best else "speedup n/a")

            agent.release_dlist_handle(l_side)
            agent.release_dlist_handle(r_side)

        barrier(A.sync, A.rank, f"pxfer.{tag}")
        check_bytes(A, buf, nbytes, row)
        agent.deregister_memory(reg)
        rows.append(row)
        del buf
        torch.cuda.empty_cache()
        barrier(A.sync, A.rank, f"pdone.{tag}")
    return rows


def mode_batch(A):
    """N descriptors in one transfer: the scatter-gather shape of a KV cache."""
    agent, torch, args = A.agent, A.torch, A.args
    rows = []
    for ndesc in A.batch_counts:
        per = max(A.total_bytes // ndesc, 4096)
        tag = f"n{ndesc}"
        log(f"---- batch {ndesc} descs x {size_label(per)} ----")
        bufs = [make_buf(torch, per, args.mem, A.gpu,
                         fill=0xAB if not A.is_init else None)
                for _ in range(ndesc)]
        regs = [agent.register_memory(b) for b in bufs]
        if not all(regs):
            sys.exit("ERROR: register_memory failed in batch mode")
        mt = mem_type_for(agent, args.mem)
        tups = [(b.data_ptr(), per, dev_for(args.mem, A.gpu)) for b in bufs]
        local_descs = agent.get_xfer_descs(tups, mem_type=mt)

        publish(A.sync, f"bm.{tag}.{A.rank}", agent.get_agent_metadata())
        publish(A.sync, f"bd.{tag}.{A.rank}", agent.get_serialized_descs(local_descs))
        barrier(A.sync, A.rank, f"bmeta.{tag}")
        peer = agent.add_remote_agent(collect(A.sync, f"bm.{tag}.{A.peer}"))
        remote_descs = agent.deserialize_descs(collect(A.sync, f"bd.{tag}.{A.peer}"))

        moved = per * ndesc
        row = dict(mode="batch", ndesc=ndesc, per_desc_bytes=per, bytes=moved,
                   label=tag)

        if A.is_init:
            h = agent.initialize_xfer(args.op, local_descs, remote_descs,
                                      peer, b"b")
            if not h:
                sys.exit("ERROR: initialize_xfer returned no handle (batch)")

            def once():
                if agent.transfer(h) == "ERR":
                    sys.exit("ERROR: batch transfer() returned ERR")
                while True:
                    st = agent.check_xfer_state(h)
                    if st == "DONE":
                        return
                    if st == "ERR":
                        sys.exit("ERROR: batch transfer entered ERR")

            for _ in range(args.warmup):
                once()
            s = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                once()
                s.append(time.perf_counter() - t0)
            best, mean = min(s), sum(s) / len(s)
            row.update(best_us=best * 1e6, mean_us=mean * 1e6,
                       best_gbps=moved / best / 1e9,
                       per_desc_us=best * 1e6 / ndesc)
            log(f"{ndesc:4d} descs | total {size_label(moved)} | "
                f"best {best*1e6:9.1f} us -> {moved/best/1e9:7.3f} GB/s | "
                f"{best*1e6/ndesc:8.2f} us/desc")
            agent.release_xfer_handle(h)

        barrier(A.sync, A.rank, f"bxfer.{tag}")
        # verify every buffer, not just the first: a partial scatter that
        # fills one descriptor and drops the rest would otherwise pass.
        if A.checker:
            if args.mem == "VRAM":
                torch.cuda.synchronize()
            bad = 0
            for b in bufs:
                span = min(per, 1 << 16)
                exp = torch.full((span,), 0xAB, dtype=torch.uint8, device=b.device)
                if not torch.equal(b[:span], exp):
                    bad += 1
            row["byte_exact"] = (bad == 0)
            row["bad_descs"] = bad
            log(f"byte-exact check: {'PASS' if bad == 0 else f'FAIL ({bad}/{ndesc} descs)'}")

        for r in regs:
            agent.deregister_memory(r)
        rows.append(row)
        del bufs
        torch.cuda.empty_cache()
        barrier(A.sync, A.rank, f"bdone.{tag}")
    return rows


def mode_notif(A):
    """Completion via notification rather than polling check_xfer_state."""
    agent, torch, args = A.agent, A.torch, A.args
    rows = []
    for nbytes in A.sizes:
        tag = size_label(nbytes)
        log(f"---- notif {tag} ----")
        buf = make_buf(torch, nbytes, args.mem, A.gpu,
                       fill=0xAB if not A.is_init else None)
        reg = agent.register_memory(buf)
        mt = mem_type_for(agent, args.mem)
        tup = [(buf.data_ptr(), nbytes, dev_for(args.mem, A.gpu))]
        local_descs = agent.get_xfer_descs(tup, mem_type=mt)

        publish(A.sync, f"nm.{tag}.{A.rank}", agent.get_agent_metadata())
        publish(A.sync, f"nd.{tag}.{A.rank}", agent.get_serialized_descs(local_descs))
        barrier(A.sync, A.rank, f"nmeta.{tag}")
        peer = agent.add_remote_agent(collect(A.sync, f"nm.{tag}.{A.peer}"))
        remote_descs = agent.deserialize_descs(collect(A.sync, f"nd.{tag}.{A.peer}"))

        row = dict(mode="notif", bytes=nbytes, label=tag)
        msg = f"xfer-{tag}".encode()

        if A.is_init:
            # notif_msg rides with the transfer: the target is told the
            # buffer is ready without either side polling transfer state.
            h = agent.initialize_xfer(args.op, local_descs, remote_descs,
                                      peer, msg)
            t0 = time.perf_counter()
            if agent.transfer(h) == "ERR":
                sys.exit("ERROR: notif transfer() returned ERR")
            while True:
                st = agent.check_xfer_state(h)
                if st == "DONE":
                    break
                if st == "ERR":
                    sys.exit("ERROR: notif transfer entered ERR")
            t_xfer = time.perf_counter() - t0

            # does the initiator observe the remote acknowledging it?
            t0 = time.perf_counter()
            seen = False
            end = time.time() + 30
            while time.time() < end:
                if agent.check_remote_xfer_done(peer, msg):
                    seen = True
                    break
                time.sleep(0.001)
            row.update(xfer_us=t_xfer * 1e6,
                       remote_done_us=(time.perf_counter() - t0) * 1e6,
                       remote_done_seen=seen)
            log(f"transfer {t_xfer*1e6:9.1f} us | "
                f"check_remote_xfer_done: {'seen' if seen else 'NOT SEEN'}")
            agent.release_xfer_handle(h)

            # explicit out-of-band notification, separate from the xfer one
            t0 = time.perf_counter()
            agent.send_notif(peer, b"ping")
            row["send_notif_us"] = (time.perf_counter() - t0) * 1e6
        else:
            # target side: wait for the notification to arrive
            t0 = time.perf_counter()
            got, pings = None, 0
            end = time.time() + 60
            while time.time() < end:
                notifs = agent.get_new_notifs() or {}
                for _a, msgs in notifs.items():
                    for m in msgs:
                        if m == msg:
                            got = time.perf_counter() - t0
                        if m == b"ping":
                            pings += 1
                if got is not None and pings:
                    break
                agent.update_notifs()
                time.sleep(0.001)
            row.update(notif_wait_us=(got * 1e6) if got is not None else None,
                       xfer_notif_seen=got is not None,
                       ping_notifs=pings)
            log(f"xfer notif: {'seen' if got is not None else 'NOT SEEN'} | "
                f"ping notifs: {pings}")

        barrier(A.sync, A.rank, f"nxfer.{tag}")
        check_bytes(A, buf, nbytes, row)
        agent.deregister_memory(reg)
        rows.append(row)
        del buf
        torch.cuda.empty_cache()
        barrier(A.sync, A.rank, f"ndone.{tag}")
    return rows


def mode_introspect(A):
    """Ask NIXL what it thinks it is doing; cross-check its own telemetry."""
    agent, torch, args = A.agent, A.torch, A.args
    rows = []
    out = dict(mode="introspect")

    out["plugins"] = list(agent.get_plugin_list() or [])
    log(f"plugins: {out['plugins']}")
    for b in (args.backend,):
        try:
            out[f"plugin_params.{b}"] = dict(agent.get_plugin_params(b) or {})
        except Exception as e:
            out[f"plugin_params.{b}"] = f"ERROR: {e}"
        try:
            out[f"plugin_mem_types.{b}"] = list(agent.get_plugin_mem_types(b) or [])
        except Exception as e:
            out[f"plugin_mem_types.{b}"] = f"ERROR: {e}"
        try:
            out[f"backend_params.{b}"] = dict(agent.get_backend_params(b) or {})
        except Exception as e:
            out[f"backend_params.{b}"] = f"ERROR: {e}"
        try:
            out[f"backend_mem_types.{b}"] = list(agent.get_backend_mem_types(b) or [])
        except Exception as e:
            out[f"backend_mem_types.{b}"] = f"ERROR: {e}"
    log(f"backend mem types: {out.get('backend_mem_types.' + args.backend)}")

    nbytes = A.sizes[-1]
    tag = size_label(nbytes)
    buf = make_buf(torch, nbytes, args.mem, A.gpu,
                   fill=0xAB if not A.is_init else None)
    reg = agent.register_memory(buf)
    mt = mem_type_for(agent, args.mem)
    tup = [(buf.data_ptr(), nbytes, dev_for(args.mem, A.gpu))]
    local_descs = agent.get_xfer_descs(tup, mem_type=mt)

    try:
        qm = agent.query_memory(reg, args.backend, mem_type=mt)
        out["query_memory"] = [dict(x) if x else None for x in (qm or [])]
        log(f"query_memory: {out['query_memory']}")
    except Exception as e:
        out["query_memory"] = f"ERROR: {e}"
        log(f"query_memory ERROR: {e}")

    publish(A.sync, f"im.{tag}.{A.rank}", agent.get_agent_metadata())
    publish(A.sync, f"id.{tag}.{A.rank}", agent.get_serialized_descs(local_descs))
    barrier(A.sync, A.rank, f"imeta.{tag}")
    peer = agent.add_remote_agent(collect(A.sync, f"im.{tag}.{A.peer}"))
    remote_descs = agent.deserialize_descs(collect(A.sync, f"id.{tag}.{A.peer}"))

    if A.is_init:
        h = agent.initialize_xfer(args.op, local_descs, remote_descs, peer, b"i")
        try:
            out["query_xfer_backend"] = agent.query_xfer_backend(h)
            log(f"query_xfer_backend: {out['query_xfer_backend']}")
        except Exception as e:
            out["query_xfer_backend"] = f"ERROR: {e}"
        try:
            est = agent.estimate_xfer_cost(h)
            out["estimate_xfer_cost"] = list(est) if est else None
            log(f"estimate_xfer_cost: {out['estimate_xfer_cost']}")
        except Exception as e:
            out["estimate_xfer_cost"] = f"ERROR: {e}"
            log(f"estimate_xfer_cost ERROR: {e}")

        t0 = time.perf_counter()
        if agent.transfer(h) == "ERR":
            sys.exit("ERROR: introspect transfer() returned ERR")
        while True:
            st = agent.check_xfer_state(h)
            if st == "DONE":
                break
            if st == "ERR":
                sys.exit("ERROR: introspect transfer entered ERR")
        wall_us = (time.perf_counter() - t0) * 1e6
        out["wall_us"] = wall_us

        try:
            tel = agent.get_xfer_telemetry(h)
            fields = {k: getattr(tel, k) for k in dir(tel)
                      if not k.startswith("_") and
                      isinstance(getattr(tel, k), (int, float, str))}
            out["telemetry"] = fields
            log(f"telemetry: {fields}")
            # cross-check: if NIXL reports its own duration, compare it to
            # our wall clock. Disagreement means one of the two is wrong.
            for k, v in fields.items():
                if "us" in k.lower() and isinstance(v, (int, float)) and v > 0:
                    out["telemetry_vs_wall_ratio"] = wall_us / v
                    log(f"wall {wall_us:.1f} us vs telemetry {k}={v} "
                        f"(ratio {wall_us/v:.2f})")
                    break
        except Exception as e:
            out["telemetry"] = f"ERROR: {e}"
            log(f"get_xfer_telemetry ERROR: {e}")
        agent.release_xfer_handle(h)

    barrier(A.sync, A.rank, f"ixfer.{tag}")
    check_bytes(A, buf, nbytes, out)
    agent.deregister_memory(reg)

    # teardown paths the basic bench never calls
    try:
        agent.remove_remote_agent(peer)
        out["remove_remote_agent"] = "ok"
        log("remove_remote_agent ok")
    except Exception as e:
        out["remove_remote_agent"] = f"ERROR: {e}"
        log(f"remove_remote_agent ERROR: {e}")

    rows.append(out)
    del buf
    torch.cuda.empty_cache()
    barrier(A.sync, A.rank, f"idone.{tag}")
    return rows


def check_bytes(A, buf, nbytes, row):
    """The receiving side proves bytes actually moved."""
    if not A.checker:
        return
    torch = A.torch
    if A.args.mem == "VRAM":
        torch.cuda.synchronize()
    span = min(nbytes, 1 << 20)
    exp = torch.full((span,), 0xAB, dtype=torch.uint8, device=buf.device)
    ok = bool(torch.equal(buf[:span], exp))
    row["byte_exact"] = ok
    log(f"byte-exact check: {'PASS' if ok else 'FAIL'}")


class Ctx:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("prepped", "batch", "notif", "introspect"))
    ap.add_argument("--backend", default="LIBFABRIC")
    ap.add_argument("--mem", choices=("VRAM", "DRAM"), default="VRAM")
    ap.add_argument("--op", choices=("READ", "WRITE"), default="READ",
                    help="WRITE is unsupported on CXI; READ is the real path")
    ap.add_argument("--sizes", default="65536,16777216,1073741824")
    ap.add_argument("--batch-counts", default="1,4,16,64,256")
    ap.add_argument("--batch-total", type=int, default=1 << 26,
                    help="total bytes spread across the descriptors")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--sync-dir", required=True)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--expect", choices=("same-node", "cross-node", "any"),
                    default="any")
    args = ap.parse_args()

    rank = int(os.environ.get("PALS_RANKID", os.environ.get("RANK", "0")))
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID",
                                    os.environ.get("LOCAL_RANK", "0")))
    host = socket.gethostname().split(".")[0]
    is_init = (rank == 0)
    role = "initiator" if is_init else "target"
    os.makedirs(args.sync_dir, exist_ok=True)

    publish(args.sync_dir, f"host.{rank}", host)
    barrier(args.sync_dir, rank, "hosts")
    peer_rank = 1 - rank
    peer_host = collect(args.sync_dir, f"host.{peer_rank}").decode().strip()
    placement = "same-node" if peer_host == host else "cross-node"

    import torch
    gpu = local_rank % max(torch.cuda.device_count(), 1)
    torch.cuda.set_device(gpu)
    log(f"{role} rank={rank} host={host} gpu={gpu} peer={peer_host} "
        f"placement={placement} mode={args.mode}")

    if args.expect != "any" and args.expect != placement:
        sys.exit(f"ERROR: expected {args.expect} placement but got {placement}. "
                 f"Refusing to report a {placement} result as {args.expect}.")

    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except Exception as exc:
        sys.exit(f"ERROR: cannot import nixl ({exc})")

    agent = nixl_agent(f"{role}-r{rank}",
                       nixl_agent_config(backends=[args.backend]))
    log(f"agent up, backend={args.backend} mem={args.mem} op={args.op}")

    A = Ctx()
    A.agent, A.torch, A.args = agent, torch, args
    A.rank, A.peer, A.gpu = rank, peer_rank, gpu
    A.is_init, A.sync = is_init, args.sync_dir
    A.sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    A.batch_counts = [int(s) for s in args.batch_counts.split(",") if s.strip()]
    A.total_bytes = args.batch_total
    # whoever RECEIVES the bytes validates them
    A.checker = is_init if args.op == "READ" else (not is_init)

    rows = dict(prepped=mode_prepped, batch=mode_batch,
                notif=mode_notif, introspect=mode_introspect)[args.mode](A)

    if is_init and args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(dict(layer="nixl_modes", mode=args.mode, op=args.op,
                           mem=args.mem, placement=placement, host=host,
                           peer_host=peer_host, rows=rows), fh, indent=1)

    log(f"{role} complete (mode={args.mode}, {placement})")


if __name__ == "__main__":
    sys.exit(main())
