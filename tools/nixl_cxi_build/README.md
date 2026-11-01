# Building NIXL with the LIBFABRIC plugin on Cray/Slingshot

The `nixl` pip wheel advertises a `LIBFABRIC` backend through
`get_plugin_list()` but **does not ship it**. `getPluginParams('LIBFABRIC')`
returns `NIXL_ERR_NOT_FOUND`. The wheel's bundled UCX has only `ib`, `ib_mlx5`,
`ib_efa`, `cma` and `cuda` transports — a Cassini/CXI NIC matches none of them,
so UCX silently falls back to **kernel TCP**.

On a 200 Gb/s Slingshot link that measured **0.37 GB/s, 1.5% of the 23.92 GB/s**
the same link does with raw CXI RDMA. Nothing errors. Every correctness check
passes. The only symptom is the bandwidth.

Building NIXL from source with `-Denable_plugins=LIBFABRIC` takes that path to
**23.99 GB/s (100.3% of the raw-CXI reference)** — a 65× improvement.

```
tools/nixl_cxi_build/
├── build_nixl_cxi.sh     configure + build + install, with the three fixes
└── patch_nixl_cray.py    idempotent source patch for blocker A
```

Both scripts carry site-specific paths at the top (`W`, `C13`, `FI`, the venv).
Edit those for your machine; the rest is portable to any Cray libfabric site.

## Verify before you trust any number

**Listed ≠ usable.** Always make this succeed before believing a measurement:

```python
getPluginParams('LIBFABRIC')
# -> mem types ['DRAM_SEG', 'VRAM_SEG']
#    params {'split_batch_size': '1024', 'num_threads': '0'}
```

If it raises `NIXL_ERR_NOT_FOUND`, the plugin was never built and you are
measuring TCP.

## The three blockers, in the order you hit them

### A — `fi_mr_attr.device.rocr` does not exist

```
libfabric_rail.cpp:1748: error: 'union fi_mr_attr::<unnamed>' has no member named 'rocr'
```

NIXL guards that write with a **version** test:

```c
#if FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION) >= FI_VERSION(2, 3)
```

Cray libfabric 2.3.1 reports `FI_MAJOR_VERSION 2` / `FI_MINOR_VERSION 3`, so the
guard passes — but its `fi_mr_attr.device` union is only
`{reserved, cuda, ze, neuron, synapseai}`. `rocr` landed in a later upstream
2.3.x. **A version test is the wrong predicate for a member problem.**

`patch_nixl_cray.py` adds a `cpp.compiles()` probe in
`src/utils/libfabric/meson.build` that defines
`-DNIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR`, and switches the `#if` to an `#ifdef` on
it. This mirrors how the same file already probes
`FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV`. Safe on NVIDIA: `rocr` is the AMD ROCm
path and is dead code there — the CUDA arm (`mr_attr.device.cuda`) is what runs.

### B — `ModuleNotFoundError: No module named 'distutils.core'`

The `nixl-meta` wheel target shells out to `uv build`; on Python 3.12 distutils
is gone. That target is gated on `find_program('uv', required: false)`, so the
cheapest fix is to **drop `uv` from `PATH`** for the build. meson emits
`WARNING: uv not found, skipping meta package build` and carries on. The C++
library, the LIBFABRIC plugin and the pybind11 `_bindings` module are separate
targets and are unaffected.

### C — `ld: cannot find -lfabric`

`-Dlibfabric_path=PREFIX` is a trap. `meson.build` does:

```meson
libfabric_lib_path = libfabric_path + '/lib'
```

Cray ships `libfabric.so` in **`PREFIX/lib64`**, and there is no `lib`. The
option that looks like "point at my libfabric" is exactly what breaks the link.

**Do not pass `-Dlibfabric_path`.** Omitting it takes the
`dependency('libfabric')` pkg-config branch instead, and
`PKG_CONFIG_PATH=<prefix>/lib64/pkgconfig` resolves `-L.../lib64 -lfabric`
correctly.

## Working configure line

```bash
export CC=/opt/cray/pe/gcc-native/14/bin/gcc
export CXX=/opt/cray/pe/gcc-native/14/bin/g++
export PKG_CONFIG_PATH=/opt/cray/libfabric/2.3.1/lib64/pkgconfig:$PKG_CONFIG_PATH
# uv removed from PATH here (blocker B)
meson setup build --prefix=$W/nixl_cxi \
  -Denable_plugins=LIBFABRIC \
  -Dcudapath_inc=$C13/include -Dcudapath_lib=$C13/lib64 \
  -Dbuild_docs=false -Dwith_trace=false \
  -Dbuild_tests=false -Dbuild_examples=false \
  -Dnixl_cuda_arch_list=90
```

`-Denable_plugins=LIBFABRIC` also dodges a separate failure where the GDS plugin
hardcodes `gds_path=/usr/local/cuda/`, which does not exist when CUDA lives
under `/opt/nvidia/hpc_sdk/...`.

## Runtime wiring

The installed plugin has no RPATH to its siblings (`libnixl_common.so => not
found`). Set all three:

```bash
export LD_LIBRARY_PATH=$P/lib64:$C13/lib64:/opt/cray/libfabric/2.3.1/lib64:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=$P/lib64/plugins
export PYTHONPATH=$P/lib/python3.12/site-packages   # installs as nixl_cu13
```

Put `PYTHONPATH` **first**, or you silently re-import the UCX-only pip wheel and
measure the old TCP number again.

Provider selection is automatic: `libfabric_common.cpp` prefers `"cxi"` when it
appears in the provider/device map, so no `FI_PROVIDER` override is needed.

## Harness pitfall

NIXL's model is **asymmetric**. The target must register, publish metadata, and
stay alive *progressing* while the initiator connects. Two ranks that both call
`add_remote_agent()` and then exit produce a bogus
`connect(...) failed: Connection refused` → `NIXL_ERR_BACKEND`. That is a
harness defect, not a fabric fault — whichever rank reaches teardown first kills
its listener mid-dial.

## Versions this was measured against

| Component | Version |
|---|---|
| NIXL | 1.6.0, git `46ac6014e9477889d63711e3e17072b9521b85b3` |
| libfabric | 2.3.1 (`libfabric.so.1.29.1`), Cray, libs in `lib64` |
| libcxi | `libcxi.so.1` |
| CUDA | 13.0 (nvcc V13.0.48) |
| NVIDIA driver | 580.65.06, open kernel module, aarch64 |
| GCC | 14.3.0 (Cray gcc-native/14) |
| Python | 3.12.12 |
| meson / ninja | 1.12.1 / 1.13.2 |
| OS / kernel | SLES 15-SP7, 6.4.0-150700.53.63-64kb, aarch64 (GH200) |
| NIC | 4× Cassini CXI, 200 Gb/s each |
