"""Cover the pci_fixed_cuda size/overflow contract without a GPU.

The checks here are on the C++ source itself, compiled and run on the host
with -DNO_MPI and a stub CUDA runtime. That keeps them runnable on any
machine -- including CI hosts with no nvcc and no GPU -- while still
exercising the real parse_sizes()/report() code paths rather than a
re-implementation of them.

REVIEW FIX (20261024T071020Z-dlcomm3c). The independent review found that
DLCOMM_TRANSFER_BYTES accepted any value >= sizeof(int), while the element
counts derived from it round DOWN:

    max_elems = max_bytes / sizeof(int)     # 5 / 4 == 1
    std::vector<int> ref(max_elems);        # 4 bytes allocated
    std::memcpy(ref.data(), a_cpu, N_byte); # 5 bytes written

Confirmed as a heap-buffer-overflow under AddressSanitizer. The fix rejects
any size that is not a positive multiple of sizeof(int); these tests pin that
and the surrounding input validation.

Run: python3 tests/test_pci_fixed_cuda_sizes.py
(plain unittest; the Tara dlcomm-env has no pytest)
"""

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "dl_comm", "transfer", "pci_fixed_cuda.cu")

# Minimal cuda_runtime.h so the translation unit compiles with a host C++
# compiler. Every entry point the benchmark touches is present; the ones the
# size tests reach (parse_sizes, report) never call into it, and main() exits
# at cudaGetDeviceCount before the rest matters.
CUDA_STUB = r"""
#pragma once
#include <cstring>
#include <cstdlib>
typedef enum { cudaSuccess = 0, cudaErrorUnknown = 1 } cudaError_t;
typedef void *cudaStream_t;
typedef enum { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
               cudaMemcpyDefault } cudaMemcpyKind;
typedef enum { cudaHostAllocDefault = 0 } cudaHostAllocFlags;
struct cudaDeviceProp { char name[256]; };
static inline const char *cudaGetErrorName(cudaError_t) { return "STUB"; }
static inline cudaError_t cudaGetDeviceCount(int *n) { *n = 1; return cudaSuccess; }
static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
static inline cudaError_t cudaStreamCreate(cudaStream_t *s) { *s = nullptr; return cudaSuccess; }
static inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaMalloc(int **p, size_t n) { *p = (int *)std::malloc(n); return *p ? cudaSuccess : cudaErrorUnknown; }
static inline cudaError_t cudaFree(void *p) { std::free(p); return cudaSuccess; }
static inline cudaError_t cudaHostAlloc(int **p, size_t n, int) { *p = (int *)std::malloc(n); return *p ? cudaSuccess : cudaErrorUnknown; }
static inline cudaError_t cudaFreeHost(void *p) { std::free(p); return cudaSuccess; }
static inline cudaError_t cudaMemset(void *p, int v, size_t n) { std::memset(p, v, n); return cudaSuccess; }
static inline cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
static inline cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, cudaMemcpyKind, cudaStream_t) { std::memcpy(d, s, n); return cudaSuccess; }
static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int) { std::strcpy(p->name, "stub"); return cudaSuccess; }
"""


def _compiler():
    for cc in ("g++", "c++", "clang++"):
        if shutil.which(cc):
            return cc
    return None


class PciFixedSizeContract(unittest.TestCase):
    tmpdir = None
    binary = None
    build_error = None

    @classmethod
    def setUpClass(cls):
        cc = _compiler()
        if cc is None:
            raise unittest.SkipTest("no host C++ compiler available")
        if not os.path.exists(SRC):
            raise unittest.SkipTest(f"source not found: {SRC}")
        cls.tmpdir = tempfile.mkdtemp(prefix="pcifixed_")
        with open(os.path.join(cls.tmpdir, "cuda_runtime.h"), "w") as fh:
            fh.write(CUDA_STUB)
        cls.binary = os.path.join(cls.tmpdir, "pci_fixed_host")
        proc = subprocess.run(
            [cc, "-std=c++17", "-O0", "-DNO_MPI", "-x", "c++",
             os.path.abspath(SRC), "-I", cls.tmpdir, "-o", cls.binary],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            cls.build_error = proc.stderr[-2000:]

    @classmethod
    def tearDownClass(cls):
        if cls.tmpdir:
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        if self.build_error:
            self.fail(f"host build of pci_fixed_cuda.cu failed:\n{self.build_error}")

    def _run(self, bytes_env):
        env = dict(os.environ)
        env["DLCOMM_TRANSFER_BYTES"] = bytes_env
        env["DLCOMM_TRANSFER_ITERS"] = "1"
        p = subprocess.run([self.binary], capture_output=True, text=True,
                           env=env, timeout=120)
        return p.returncode, p.stdout + p.stderr

    # ---- the overflow the reviewer found -------------------------------
    def test_non_multiple_of_int_is_rejected(self):
        rc, out = self._run("5")
        self.assertNotEqual(rc, 0, "5 bytes must be rejected, not accepted")
        self.assertIn("bad_transfer_bytes_not_multiple_of", out)

    def test_several_non_multiples_all_rejected(self):
        for v in ("1", "2", "3", "7", "4095", "1048575"):
            rc, out = self._run(v)
            self.assertNotEqual(rc, 0, f"{v} must be rejected")
            self.assertIn("bad_transfer_bytes", out, f"value {v}")

    def test_zero_is_rejected(self):
        rc, out = self._run("0")
        self.assertNotEqual(rc, 0)
        self.assertIn("bad_transfer_bytes_zero", out)

    def test_garbage_is_rejected_not_an_uncaught_exception(self):
        # std::stoull used to throw std::invalid_argument straight out of
        # main(), which aborts with no diagnostic the user can act on.
        rc, out = self._run("abc")
        self.assertNotEqual(rc, 0)
        self.assertIn("bad_transfer_bytes_not_a_number", out)
        self.assertNotIn("terminate called", out)

    def test_trailing_garbage_is_rejected(self):
        rc, out = self._run("4096xyz")
        self.assertNotEqual(rc, 0)
        self.assertIn("bad_transfer_bytes", out)
        self.assertNotIn("terminate called", out)

    def test_valid_multiple_is_accepted(self):
        # A valid size runs to completion against the stub runtime, which
        # proves parse_sizes accepted it.
        rc, out = self._run("4096")
        self.assertEqual(rc, 0, out[-2000:])
        self.assertNotIn("bad_transfer_bytes", out)
        self.assertIn("CORRECTNESS", out)
        self.assertIn("failures=0", out)

    def test_valid_list_is_accepted(self):
        rc, out = self._run("4096, 8192 ,16384")
        self.assertEqual(rc, 0, out[-2000:])
        self.assertNotIn("bad_transfer_bytes", out)
        self.assertIn("failures=0", out)

    def test_payload_verification_actually_runs(self):
        # FIX 3 / RF4: every pattern must contribute checks, including the
        # two bidirectional destinations that were previously unverified.
        rc, out = self._run("4096")
        self.assertEqual(rc, 0, out[-2000:])
        checks = [ln for ln in out.splitlines() if "CORRECTNESS" in ln]
        self.assertTrue(checks, out[-2000:])
        n = int(checks[0].split("checks=")[1].split()[0])
        # 2 host-memory arms x 5 verified destinations x 3 spot checks.
        self.assertEqual(n, 30, checks[0])

    def test_timing_fields_are_present(self):
        rc, out = self._run("4096")
        self.assertEqual(rc, 0, out[-2000:])
        self.assertIn("TIME_NS=", out)
        self.assertIn("PATTERN=bidirectional", out)
        self.assertIn("HOSTMEM=pinned", out)
        self.assertIn("HOSTMEM=pageable", out)

    def test_one_bad_entry_in_a_list_rejects_the_run(self):
        rc, out = self._run("4096,5,8192")
        self.assertNotEqual(rc, 0)
        self.assertIn("bad_transfer_bytes_not_multiple_of", out)


class PciFixedSourceContract(unittest.TestCase):
    """Source-level invariants the review required.

    These are cheap greps rather than behavioural tests, but each pins a
    specific regression that behaviour alone would not catch on a host with
    no GPU.

    Comments are stripped before matching. The first version of these tests
    searched the raw file, so `test_timing_uses_explicit_nanosecond_cast`
    failed against the very comment that documents the fix -- a test that
    reads prose rather than code.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(SRC):
            raise unittest.SkipTest(f"source not found: {SRC}")
        with open(SRC) as fh:
            raw = fh.read()
        cls.text = raw
        # Strip // line comments and /* */ blocks so assertions see code only.
        no_block = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
        cls.code = re.sub(r"//[^\n]*", "", no_block)

    def test_warmup_actually_exists(self):
        # The header claimed a warmup fix that was never implemented.
        self.assertIn("DLCOMM_TRANSFER_WARMUP", self.code)
        self.assertIn("for (int w = 0; w < warmup; ++w)", self.code)

    def test_bidirectional_uses_more_than_one_stream(self):
        # Two copies on one stream serialize, so the duplex number was false.
        self.assertIn("std::vector<cudaStream_t> streams", self.code)
        self.assertIn("streams[k]", self.code)

    def test_bidirectional_destinations_are_verified(self):
        self.assertIn('verify("bidirectional_d2h"', self.code)
        self.assertIn('verify("bidirectional_h2d"', self.code)

    def test_copy_errors_checked_before_reporting(self):
        self.assertIn("abort_if_copy_failed", self.code)
        bidir = self.code.index('report("bidirectional"')
        guard = self.code.index('abort_if_copy_failed(N_byte, "bidirectional")')
        self.assertLess(guard, bidir,
                        "error check must precede the bandwidth report")

    def test_timing_uses_explicit_nanosecond_cast(self):
        # TIME_NS must actually be nanoseconds.
        self.assertIn("duration_cast<std::chrono::nanoseconds>", self.code)
        self.assertNotIn("high_resolution_clock", self.code)

    def test_error_paths_abort_rather_than_returning(self):
        self.assertIn("fail_fast", self.code)
        self.assertIn("MPI_Abort", self.code)

    def test_byte_product_overflow_is_checked(self):
        self.assertIn("byte_product_overflow", self.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
