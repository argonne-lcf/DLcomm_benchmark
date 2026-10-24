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
        self.assertIn('verify_now("bidirectional_d2h"', self.code)
        self.assertIn('verify_now("bidirectional_h2d"', self.code)

    def test_copy_errors_checked_before_reporting(self):
        self.assertIn("abort_if_copy_failed", self.code)
        # bidirectional reports under a computed pattern name
        bidir = self.code.index('report(bidir_pattern(pinned)')
        guard = self.code.index(
            'abort_if_copy_failed(N_byte, bidir_pattern(pinned))')
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

    # -- round-2 review blockers ------------------------------------------
    def test_verification_precedes_report_for_every_pattern(self):
        """R2-1: a corrupt copy must not emit a bandwidth line.

        Round 2 found report() running before verify() for all four
        patterns, so a transfer whose CUDA calls succeeded but whose bytes
        arrived wrong still printed a plausible GBPS record. Exiting
        non-zero later does not retract a line already scraped from the log.
        """
        for pattern, verify_tag in (
            ('report("h2d"', '"h2d"'),
            ('report("d2h"', '"d2h"'),
            ('report("d2d"', '"d2d"'),
        ):
            with self.subTest(pattern=pattern):
                r = self.code.index(pattern)
                v = self.code.index(f"verify_now({verify_tag}")
                self.assertLess(
                    v, r, f"verify_now must precede report for {pattern}")

        # bidirectional reports under a computed name, so match separately
        r = self.code.index("report(bidir_pattern(pinned)")
        v = self.code.index('verify_now("bidirectional_d2h"')
        self.assertLess(v, r, "verify_now must precede bidirectional report")

    def test_report_takes_failure_count_and_suppresses_bandwidth(self):
        self.assertIn("long long failures", self.code)
        self.assertIn("ERROR=payload_verification_failed", self.code)

    def test_pageable_bidirectional_is_not_labelled_duplex(self):
        """R2-2: the pageable arm cannot overlap, so it must not say so.

        cudaMemcpyAsync from unpinned memory is synchronous w.r.t. the host.
        Documenting that in a comment was rejected: the emitted record still
        read PATTERN=bidirectional with two copies, and no downstream parser
        reads comments.
        """
        self.assertIn("bidirectional_serialized", self.code)
        self.assertIn("OVERLAP=", self.code)
        # the flag must be computed, not hardcoded
        self.assertIn("(copies > 1 && pinned) ? 1 : 0", self.code)

    def test_failures_are_reduced_across_ranks_before_reporting(self):
        """R3-1: a corrupt NON-ROOT rank must suppress the aggregate line.

        Round-3 review: verify_now() returns only the calling rank's count
        and report() runs on rank 0 with rank 0's count, so corruption on
        rank 1..N still printed a GBPS line for a cross-rank aggregate
        measurement. The trailing MPI_Reduce surfaced it only after every
        record had been emitted.
        """
        self.assertIn("global_failures", self.code)
        self.assertIn("MPI_Allreduce", self.code)
        # every pattern must reduce before reporting
        self.assertEqual(self.code.count("global_failures(f)"), 4,
                         "all four patterns must reduce the failure count")
        for marker in ('report("h2d"', 'report("d2h"', 'report("d2d"',
                       'report(bidir_pattern(pinned)'):
            with self.subTest(marker=marker):
                r = self.code.index(marker)
                g = self.code.rfind("global_failures(f)", 0, r)
                self.assertNotEqual(
                    g, -1, f"{marker} must be preceded by a global reduce")

    def test_allreduce_not_reduce_so_all_ranks_participate(self):
        # MPI_Reduce with only root reading would leave non-root ranks
        # unsynchronised; the gate must be collective.
        idx = self.code.index("static long long global_failures")
        body = self.code[idx:idx + 600]
        self.assertIn("MPI_Allreduce", body)

    def test_no_stale_verify_helper_remains(self):
        # The pre-round-2 verify() returned void; both cannot coexist.
        self.assertNotIn("static void verify(", self.code)


class PciFixedMultiRank(unittest.TestCase):
    """Behavioural proof of R3-1 with real MPI ranks.

    Skips when no MPI compiler/launcher is present. On Tara this runs for
    real: 2 ranks, corruption injected on rank 1 only, and rank 0 must NOT
    print a bandwidth number.
    """

    @classmethod
    def setUpClass(cls):
        cls.mpicxx = shutil.which("mpicxx") or shutil.which("mpic++")
        cls.mpirun = shutil.which("mpiexec") or shutil.which("mpirun")
        if not cls.mpicxx or not cls.mpirun:
            raise unittest.SkipTest("MPI toolchain not available")
        cls.tmp = tempfile.mkdtemp(prefix="pfc_mpi_")

        # The launcher existing on PATH does not mean it can launch here.
        # On Cray/PALS login nodes mpiexec resolves but the shepherd is
        # absent ("Couldn't execute local shepherd ... palsd"), so probe
        # with a trivial MPI program and skip rather than report a failure
        # that is really a missing job allocation.
        probe_src = os.path.join(cls.tmp, "probe.cpp")
        with open(probe_src, "w") as fh:
            fh.write('#include <mpi.h>\n#include <cstdio>\n'
                     'int main(int c,char**v){MPI_Init(&c,&v);'
                     'std::printf("PROBE_OK\\n");MPI_Finalize();return 0;}\n')
        probe_bin = os.path.join(cls.tmp, "probe")
        p = subprocess.run([cls.mpicxx, "-std=c++17", probe_src,
                            "-o", probe_bin],
                           capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            raise unittest.SkipTest(f"MPI build unavailable: {p.stderr[-300:]}")
        p = subprocess.run([cls.mpirun, "-n", "2", probe_bin],
                           capture_output=True, text=True, timeout=300)
        if p.returncode != 0 or "PROBE_OK" not in p.stdout:
            raise unittest.SkipTest(
                "MPI launcher cannot start ranks here (needs a job "
                f"allocation): {(p.stderr or p.stdout)[-200:]}")

        with open(os.path.join(cls.tmp, "cuda_runtime.h"), "w") as fh:
            fh.write(CUDA_STUB)
        src = os.path.join(cls.tmp, "src.cpp")
        with open(SRC) as fh:
            code = fh.read()
        # Inject corruption on rank 1 only, immediately before the h2d
        # verify so the bytes rank 1 checks are wrong.
        needle = ('        f = verify_now("h2d", ref.data(), got.data(), '
                  'n_elems, world_rank);')
        assert code.count(needle) == 1, "injection point not found"
        code = code.replace(
            needle,
            '        if (world_rank == 1 && getenv("PFC_CORRUPT_RANK1")) '
            'got[0] = ~got[0];\n' + needle)
        with open(src, "w") as fh:
            fh.write(code)
        cls.bin = os.path.join(cls.tmp, "pfc_mpi")
        p = subprocess.run(
            [cls.mpicxx, "-std=c++17", "-O0", "-x", "c++", src,
             f"-I{cls.tmp}", "-o", cls.bin],
            capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            raise unittest.SkipTest(f"MPI build failed: {p.stderr[-400:]}")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "tmp", None):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, env_extra):
        env = dict(os.environ)
        env.update({"DLCOMM_TRANSFER_BYTES": "4096",
                    "DLCOMM_TRANSFER_ITERS": "2"})
        env.update(env_extra)
        p = subprocess.run(
            [self.mpirun, "-n", "2", self.bin],
            capture_output=True, text=True, timeout=300, env=env)
        return p.returncode, p.stdout + p.stderr

    def test_clean_two_rank_run_reports_bandwidth(self):
        rc, out = self._run({})
        self.assertIn("PATTERN=h2d", out)
        self.assertIn("GBPS=", out)

    def test_corrupt_non_root_rank_suppresses_bandwidth(self):
        rc, out = self._run({"PFC_CORRUPT_RANK1": "1"})
        h2d = [ln for ln in out.splitlines() if "PATTERN=h2d" in ln]
        self.assertTrue(h2d, f"no h2d record emitted:\n{out[-800:]}")
        for ln in h2d:
            self.assertNotIn(
                "GBPS=", ln,
                "rank-0-clean/rank-1-corrupt must not print bandwidth")
            self.assertIn("ERROR=payload_verification_failed", ln)
        self.assertNotEqual(rc, 0, "a corrupt run must exit non-zero")


if __name__ == "__main__":
    unittest.main(verbosity=2)
