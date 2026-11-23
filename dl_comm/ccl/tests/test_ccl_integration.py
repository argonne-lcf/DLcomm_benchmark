"""Source-contract regressions, NOT a GPU/NCCL execution test."""
import pathlib
import unittest

SOURCE = (pathlib.Path(__file__).parents[1] / "ccl_bench.cpp").read_text()
NCCL = SOURCE.split("  // NCCL / RCCL path.", 1)[1]

class IntegrationContracts(unittest.TestCase):
    def test_each_iteration_prepares_times_then_verifies_full_output(self):
        self.assertIn("dlcomm_correctness::verify", NCCL)
        prepare = NCCL.index("GPU_COPY_H2D(sbuf")
        start = NCCL.index("auto t0 = clk::now()", prepare)
        stop = NCCL.index("auto t1 = clk::now()", start)
        copy = NCCL.index("GPU_COPY_D2H", stop)
        check = NCCL.index("dlcomm_correctness::verify", copy)
        self.assertLess(prepare, start)
        self.assertLess(stop, copy)
        self.assertLess(copy, check)
        self.assertIn("expected.output_count()", NCCL)
        self.assertNotIn("size_t idxs[3]", NCCL)
    def test_collective_coverage_and_selection(self):
        for op in ("allreduce", "allgather", "broadcast", "reduce", "reduce_scatter", "sendrecv", "alltoall"):
            self.assertIn('"' + op + '"', NCCL)
        for api in ("ncclAllReduce", "ncclAllGather", "ncclBroadcast", "ncclReduce", "ncclReduceScatter", "ncclSend", "ncclRecv", "ncclGroupStart", "ncclGroupEnd"):
            self.assertIn("NCCL_CHECK(" + api + "(", NCCL)
        self.assertIn("if (!want(op)) continue;", NCCL)
    def test_all_rank_failure_aggregation_and_rows(self):
        self.assertIn("MPI_Allreduce", NCCL)
        self.assertIn("tot_failures", NCCL)
        self.assertIn("return tot_failures > 0 ? 1 : 0;", NCCL)
        self.assertIn('" CHECKS="', NCCL)
        self.assertIn('" FAILURES="', NCCL)
    def test_gpu_and_nccl_statuses_and_rccl_copies(self):
        for api in ("GPU_MALLOC", "GPU_FREE", "GPU_SYNC", "GPU_GET_DEVICE_COUNT", "GPU_SET_DEVICE", "GPU_COPY_H2D", "GPU_COPY_D2H", "GPU_MEMSET"):
            self.assertIn("GPU_CHECK(" + api + "(", NCCL)
        for api in ("ncclGetUniqueId", "ncclCommInitRank", "ncclCommGetAsyncError", "ncclCommDestroy"):
            self.assertIn("NCCL_CHECK(" + api + "(", NCCL)
        self.assertNotIn("cudaMemcpy(", NCCL)
        self.assertIn("hipMemcpy", SOURCE)
    def test_tail_injection_writes_device_after_timing(self):
        self.assertIn("DLCOMM_INJECT_CORRUPTION", NCCL)
        self.assertIn("output_count - 1", NCCL)
        self.assertIn("injected = true", NCCL)
        self.assertIn('" INJECTED="', NCCL)

if __name__ == "__main__":
    unittest.main()
