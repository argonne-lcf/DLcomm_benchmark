/* Lifecycle cost of a NCCL collective: init, first op, steady op, teardown.
 *
 * NCCL splits into more phases than MPI, and they matter separately:
 *
 *   bootstrap   MPI_Init + broadcasting the ncclUniqueId (out-of-band)
 *   init        ncclCommInitRank: the real cost -- ring/tree construction,
 *               transport selection, plugin load, connection setup
 *   first_op    first ncclAllReduce: channel + buffer registration
 *   steady      median of N subsequent identical collectives
 *   destroy     ncclCommDestroy: connection teardown
 *
 * Every collective is validated. NCCL allreduce on this machine is known to
 * return silently wrong data at 1 MiB over CXI, so an unvalidated timing
 * number here would be worthless.
 */
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDACHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("RESULT error=cuda_%s\n", cudaGetErrorString(e)); return 2; } } while (0)
#define NCCLCHK(x) do { ncclResult_t r = (x); if (r != ncclSuccess) { \
    printf("RESULT error=nccl_%s\n", ncclGetErrorString(r)); return 2; } } while (0)

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    long bytes = 16777216;
    int iters = 20;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bytes") && i + 1 < argc) bytes = atol(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }

    double tb = now_us();
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const char *lr = getenv("PALS_LOCAL_RANKID");
    int local = lr ? atoi(lr) : 0;
    CUDACHK(cudaSetDevice(local));

    ncclUniqueId id;
    if (rank == 0) NCCLCHK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    double t_bootstrap = now_us() - tb;

    cudaStream_t stream;
    CUDACHK(cudaStreamCreate(&stream));

    /* The measurement of interest: everything NCCL does before it can move
     * a single byte. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now_us();
    ncclComm_t comm;
    NCCLCHK(ncclCommInitRank(&comm, world, id, rank));
    double t_init = now_us() - t0;

    size_t n = (size_t)bytes / sizeof(float);
    float *sbuf, *rbuf;
    CUDACHK(cudaMalloc(&sbuf, n * sizeof(float)));
    CUDACHK(cudaMalloc(&rbuf, n * sizeof(float)));

    /* Each rank contributes (rank+1) so the reduced value is a known
     * constant and corruption is detectable. */
    float fill = (float)(rank + 1);
    float *host = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) host[i] = fill;
    CUDACHK(cudaMemcpy(sbuf, host, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = now_us();
    NCCLCHK(ncclAllReduce(sbuf, rbuf, n, ncclFloat, ncclSum, comm, stream));
    CUDACHK(cudaStreamSynchronize(stream));
    double t_first = now_us() - t1;

    double *samples = (double *)malloc(sizeof(double) * iters);
    for (int i = 0; i < iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double s = now_us();
        NCCLCHK(ncclAllReduce(sbuf, rbuf, n, ncclFloat, ncclSum, comm, stream));
        CUDACHK(cudaStreamSynchronize(stream));
        samples[i] = now_us() - s;
    }
    qsort(samples, iters, sizeof(double), cmp_d);
    double steady = samples[iters / 2];
    double p90 = samples[(int)(iters * 0.9)];

    /* Validate: a fast wrong answer is not a result. */
    CUDACHK(cudaMemcpy(host, rbuf, n * sizeof(float), cudaMemcpyDeviceToHost));
    double expect = (double)world * (world + 1) / 2.0;
    long bad = 0;
    for (size_t i = 0; i < n; i++) if (host[i] != (float)expect) bad++;
    long bad_total = 0;
    MPI_Reduce(&bad, &bad_total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    CUDACHK(cudaFree(sbuf));
    CUDACHK(cudaFree(rbuf));

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = now_us();
    NCCLCHK(ncclCommDestroy(comm));
    double t_destroy = now_us() - t2;

    if (rank == 0) {
        double busbw = 2.0 * (world - 1) / world * bytes / (steady * 1e-6) / 1e9;
        if (bad_total) {
            printf("RESULT lib=nccl world=%d bytes=%ld VALIDATION_FAILED wrong=%ld\n",
                   world, bytes, bad_total);
        } else {
            printf("RESULT lib=nccl world=%d bytes=%ld bootstrap_us=%.1f "
                   "init_us=%.1f first_us=%.1f steady_us=%.1f p90_us=%.1f "
                   "destroy_us=%.1f gbps=%.2f\n",
                   world, bytes, t_bootstrap, t_init, t_first, steady, p90,
                   t_destroy, busbw);
        }
    }
    free(host);
    free(samples);
    MPI_Finalize();
    return bad_total ? 3 : 0;
}
