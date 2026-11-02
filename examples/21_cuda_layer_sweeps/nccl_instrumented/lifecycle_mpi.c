/* Lifecycle cost of an MPI collective: init, first op, steady op, teardown.
 *
 * The four phases are timed separately because they have very different
 * causes and very different fixes:
 *
 *   init        MPI_Init: process/rank wireup, no network traffic yet
 *   first_op    the FIRST collective: lazily builds communication state,
 *               registers memory, opens endpoints. Nearly always the most
 *               expensive single operation in the job.
 *   steady      median of N subsequent identical collectives
 *   destroy     MPI_Finalize: endpoint teardown
 *
 * Device buffers throughout (GTL build + MPICH_GPU_SUPPORT_ENABLED=1), so
 * this is comparable to the NCCL and torch numbers rather than measuring a
 * host staging copy.
 *
 * Output is one machine-readable line from rank 0.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    /* Select the device BEFORE MPI_Init so MPICH builds its GPU/CXI
     * registration state against the device this rank will actually use.
     * Doing it afterwards fails inter-node with cxil_map write errors. */
    const char *lr = getenv("PALS_LOCAL_RANKID");
    int local = lr ? atoi(lr) : 0;
    if (cudaSetDevice(local) != cudaSuccess) {
        printf("RESULT error=cudaSetDevice_failed\n");
        return 2;
    }
    cudaFree(0);  /* force context creation now, not lazily inside MPI */

    /* Clock starts before MPI_Init: that call is the thing being measured. */
    double t0 = now_us();
    MPI_Init(&argc, &argv);
    double t_init = now_us() - t0;

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t n = (size_t)bytes / sizeof(float);
    float *sbuf = NULL, *rbuf = NULL;
    if (cudaMalloc((void **)&sbuf, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&rbuf, n * sizeof(float)) != cudaSuccess) {
        if (!rank) printf("RESULT error=cudaMalloc_failed\n");
        MPI_Finalize();
        return 2;
    }
    cudaMemset(sbuf, 0, n * sizeof(float));
    cudaDeviceSynchronize();

    /* Barrier so the first-op timing measures the collective itself and not
     * the spread in arrival times between ranks. */
    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = now_us();
    MPI_Allreduce(sbuf, rbuf, (int)n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    double t_first = now_us() - t1;

    double *samples = (double *)malloc(sizeof(double) * iters);
    for (int i = 0; i < iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double s = now_us();
        MPI_Allreduce(sbuf, rbuf, (int)n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        cudaDeviceSynchronize();
        samples[i] = now_us() - s;
    }
    qsort(samples, iters, sizeof(double), cmp_d);
    double steady = samples[iters / 2];
    double p90 = samples[(int)(iters * 0.9)];

    cudaFree(sbuf);
    cudaFree(rbuf);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = now_us();
    MPI_Finalize();
    double t_destroy = now_us() - t2;

    if (rank == 0) {
        double busbw = 2.0 * (world - 1) / world * bytes / (steady * 1e-6) / 1e9;
        printf("RESULT lib=mpi world=%d bytes=%ld init_us=%.1f first_us=%.1f "
               "steady_us=%.1f p90_us=%.1f destroy_us=%.1f gbps=%.2f\n",
               world, bytes, t_init, t_first, steady, p90, t_destroy, busbw);
    }
    free(samples);
    return 0;
}
