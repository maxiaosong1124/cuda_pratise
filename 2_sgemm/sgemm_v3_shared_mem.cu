#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)
template<const uint BLOCKSIZE>
__global__ void sgemm_coaliescing(const float* A, const float* B, float* C,
                            int M, int N, int K, float alpha, float beta)
{
    // the output block that we want to compute in this threadblock
    const uint cRow = blockIdx.y;  // 修正：blockIdx.y 对应行
    const uint cCol = blockIdx.x;  // 修正：blockIdx.x 对应列

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const uint threadCol = threadIdx.x;  // 修正：直接使用 threadIdx.x
    const uint threadRow = threadIdx.y;  // 修正：直接使用 threadIdx.y

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // 边界检查并加载 A 和 B
        if (cRow * BLOCKSIZE + threadRow < M && bkIdx + threadCol < K) {
            As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        } else {
            As[threadRow * BLOCKSIZE + threadCol] = 0.0f;
        }
        
        if (bkIdx + threadRow < K && cCol * BLOCKSIZE + threadCol < N) {
            Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        } else {
            Bs[threadRow * BLOCKSIZE + threadCol] = 0.0f;
        }

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    
    // 边界检查并写回结果
    if (cRow * BLOCKSIZE + threadRow < M && cCol * BLOCKSIZE + threadCol < N) {
        C[threadRow * N + threadCol] =
            alpha * tmp + beta * C[threadRow * N + threadCol];
    }
}

void sgemm_cpu(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta)
{
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            float tmp = 0;
            for(int k = 0; k < K; ++k)
            {
                tmp += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * tmp + beta * C[m * N + n];
        }
    }
}

static void init_matrix(float* ptr, size_t size, unsigned seed, float scale)
{
    srand(seed);
    for(size_t i = 0; i < size; ++i)
    {
        // Simple pseudo-random values in [0, scale)
        ptr[i] = scale * (float)rand() / (float)RAND_MAX;
    }
}

static size_t compare_results(const float* ref, const float* got, size_t size,
                              float tol, float* out_max_abs_err)
{
    double max_abs = 0.0;
    size_t mismatches = 0;
    for(size_t i = 0; i < size; ++i)
    {
        double diff = (double)got[i] - (double)ref[i];
        double abd = fabs(diff);
        if (abd > max_abs) max_abs = abd;
        if (abd > (double)tol) ++mismatches;
    }
    *out_max_abs_err = (float)max_abs;
    return mismatches;
}

int main(int argc, char** argv)
{
    // Problem size (M x K) * (K x N) = (M x N)
    constexpr uint BLOCKSIZE = 32;
    int M = 256;
    int N = 256;
    int K = 256;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    const float alpha = 1.0f;
    const float beta  = 0.0f; // set beta=0 for a simpler check

    size_t sizeA = (size_t)M * (size_t)K;
    size_t sizeB = (size_t)K * (size_t)N;
    size_t sizeC = (size_t)M * (size_t)N;

    float *hA = (float*)malloc(sizeA * sizeof(float));
    float *hB = (float*)malloc(sizeB * sizeof(float));
    float *hC = (float*)malloc(sizeC * sizeof(float));
    float *hC_ref = (float*)malloc(sizeC * sizeof(float));
    if (!hA || !hB || !hC || !hC_ref) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    init_matrix(hA, sizeA, 123u, 1.0f);
    init_matrix(hB, sizeB, 456u, 1.0f);
    init_matrix(hC, sizeC, 789u, 1.0f); // initial C for beta term
    // Copy initial C to reference buffer
    for (size_t i = 0; i < sizeC; ++i) hC_ref[i] = hC[i];

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dB, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice));

    // Use 2D block layout for shared memory version
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);

    // Optional timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    sgemm_coaliescing<32><<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    sgemm_cpu(hA, hB, hC_ref, M, N, K, alpha, beta);

    const float tol = 5e-3f; // 0.005
    float max_abs_err = 0.0f;
    size_t mismatches = compare_results(hC_ref, hC, sizeC, tol, &max_abs_err);

    printf("SGEMM shared memory GPU vs CPU\n");
    printf("Dims: M=%d N=%d K=%d, block=(%d,%d) grid=(%d,%d)\n",
           M, N, K, block.x, block.y, grid.x, grid.y);
    printf("Kernel time: %.3f ms\n", ms);
    printf("Abs tolerance: %.6f\n", tol);
    printf("Max abs error: %.6e\n", max_abs_err);
    printf("Mismatches: %zu / %zu\n", mismatches, sizeC);

    bool ok = (mismatches == 0);
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hC_ref);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}