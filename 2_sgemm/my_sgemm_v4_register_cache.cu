#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define A(i, j) a[((i) * (n)) + (j)]
#define B(i, j) b[((i) * (n)) + (j)]

void random_matrix(int m, int n, float* a)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
#if 1   
            A(i, j) = 2.0 * (float) drand48() - 1.0;
#else
            A(i, j) = (j - i) % 3;
#endif
        }
    }
}

float compare_matrices(int m, int n, float* a, float* b)
{
    float max_diff = 0.0f, diff;
    int printed = 0;

    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            diff = abs(A(i, j) - B(i, j));
            max_diff = (diff > max_diff) ? diff : max_diff;
            if(0 == printed)
            {
                if(max_diff > 0.5f || max_diff < -0.5f)
                {
                    printf("A(%d, %d) = %f, B(%d, %d) = %f\n", i, j, A(i, j), i, j, B(i, j));
                    printed = 1;
                }
            }
        }
    }
    return max_diff;
}

void cpu_sgemm(float* A, float* B, float* C, const int M, const int N, const int K)
{
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            float temp = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                temp += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = temp;
        }
    }
}

// SGEMM kernel with register-based cache optimization
// Each thread computes a 4x4 block of C matrix using register arrays
template <unsigned int BLOCK_SIZE>
__global__ void gpu_sgemm(float* A, float* B, float* C, const int M, const int N, const int K)
{
    // Shared memory for data tiles
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    // Thread block and thread indices
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    // Each thread computes THREAD_TILE_M x THREAD_TILE_N elements
    const int THREAD_TILE_M = 4;
    const int THREAD_TILE_N = 4;
    
    // Register arrays to cache intermediate results (register-based L2 cache simulation)
    float reg_C[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
    float reg_A[THREAD_TILE_M];  // Cache for A tile
    float reg_B[THREAD_TILE_N];  // Cache for B tile
    
    // Global matrix positions for this thread's tile
    int global_row_start = block_row * BLOCK_SIZE + thread_row * THREAD_TILE_M;
    int global_col_start = block_col * BLOCK_SIZE + thread_col * THREAD_TILE_N;
    
    // Sliding window over K dimension
    for(int tile_k = 0; tile_k < K; tile_k += BLOCK_SIZE)
    {
        // Collaborative loading of A and B tiles into shared memory
        // Each thread loads multiple elements using strided access
        
        // Load A tile: A[block_row*BLOCK_SIZE:(block_row+1)*BLOCK_SIZE, tile_k:tile_k+BLOCK_SIZE]
        for(int load_offset = 0; load_offset < BLOCK_SIZE; load_offset += blockDim.y * THREAD_TILE_M)
        {
            for(int load_stride = 0; load_stride < THREAD_TILE_M; ++load_stride)
            {
                int load_row = thread_row * THREAD_TILE_M + load_stride + load_offset;
                if(load_row < BLOCK_SIZE)
                {
                    for(int col_stride = 0; col_stride < THREAD_TILE_N; ++col_stride)
                    {
                        int load_col = thread_col * THREAD_TILE_N + col_stride;
                        if(load_col < BLOCK_SIZE)
                        {
                            int global_row = block_row * BLOCK_SIZE + load_row;
                            int global_col = tile_k + load_col;
                            
                            if(global_row < M && global_col < K)
                            {
                                A_shared[load_row][load_col] = A[global_row * K + global_col];
                            }
                            else
                            {
                                A_shared[load_row][load_col] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
        
        // Load B tile: B[tile_k:tile_k+BLOCK_SIZE, block_col*BLOCK_SIZE:(block_col+1)*BLOCK_SIZE]
        for(int load_offset = 0; load_offset < BLOCK_SIZE; load_offset += blockDim.x * THREAD_TILE_N)
        {
            for(int load_stride = 0; load_stride < THREAD_TILE_N; ++load_stride)
            {
                int load_col = thread_col * THREAD_TILE_N + load_stride + load_offset;
                if(load_col < BLOCK_SIZE)
                {
                    for(int row_stride = 0; row_stride < THREAD_TILE_M; ++row_stride)
                    {
                        int load_row = thread_row * THREAD_TILE_M + row_stride;
                        if(load_row < BLOCK_SIZE)
                        {
                            int global_row = tile_k + load_row;
                            int global_col = block_col * BLOCK_SIZE + load_col;
                            
                            if(global_row < K && global_col < N)
                            {
                                B_shared[load_row][load_col] = B[global_row * N + global_col];
                            }
                            else
                            {
                                B_shared[load_row][load_col] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute using register-cached data
        // This is the key optimization: minimize shared memory accesses by caching in registers
        for(int k = 0; k < BLOCK_SIZE; ++k)
        {
            // Cache A elements in registers (simulate L2 cache)
            for(int tm = 0; tm < THREAD_TILE_M; ++tm)
            {
                int shared_row = thread_row * THREAD_TILE_M + tm;
                if(shared_row < BLOCK_SIZE)
                {
                    reg_A[tm] = A_shared[shared_row][k];
                }
                else
                {
                    reg_A[tm] = 0.0f;
                }
            }
            
            // Cache B elements in registers (simulate L2 cache)
            for(int tn = 0; tn < THREAD_TILE_N; ++tn)
            {
                int shared_col = thread_col * THREAD_TILE_N + tn;
                if(shared_col < BLOCK_SIZE)
                {
                    reg_B[tn] = B_shared[k][shared_col];
                }
                else
                {
                    reg_B[tn] = 0.0f;
                }
            }
            
            // Perform computation using cached register values
            // This maximizes data reuse and minimizes memory accesses
            for(int tm = 0; tm < THREAD_TILE_M; ++tm)
            {
                for(int tn = 0; tn < THREAD_TILE_N; ++tn)
                {
                    reg_C[tm][tn] += reg_A[tm] * reg_B[tn];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    for(int tm = 0; tm < THREAD_TILE_M; ++tm)
    {
        for(int tn = 0; tn < THREAD_TILE_N; ++tn)
        {
            int global_row = global_row_start + tm;
            int global_col = global_col_start + tn;
            
            if(global_row < M && global_col < N)
            {
                C[global_row * N + global_col] = reg_C[tm][tn];
            }
        }
    }
}

int main()
{
    constexpr int m = 512;
    constexpr int n = 512;
    constexpr int k = 512;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float* matrix_A_host = (float*) malloc(mem_size_A);
    float* matrix_B_host = (float*) malloc(mem_size_B);
    
    float* matrix_C_host_gpu_calc = (float*) malloc(mem_size_C);
    float* matrix_C_host_cpu_calc = (float*) malloc(mem_size_C);

    // Check memory allocation
    if (!matrix_A_host || !matrix_B_host || !matrix_C_host_gpu_calc || !matrix_C_host_cpu_calc) {
        printf("Error: Failed to allocate host memory\n");
        return -1;
    }

    // Initialize matrices A and B
    printf("Initializing matrices...\n");
    random_matrix(m, k, matrix_A_host);  // A: m x k
    random_matrix(k, n, matrix_B_host);  // B: k x n

    memset(matrix_C_host_cpu_calc, 0, mem_size_C);
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);

    float* matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void**)&matrix_A_device, mem_size_A);
    cudaMalloc((void**)&matrix_B_device, mem_size_B);
    cudaMalloc((void**)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);
    
    // CPU computation for verification
    float cpu_start = 0.0f, cpu_end = 0.0f;
    cpu_start = static_cast<float>(clock()) / CLOCKS_PER_SEC;
    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);
    cpu_end = static_cast<float>(clock()) / CLOCKS_PER_SEC;
    printf("CPU sgemm time: %f ms\n", (cpu_end - cpu_start) * 1000.0f);
    
    // GPU computation with register cache optimization
    constexpr int BLOCK = 16;
    dim3 block(BLOCK/4, BLOCK/4);  // 4x4 threads, each computes 4x4 tile, total 16x16
    dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gpu_sgemm<BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU sgemm time (register cache): %f ms\n", milliseconds);
    
    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    float diff = compare_matrices(m, n, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);

    if(diff > 0.5f || diff < -0.5f)
    {
        printf("Difference detected: %f\n", diff);
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Right!\n");
    }

    printf("Register cache optimization SGEMM completed successfully!\n");
    
    // Calculate performance metrics
    double gflops = (2.0 * m * n * k) / (milliseconds * 1e6);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Free CUDA memory
    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    
    // Free host memory
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_gpu_calc);
    free(matrix_C_host_cpu_calc);
    
    return 0;
}
