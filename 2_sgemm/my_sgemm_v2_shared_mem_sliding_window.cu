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

//shared memory sgemm kernel
template <unsigned int BLOCK_SIZE>
__global__ void gpu_sgemm(float* A, float* B, float* C, const int M, const int N, const int K)
{
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    
    // Load data into shared memory with proper boundary checking
    for(int s = 0; s < K; s += BLOCK_SIZE)
    {
        // Load A matrix: A[blockIdx.y * BLOCK_SIZE + threadIdx.y][s + threadIdx.x]
        if ((blockIdx.y * BLOCK_SIZE + threadIdx.y) < M && (s + threadIdx.x) < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * K + (s + threadIdx.x)];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B matrix: B[s + threadIdx.y][blockIdx.x * BLOCK_SIZE + threadIdx.x]
        if ((s + threadIdx.y) < K && (blockIdx.x * BLOCK_SIZE + threadIdx.x) < N) {
            B_shared[threadIdx.y][threadIdx.x] = B[(s + threadIdx.y) * N + (blockIdx.x * BLOCK_SIZE + threadIdx.x)];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for(int k = 0; k < BLOCK_SIZE; ++k)
        {
            temp += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }   
        
        __syncthreads();
    }
    
    if (m < M && n < N) {
        C[m * N + n] = temp;
    }
}

int main()
{
    constexpr int m = 1024;
    constexpr int n = 1024;
    constexpr int k = 1024;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float* matrix_A_host = (float*) malloc(mem_size_A);
    float* matrix_B_host = (float*) malloc(mem_size_B);
    
    float* matrix_C_host_gpu_calc = (float*) malloc(mem_size_C);
    float* matrix_C_host_cpu_calc = (float*) malloc(mem_size_C);

    // 检查内存分配是否成功
    if (!matrix_A_host || !matrix_B_host || !matrix_C_host_gpu_calc || !matrix_C_host_cpu_calc) {
        printf("Error: Failed to allocate host memory\n");
        return -1;
    }

    // 初始化矩阵 A 和 B
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
    
    //cpu计算时间
    float cpu_start = 0.0f, cpu_end = 0.0f;
    cpu_start = static_cast<float>(clock()) / CLOCKS_PER_SEC;
    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);
    cpu_end = static_cast<float>(clock()) / CLOCKS_PER_SEC;
    printf("CPU sgemm time: %f ms\n", (cpu_end - cpu_start) * 1000.0f);
    //调用gpu sgemm kernel
    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
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
    printf("GPU sgemm time: %f ms\n", milliseconds);
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


    printf("sgemm\n");
    
    // 释放CUDA内存
    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    
    // 释放主机内存
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_gpu_calc);
    free(matrix_C_host_cpu_calc);
    
    return 0;
}