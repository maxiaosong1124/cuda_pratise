#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define THREAD_PER_BLOCK 256

__device__ void warp_reduce(volatile float* cache, unsigned int tid)
{
        cache[tid] += cache[tid + 32];
        cache[tid] += cache[tid + 16];
        cache[tid] += cache[tid + 8];
        cache[tid] += cache[tid + 4];
        cache[tid] += cache[tid + 2];
        cache[tid] += cache[tid + 1];
}

__global__ void reduce(float* d_input, float* d_output)
{
     __shared__ float shared[THREAD_PER_BLOCK];

    float* input_begin = d_input + blockDim.x * blockIdx.x * 2;
    shared[threadIdx.x]  = input_begin[threadIdx.x] + input_begin[threadIdx.x + blockDim.x];
    __syncthreads();

    //1.使用宏对循环进行完全展开
    // #pragma unroll
    // for(int i = blockDim.x / 2; i > 32; i /= 2)
    // {
    //     if(threadIdx.x < i)
    //     {
    //         shared[threadIdx.x] += shared[threadIdx.x + i];
    //     }
    //     __syncthreads();
    // }
    //2.手动展开循环
    if(THREAD_PER_BLOCK >= 512) //为了支持更大的线程块
    {
        if(threadIdx.x < 256)
        {
            shared[threadIdx.x] += shared[threadIdx.x + 256];
        }
        __syncthreads();
    }

    if(THREAD_PER_BLOCK >= 256)
    {
        if(threadIdx.x < 128)
        {
            shared[threadIdx.x] += shared[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if(THREAD_PER_BLOCK >= 128)
    {
        if(threadIdx.x < 64)
        {
            shared[threadIdx.x] += shared[threadIdx.x + 64];
            __syncthreads();
        }
    }

    if(threadIdx.x < 32)
    {
        warp_reduce(shared, threadIdx.x);
    }

    if(threadIdx.x == 0)
    {
        d_output[blockIdx.x] = shared[0];
    }
}

bool check_result(float* result, float* output, int n)
{
    for(int i = 0; i < n; ++i)
    {
        if(abs(result[i] - output[i]) > 0.005)
        {
            printf("The ans is wrong!\n");
            printf("The result is at index %d: %f, but the output is %f\n", i, result[i], output[i]);
            return false;
        }
    }
    printf("The ans is right!\n");
    return true;
}

int main()
{
    const int N = 32 * 1024 * 1024;

    float* input = (float*) malloc (N * sizeof(float));
    float* d_input = nullptr;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK / 2;
    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output = nullptr;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float* result = (float*) malloc(block_num * sizeof(float));
    
    for(int i = 0; i < N; ++i)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    //cpu_calc
    for(int i = 0; i < block_num; ++i)
    {
        float cur = 0;
        for(int j = 0; j < 2 * THREAD_PER_BLOCK; ++j)
        {
            cur += input[i * 2 * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    //copy input data to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(block_num);
    dim3 block(THREAD_PER_BLOCK);

    //launch kernel
    reduce<<<grid, block>>>(d_input, d_output);

    //copy output data to host
    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    //check result
    bool is_correct = check_result(result, output, block_num);
    if (is_correct) {
        printf("All results are correct!\n");
    } else {
        printf("There are errors in the results.\n");
    }


    //free memory
    free(input);
    free(output);
    free(result);

    cudaFree(d_input);
    cudaFree(d_output);


    return 0;
}
