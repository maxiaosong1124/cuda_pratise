#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce(float* d_input, float* d_output)
{   
    
    //2.按照GPU的线程索引来进行实现
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float shared[THREAD_PER_BLOCK];
    shared[threadIdx.x] = d_input[tid];
    __syncthreads();
    for(int i = 1; i < blockDim.x; i *= 2)
    {
        if(threadIdx.x < blockDim.x /(i * 2))
        {
            int index = threadIdx.x * 2 * i;
            shared[index] += shared[index + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        d_output[blockIdx.x] = shared[0];
    }


    //1.按照block内的线程偏移地址来进行实现，类似于CPU的计算方式,这里是使用了连续的线程来进行计算，与之前的v1中的在进入后续迭代之后会有闲置线程不同，也可以减少分支分歧
    // __shared__ float shared[THREAD_PER_BLOCK];

    // float* input_begin = d_input + blockDim.x * blockIdx.x;
    // shared[threadIdx.x] = input_begin[threadIdx.x];
    // __syncthreads();
    // for(int i = 1; i < blockDim.x; i *= 2)
    // {
    //     if(threadIdx.x < blockDim.x / (i * 2))
    //     {
    //         //input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
    //         int index = threadIdx.x * 2 * i;
    //         shared[index] += shared[index + i];
    //     }
    //     __syncthreads();
    // }

    // if(threadIdx.x == 0)
    // {
    //     d_output[blockIdx.x] = shared[0];
    // }
}

bool check(float* output, float* result, int n)
{
    for(int i = 0; i < n; ++i)
    {
        if(abs(output[i] - result[i]) > 0.005)
        {
            printf("Error at index %d: output = %f, result = %f\n", i, output[i], result[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int N = 32 * 1024 * 1024;
    float* input = (float*)malloc(N * sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK;
    float* output = (float*)malloc(N / THREAD_PER_BLOCK * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, N / THREAD_PER_BLOCK * sizeof(float));
    float* result = (float*)malloc(N / THREAD_PER_BLOCK * sizeof(float));

    //初始化
    for(int i = 0; i < N; ++i)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    //cpu计算结果
    for(int i = 0; i < block_num; ++i)
    {
        float cur = 0;
        for(int j = 0; j < THREAD_PER_BLOCK; ++j)
        {
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(N / THREAD_PER_BLOCK);
    dim3 block(THREAD_PER_BLOCK);

    reduce<<<grid, block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, N / THREAD_PER_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);
    if(check(output, result, block_num))
    {
        printf("The result is right\n");
    }
    else
    {
        printf("The result is wrong\n");
    }

    free(input);
    free(output);

    free(result);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaDeviceReset();

    return 0;
}