#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <chrono>

void softmax_cpu(const float* d_in, float* d_out, int N, int C)
{ 
    for(int i = 0; i < N; ++i)
    {
        const float* in_row = d_in + i * C;
        float* out_row = d_out + i * C;
        float maxval = -INFINITY;
        for(int j = 0; j < C; ++j)
        {
            if(maxval < in_row[j])
            {
                maxval = in_row[j];
            }
        }
        float sum = 0.f;
        for(int j = 0; j < C; ++j)
        {
            out_row[j] = expf(in_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.f / float(sum);
        for(int j = 0; j < C; ++j)
        {
            out_row[j] *= norm;
        }
    }
}

//cuda kernel, per block one thread
__global__ void softmax_kernel1(float* out, const float* in, int N, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
    {
        const float* in_row = in + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for(int j = 0; j < C; ++j)
        {
            if(in_row[j] > maxval)
            {
                maxval = in_row[j];
            }
        }

        float sum = 0.f;
        for(int j = 0; j < C; ++j)
        {
            out_row[j] = expf(in_row[j] - maxval);
            sum += out_row[j];
        }

        for(int j = 0; j < C; ++j)
        {
            out_row[j] /= float(sum);
        }
    }
}

//cuda kernel2, shared mem， per block 128 threads
__global__ void softmax_kernel2(float* out, const float* in, int N, int C)
{
    extern __shared__ float shared[];
    int bx = blockIdx.x; // range [0, N)
    int tid = threadIdx.x;// range[0, block_size)
    int block_size = blockDim.x;
    const float* x = in + bx * C; //idx-th row of in
    //thread coarsening
    float maxval = -INFINITY;
    for(int i = tid; i < C; i += block_size)
    {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;
    //__syncthreads();
    for(int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if(tid < stride)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();

    float offset = shared[0];
    //compute expf and write the result to global mem
    for(int i = tid; i < C; i += block_size)
    {
        out[bx * C  + i] = expf(x[i] - offset);
    }
    __syncthreads();

    //thread coarsening again, for the sum
    x = out + bx * C;
    float sumval = 0.f;
    for(int i = tid; i < C; i += block_size)
    {
        sumval += x[i];
    }
    shared[tid] = sumval;
    //__syncthreads();
    //reductions
    for(int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if(tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
    }
    __syncthreads();
    float sum = shared[0];

    for(int i = tid; i < C; i += block_size)
    {
        out[bx * C + i] = x[i] / sum;
    }
}


__device__ float warpReduceMax(float val)
{
    for(int offset = 16; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    return val;
}

__device__ float warpReduceSum(float val)
{
    for(int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    return val;
}

// per block 32 threads, 引入了warp shfl的算法，减少了共享内存和全局内存的同步开销，使用寄存器进行通信
__global__ void softmax_kernel3(float* out, float* in, int N, int C)
{
    int bx = blockIdx.x; // range [0, N)
    int tid = threadIdx.x;// range[0, block_size)
    const float* x = in + bx * C; //idx-th row of in
    //thread coarsening
    float maxval = -INFINITY;
    for(int i = tid; i < C; i += blockDim.x)
    {
        maxval = fmaxf(maxval, x[i]);
    }
    
    maxval = warpReduceMax(maxval);

    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);

    //compute expf and write the result to global mem
    for(int i = tid; i < C; i += blockDim.x)
    {
        out[bx * C  + i] = expf(x[i] - offset);
    }

    //thread coarsening again, for the sum
    x = out + bx * C;
    float sumval = 0.f;
    for(int i = tid; i < C; i += blockDim.x)
    {
        sumval += x[i];
    }

    sumval = warpReduceSum(sumval);
    float sum = __shfl_sync(0xFFFFFFFF, sumval, 0);

    for(int i = tid; i < C; i += blockDim.x)
    {
        out[bx * C + i] = x[i] / sum;
    }
}

//同时引入shared mem和warp shfl
__global__ void softmax_kernel4(float* out, float* in, int N, int C)
{
    extern __shared__ float shared[];
    int bx = blockIdx.x; // range [0, N)
    int tid = threadIdx.x;// range[0, block_size)
    int block_size = blockDim.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    int warpsPerBlock = blockDim.x / 32;

    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    const float* x = in + bx * C; //idx-th row of in
    //thread coarsening
    float maxval = -INFINITY;
    for(int i = tid; i < C; i += block_size)
    {
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval);

    if(laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    if(tid == 0)
    {
        float val = maxvals[tid];
        for(int i = 1; i < warpsPerBlock; ++i)
        {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    
    float offset = maxvals[0];;
    //compute expf and write the result to global mem
    for(int i = tid; i < C; i += block_size)
    {
        out[bx * C  + i] = expf(x[i] - offset);
    }

    //thread coarsening again, for the sum
    x = out + bx * C;
    float sumval = 0.f;
    for(int i = tid; i < C; i += block_size)
    {
        sumval += x[i];
    }
    
    sumval = warpReduceSum(sumval);
    if(laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    if(tid == 0)
    {
        float val = sumvals[tid];
        for(int i = 1; i < warpsPerBlock; ++i)
        {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();

    float sum = sumvals[0];

    for(int i = tid; i < C; i += block_size)
    {
        out[bx * C + i] = x[i] / sum;
    }
}

bool compare_results(const float* cpu, const float* gpu, int N, int C, float epsilon = 1e-3f)
{
    for (int i = 0; i < N * C; ++i) {
    if (fabs(cpu[i] - gpu[i]) > epsilon) {
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  // Example: batch size N=32, classes C=4096
  int N = 32;
  int C = 4096;

  size_t num_elements = N * C;
  float *inp = (float *)malloc(num_elements * sizeof(float));
  float *out_cpu = (float *)malloc(num_elements * sizeof(float));
  float *out_gpu = (float *)malloc(num_elements * sizeof(float));

  // Initialize input with sample data
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      inp[n * C + c] = float(c);
    }
  }

  // Run CPU version and measure time
  auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_cpu(inp, out_cpu, N, C);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

  // Run GPU version and measure time using CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *d_out, *d_inp;
  cudaMalloc((void **)&d_out, N * C * sizeof(float));
  cudaMalloc((void **)&d_inp, N * C * sizeof(float));
  cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  // Launch kernel
  int blockSize = 128;
  //int blockSize = 32; 在启动kernel3的时候将blockSize设置为32，因为kernel3每个线程只能为32，每一行只用了一个warp进行计算，如果超过就会有问题
  int numBlocks = N;
  softmax_kernel2<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
  cudaEventRecord(stop);

  // Wait for the event to complete
  cudaEventSynchronize(stop);

  // Calculate milliseconds
  float gpu_time_ms = 0;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);

  // Copy result back to host
  cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_out);
  cudaFree(d_inp);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Compare results
  bool success = compare_results(out_cpu, out_gpu, N, C);
  std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

  // Print performance comparison
  std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
  std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
  std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
            << std::endl;

  // Cleanup
  free(inp);
  free(out_cpu);
  free(out_gpu);

  return 0;
}
