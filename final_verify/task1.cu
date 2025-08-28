#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 新增：检查核函数启动错误
#define CHECK_KERNEL_LAUNCH() \
    do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

// ==================== 主测试函数 ====================
void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "=== GPU设备信息 ===\n";
    std::cout << "设备名称: " << prop.name << "\n";
    std::cout << "计算能力: " << prop.major << "." << prop.minor << "\n";
    std::cout << "全局内存: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
    std::cout << "SM数量: " << prop.multiProcessorCount << "\n";
    std::cout << "最大线程/块: " << prop.maxThreadsPerBlock << "\n\n";
}

// ==================== 向量点积 ====================
__global__ void vector_mul(float* A, float* B, float* C, int M)
{
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = (gtid < M) ? A[gtid] * B[gtid] : 0.0f;
    __syncthreads();
    for(int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if(threadIdx.x < i)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        atomicAdd(C, sdata[0]);
    }
}

bool test_vector_mul(int M)
{
    float* A = (float*)malloc(sizeof(float) * M);
    float* B = (float*)malloc(sizeof(float) * M);


    for(int i = 0; i < M; ++i)
    {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }
    float sum = 0.f;
    for(int i = 0; i < M; ++i)
    {
        sum += A[i] * B[i];
    }

    // 修复：正确定义设备指针和主机结果变量
    float h_C;
    float* d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float)* M));  // 修复：使用设备指针
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float)* M));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float)));

    // 修复：修正memcpy参数顺序和指针
    CUDA_CHECK(cudaMemcpy(d_A, A, M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float)));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x );
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vector_mul<<<blocksPerGrid, threadsPerBlock, threadsPerBlock.x * sizeof(float)>>>(d_A, d_B, d_C, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "vector dot cost " << elapsedTime << "ms" << std::endl;

    // 检查核函数错误
    CHECK_KERNEL_LAUNCH();

    // 修复：正确拷贝结果到主机
    CUDA_CHECK(cudaMemcpy(&h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(A);
    free(B);


    if(fabs(h_C - sum) < 0.005f)  // 修复：使用正确的主机变量
    {
        return true;
    }
    return false;
}

// ==================== 向量L2归一化算子 ====================
__global__ void compute_block_sum(float* in, float* block_sum, int vec_len)
{
    //求每个元素的平方数
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ float sdata[];
    
    sdata[threadIdx.x] = (gtid < vec_len) ? in[gtid] * in[gtid] : 0.f;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
 
    if(threadIdx.x == 0)
    {
        block_sum[blockIdx.x] = sdata[0];
    }

}

__global__ void reduce_block_sum(float* block_sum, int num_blocks)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // 处理超过blockDim.x的情况，使用grid-stride loop
    float sum = 0.0f;
    for(int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_sum[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        block_sum[0] = sdata[0];
    }
}

__global__ void normalize_vec(const float* in, float* out, float total_sum, int vec_len)
{
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gtid >= vec_len) return;

    float inv_l2_norm = 1 / sqrt(total_sum);
    out[gtid] = in[gtid] * inv_l2_norm;
}

bool test_vector_l2_normal()
{
    // 测试向量长度，可以根据需要调整
    const int vec_len = 1024 * 1024 + 3; // 故意使用非2的幂次长度以测试边界情况
    const float epsilon = 1e-5f; // 浮点数比较容差
    
    // 1. 生成随机测试数据
    std::vector<float> h_in(vec_len);
    std::srand(std::time(nullptr)); // 随机种子
    for (int i = 0; i < vec_len; ++i)
    {
        // 生成-10.0到10.0之间的随机数
        h_in[i] = static_cast<float>(std::rand()) / RAND_MAX * 20.0f - 10.0f;
    }
    
    // 2. 分配设备内存
    float *d_in, *d_out, *d_block_sum;
    CUDA_CHECK(cudaMalloc(&d_in, vec_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, vec_len * sizeof(float)));
    
    // 计算需要的块数量
    const int block_size = 256;
    const int grid_size = (vec_len + block_size - 1) / block_size;
    
    // 为块求和结果分配内存
    CUDA_CHECK(cudaMalloc(&d_block_sum, grid_size * sizeof(float)));
    
    // 3. 将输入数据复制到设备
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), vec_len * sizeof(float), cudaMemcpyHostToDevice));
    
    // 4. 执行GPU计算
    // 4.1 计算每个块的平方和
    compute_block_sum<<<grid_size, block_size, block_size * sizeof(float)>>>(d_in, d_block_sum, vec_len);
    CHECK_KERNEL_LAUNCH();
    
    // 4.2 归约块求和结果得到总平方和
    reduce_block_sum<<<1, block_size, block_size * sizeof(float)>>>(d_block_sum, grid_size);
    CHECK_KERNEL_LAUNCH();
    
    // 4.3 将总平方和复制回主机
    float total_sum;
    CUDA_CHECK(cudaMemcpy(&total_sum, d_block_sum, sizeof(float), cudaMemcpyDeviceToHost));
    
    // 4.4 执行归一化
    normalize_vec<<<grid_size, block_size>>>(d_in, d_out, total_sum, vec_len);
    CHECK_KERNEL_LAUNCH();
    
    // 5. 将GPU结果复制回主机
    std::vector<float> h_out(vec_len);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, vec_len * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 6. 在CPU上计算参考结果
    float cpu_sum = 0.0f;
    for (int i = 0; i < vec_len; ++i)
    {
        cpu_sum += h_in[i] * h_in[i];
    }
    float cpu_norm = std::sqrt(cpu_sum);
    std::vector<float> h_ref(vec_len);
    for (int i = 0; i < vec_len; ++i)
    {
        h_ref[i] = h_in[i] / cpu_norm;
    }
    
    // 7. 验证结果
    bool success = true;
    for (int i = 0; i < vec_len; ++i)
    {
        if (std::fabs(h_out[i] - h_ref[i]) > epsilon)
        {
            success = false;
            // 打印第一个错误位置和值，方便调试
            std::cout << "验证失败 at index " << i << ": " 
                      << "GPU=" << h_out[i] << ", CPU=" << h_ref[i] 
                      << ", 差值=" << std::fabs(h_out[i] - h_ref[i]) << std::endl;
            break;
        }
    }
    
    // 8. 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "CUDA错误: " << cudaGetErrorString(err) << std::endl;
        success = false;
    }
    
    // 9. 释放内存
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_block_sum));
    
    // 10. 输出测试结果
    if (success)
    {
        std::cout << "向量L2归一化测试通过!" << std::endl;
    }
    else
    {
        std::cout << "向量L2归一化测试失败!" << std::endl;
    }
    
    return success;
}
// ==================== 向量逐元素三角函数运算 ====================

// __global__ void vector_sin(float* in, float* out, int vec_len)
// {
//     int gtid = threadIdx.x + blockDim.x * blockIdx.x;
//     if(gtid >= vec_len) return;
//     out[gtid] = sin(in[gtid]);
// }

__global__ void vector_sin(float* in, float* out, int vec_len)
{
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    int base = 4 * gtid;
    if(base + 3 < vec_len)
    {
        float4 in4 = reinterpret_cast<float4*> (in)[gtid];
        float4 out4;
        out4.x = sinf(in4.x);
        out4.y = sinf(in4.y);
        out4.z = sinf(in4.z);
        out4.w = sinf(in4.w);
        reinterpret_cast<float4*>(out)[gtid] = out4;
    }
    else
    {
        for(int i = 0; i < 4; ++i)
        {
            int gtid = base + i;
            if(gtid < vec_len)
            {
                out[gtid] = sinf(in[gtid]);
            }
        }
    }
}

bool test_vector_sin(int vec_len)
{
    float* in = (float*) malloc(sizeof(float) * vec_len);
    float* h_out = (float*) malloc(sizeof(float) * vec_len);
    float* out = (float*) malloc(sizeof(float) * vec_len);    
    memset(h_out, 0, sizeof(float)* vec_len);
    for(int i = 0; i < vec_len; ++i)
    {
        in[i] = 1.0f;
    }

    float* d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * vec_len));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * vec_len));

    CUDA_CHECK(cudaMemcpy(d_in, in, sizeof(float)*vec_len, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float) * vec_len));

    const int block_size = 256;
    const int grid_size = (vec_len + block_size - 1) / block_size;
    vector_sin <<<grid_size, block_size>>>(d_in, d_out, vec_len);

    CUDA_CHECK(cudaMemcpy(out, d_out, vec_len * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < vec_len; ++i)
    {
        h_out[i] = sin(in[i]);
    }

    bool success = true;
    for(int i = 0; i < vec_len; i++)
    {
        if(fabs(out[i] - h_out[i]) > 0.0005f)
        {
            success = false;
            break;
        }
    }

    // 释放内存
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(in);
    free(h_out);
    free(out);
    
    return success;
}

int main() {
    std::cout << "🚀 CUDA技能快速检验开始...\n\n";
    
    print_gpu_info();
    
    int passed_tests = 0;
    int total_tests = 3;  // 修复：匹配实际测试数量
    
    bool flag = test_vector_mul(1024);
    if(flag)
    {
        std::cout << "test_vector_mul successfully!" << std::endl;
        passed_tests++;  // 修复：正确计数通过的测试
    }
    else
    {
        std::cout << "Failed!" << std::endl;
    }

    flag = test_vector_l2_normal();
    if(flag)
    {
        std::cout << "test_vector_l2_normal successfully!" << std::endl;
        passed_tests++;  // 修复：正确计数通过的测试
    }
    else
    {
        std::cout << "Failed!" << std::endl;
    }

    int vec_len = 1024;
    flag = test_vector_sin(vec_len);
     if(flag)
    {
        std::cout << "test_vec_sin successfully!" << std::endl;
        passed_tests++;  // 修复：正确计数通过的测试
    }
    else
    {
        std::cout << "Failed!" << std::endl;
    }

    
    // 结果总结
    std::cout << "\n" << std::string(40, '=') << "\n";
    std::cout << "检验结果: " << passed_tests << "/" << total_tests << " 通过\n";
    
    float pass_rate = (float)passed_tests / total_tests;
    if (pass_rate >= 0.75f) {
        std::cout << "🎉 恭喜! 基础技能扎实\n";
    } else if (pass_rate >= 0.5f) {
        std::cout << "📈 不错! 继续加油\n";
    } else {
        std::cout << "📚 需要继续学习基础知识\n";
    }
    
    std::cout << "\n💡 建议:\n";
    std::cout << "• 完成更多course练习\n";
    std::cout << "• 学习使用profiling工具 (nvprof/ncu)\n";
    std::cout << "• 实现更复杂的优化算法\n";
    
    return 0;
}
