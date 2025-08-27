/*
 * CUDA学习成果检验 - 综合测试项目模板
 * 
 * 本文件提供了一个完整的测试框架，用于验证CUDA学习成果
 * 涵盖从基础kernel编写到高级优化技术的全方位评估
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <random>
#include <iomanip>

// ==================== 错误检查宏 ====================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ==================== 性能测试工具 ====================
class GPUTimer {
private:
    cudaEvent_t start_, stop_;
    
public:
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~GPUTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_, stop_));
        return elapsed_time;
    }
};

// ==================== Level 1: 基础技能测试 ====================

// TODO: 实现向量点积kernel
__global__ void dot_product_kernel(const float* a, const float* b, float* result, int n) {
    // 学员实现：使用共享内存和warp reduction优化
    // 提示：考虑内存合并访问和线程束分歧问题
}

// TODO: 实现向量归一化kernel  
__global__ void normalize_kernel(const float* input, float* output, int n) {
    // 学员实现：两阶段算法 - 先计算norm，再归一化
    // 难点：需要处理数值稳定性问题
}

// ==================== Level 2: 进阶优化技能 ====================

// TODO: 实现warp-level reduction
__device__ float warp_reduce_sum(float val) {
    // 学员实现：使用shuffle指令实现高效warp内归约
    // 要求：支持任意warp size，处理边界情况
}

// TODO: 实现tiled矩阵乘法
template<int TILE_SIZE>
__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, 
                                   int M, int N, int K) {
    // 学员实现：基于共享内存的矩阵乘法优化
    // 目标性能：达到理论峰值性能的60%以上
}

// ==================== Level 3: 高级算法实现 ====================

// TODO: 实现并行直方图统计 (避免race condition)
__global__ void histogram_kernel(const int* input, int* histogram, int n, int num_bins) {
    // 学员实现：使用原子操作或每线程私有histogram后合并
    // 考虑：负载均衡和内存冲突最小化
}

// TODO: 实现LayerNorm
__global__ void layer_norm_kernel(const float* input, float* output, 
                                 const float* gamma, const float* beta,
                                 int batch_size, int feature_dim, float eps) {
    // 学员实现：包含数值稳定性的LayerNorm
    // 挑战：处理不同的feature_dim大小，优化内存访问模式
}

// ==================== Level 4: 专家级挑战 ====================

// TODO: 实现简化版Flash Attention
__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V,
                                      float* output, int seq_len, int head_dim, 
                                      int num_heads) {
    // 学员实现：内存高效的attention计算
    // 要求：O(seq_len)内存复杂度，而非O(seq_len^2)
}

// ==================== 测试框架和评估系统 ====================

class CUDAAssessment {
private:
    std::vector<std::string> test_results_;
    int passed_tests_ = 0;
    int total_tests_ = 0;
    
public:
    // 测试结果记录
    void record_test(const std::string& test_name, bool passed, 
                    float performance_score = -1.0f) {
        total_tests_++;
        if (passed) {
            passed_tests_++;
            std::string result = "✓ " + test_name;
            if (performance_score >= 0) {
                result += " (Score: " + std::to_string(performance_score) + ")";
            }
            test_results_.push_back(result);
        } else {
            test_results_.push_back("✗ " + test_name + " FAILED");
        }
    }
    
    // Level 1测试套件
    void run_level1_tests() {
        std::cout << "\n=== Level 1: 基础技能测试 ===\n";
        
        // 测试1：向量点积正确性和性能
        test_dot_product();
        
        // 测试2：内存管理效率
        test_memory_management();
        
        // 测试3：基础kernel参数调优
        test_kernel_configuration();
    }
    
    // Level 2测试套件  
    void run_level2_tests() {
        std::cout << "\n=== Level 2: 进阶优化技能 ===\n";
        
        // 测试1：warp-level优化
        test_warp_primitives();
        
        // 测试2：共享内存优化
        test_shared_memory_optimization();
        
        // 测试3：矩阵乘法性能挑战
        test_matmul_performance();
    }
    
    // Level 3测试套件
    void run_level3_tests() {
        std::cout << "\n=== Level 3: 高级算法实现 ===\n";
        
        // 测试1：并发控制
        test_concurrency_control();
        
        // 测试2：数值稳定性
        test_numerical_stability();
        
        // 测试3：复杂算法实现
        test_advanced_algorithms();
    }
    
    // Level 4测试套件
    void run_level4_tests() {
        std::cout << "\n=== Level 4: 专家级挑战 ===\n";
        
        // 测试1：内存优化算法
        test_memory_efficient_algorithms();
        
        // 测试2：多kernel协作
        test_multi_kernel_coordination();
        
        // 测试3：系统级优化
        test_system_optimization();
    }
    
    // 综合性能评估
    void comprehensive_benchmark() {
        std::cout << "\n=== 综合性能基准测试 ===\n";
        
        // GPU信息获取
        print_gpu_info();
        
        // 内存带宽测试
        benchmark_memory_bandwidth();
        
        // 计算性能测试
        benchmark_compute_performance();
        
        // 能效比测试
        benchmark_power_efficiency();
    }
    
    // 最终成绩报告
    void generate_report() {
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "       CUDA学习成果评估报告\n";
        std::cout << std::string(50, '=') << "\n";
        
        // 测试结果汇总
        std::cout << "测试通过率: " << passed_tests_ << "/" << total_tests_ 
                  << " (" << std::fixed << std::setprecision(1) 
                  << (100.0f * passed_tests_ / total_tests_) << "%)\n\n";
        
        // 详细结果
        std::cout << "详细测试结果:\n";
        for (const auto& result : test_results_) {
            std::cout << "  " << result << "\n";
        }
        
        // 技能等级判定
        float pass_rate = (float)passed_tests_ / total_tests_;
        std::cout << "\n技能等级评定:\n";
        if (pass_rate >= 0.9f) {
            std::cout << "🎉 专家级 (Expert) - 优秀的CUDA开发能力\n";
        } else if (pass_rate >= 0.7f) {
            std::cout << "⭐ 高级 (Advanced) - 良好的优化技能\n"; 
        } else if (pass_rate >= 0.5f) {
            std::cout << "📈 中级 (Intermediate) - 基础技能扎实\n";
        } else {
            std::cout << "📚 初级 (Beginner) - 需要继续学习\n";
        }
        
        // 改进建议
        provide_improvement_suggestions(pass_rate);
    }
    
private:
    // 具体测试实现方法...
    void test_dot_product() {
        // TODO: 实现点积测试逻辑
    }
    
    void test_memory_management() {
        // TODO: 测试内存分配效率和正确性
    }
    
    void test_kernel_configuration() {
        // TODO: 测试不同配置的性能差异
    }
    
    void test_warp_primitives() {
        // TODO: 测试warp shuffle等原语使用
    }
    
    void test_shared_memory_optimization() {
        // TODO: 测试共享内存bank冲突优化
    }
    
    void test_matmul_performance() {
        // TODO: 矩阵乘法性能基准测试
    }
    
    // ... 其他测试方法实现
    
    void print_gpu_info() {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        std::cout << "GPU设备信息:\n";
        std::cout << "  设备名称: " << prop.name << "\n";
        std::cout << "  计算能力: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  全局内存: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
        std::cout << "  SM数量: " << prop.multiProcessorCount << "\n";
        std::cout << "  最大线程/块: " << prop.maxThreadsPerBlock << "\n\n";
    }
    
    void benchmark_memory_bandwidth() {
        // TODO: 实现内存带宽基准测试
    }
    
    void benchmark_compute_performance() {
        // TODO: 实现计算性能基准测试
    }
    
    void benchmark_power_efficiency() {
        // TODO: 如果支持，测试功耗效率
    }
    
    void provide_improvement_suggestions(float pass_rate) {
        std::cout << "\n📝 学习建议:\n";
        if (pass_rate < 0.5f) {
            std::cout << "  • 回顾基础概念：线程层次、内存模型\n";
            std::cout << "  • 练习简单kernel编写和调试\n";
            std::cout << "  • 学习CUDA错误处理最佳实践\n";
        } else if (pass_rate < 0.7f) {
            std::cout << "  • 深入学习性能优化技术\n";
            std::cout << "  • 掌握profiling工具使用 (nvprof, ncu)\n";
            std::cout << "  • 研究GPU架构特性\n";
        } else if (pass_rate < 0.9f) {
            std::cout << "  • 学习高级算法优化模式\n";
            std::cout << "  • 参与开源GPU项目贡献\n";
            std::cout << "  • 探索多GPU和分布式计算\n";
        } else {
            std::cout << "  • 考虑深入系统级优化\n";
            std::cout << "  • 研究最新GPU架构特性\n";
            std::cout << "  • 分享经验，指导他人学习\n";
        }
        
        std::cout << "\n🔗 推荐资源:\n";
        std::cout << "  • NVIDIA官方文档和样例\n";
        std::cout << "  • GPU架构白皮书\n";
        std::cout << "  • 开源高性能计算项目\n";
    }
};

// ==================== 主函数 - 运行完整评估 ====================
int main() {
    std::cout << "🚀 CUDA学习成果全面评估开始...\n";
    
    CUDAAssessment assessment;
    
    // 运行各级别测试
    assessment.run_level1_tests();
    assessment.run_level2_tests(); 
    assessment.run_level3_tests();
    assessment.run_level4_tests();
    
    // 综合性能基准测试
    assessment.comprehensive_benchmark();
    
    // 生成最终报告
    assessment.generate_report();
    
    return 0;
}