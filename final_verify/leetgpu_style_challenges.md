# 🏆 LeetGPU风格CUDA挑战题集

基于10节course学习内容，设计的竞技式算子优化挑战，模拟LeetGPU.com的评估方式。

## 📊 挑战评估体系

### 评分标准
- **正确性** (40%): 算法结果必须完全正确
- **性能** (40%): 基于GFLOPS、内存带宽利用率排名
- **代码质量** (20%): 可读性、错误处理、边界情况

### 难度分级
- 🟢 **Easy**: 基础算子实现
- 🟡 **Medium**: 性能优化挑战  
- 🔴 **Hard**: 复杂算法和极致优化
- 🟣 **Expert**: 工业级难题

---

## 🟢 Easy级别挑战

### Challenge 1: Vector Operations Suite
**题目**: 实现高效的向量运算算子库

```cpp
// 要求实现以下所有操作，支持任意长度向量
__global__ void vector_add(const float* a, const float* b, float* c, int n);
__global__ void vector_mul(const float* a, const float* b, float* c, int n); 
__global__ void vector_dot_product(const float* a, const float* b, float* result, int n);
__global__ void vector_norm_l2(const float* a, float* result, int n);
__global__ void vector_scale(const float* a, float scalar, float* b, int n);
```

**性能目标**:
- 向量长度 1M: > 200 GB/s 内存带宽
- 支持非对齐内存访问
- 处理任意大小输入

### Challenge 2: Matrix Transpose Optimizer
**题目**: 实现各种矩阵转置算法并优化到极致

```cpp
// 实现多种转置策略
__global__ void naive_transpose(const float* input, float* output, int rows, int cols);
__global__ void shared_memory_transpose(const float* input, float* output, int rows, int cols);
__global__ void bank_conflict_free_transpose(const float* input, float* output, int rows, int cols);
__global__ void vectorized_transpose(const float* input, float* output, int rows, int cols);
```

**Leaderboard目标**:
- 2048x2048矩阵: > 400 GB/s
- 非方阵优化: 1024x4096 > 350 GB/s
- Bank conflict < 1%

---

## 🟡 Medium级别挑战

### Challenge 3: Reduction Tournament  
**题目**: 实现最快的归约算法

```cpp
// 支持不同的归约操作
template<typename T, typename Op>
__global__ void ultimate_reduce(const T* input, T* output, int n, Op operation);

// 支持的操作: sum, max, min, product, logical_and, logical_or
```

**竞技规则**:
- 输入大小: 1M - 1B elements
- 与CUB库性能对比
- 目标: 达到CUB 90%以上性能

### Challenge 4: GEMM Speed Run
**题目**: 矩阵乘法极速挑战

```cpp
// 实现单精度矩阵乘法，追求极致性能
__global__ void speed_sgemm(const float* A, const float* B, float* C, 
                           int M, int N, int K, float alpha, float beta);
```

**Ranking System**:
- Square Matrix Benchmark: 512, 1024, 2048, 4096
- Rectangular Matrix Challenge: M≠N≠K combinations
- 目标: 达到cuBLAS 80%性能
- Bonus: 支持batched operations

### Challenge 5: Convolution Master
**题目**: 2D卷积算子优化大师赛

```cpp
// 实现高效的2D卷积
__global__ void optimized_conv2d(
    const float* input,    // [N, H, W, C]
    const float* kernel,   // [KH, KW, C, OC] 
    float* output,         // [N, OH, OW, OC]
    ConvParams params
);
```

**测试场景**:
- 典型CNN层: 224x224 input, 3x3/5x5/7x7 kernels
- 不同stride和padding组合
- 与cuDNN性能对比

---

## 🔴 Hard级别挑战

### Challenge 6: Flash Attention Implementation
**题目**: 从零实现Flash Attention

```cpp
// 内存高效的attention计算
__global__ void flash_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
);
```

**技术要求**:
- O(N)内存复杂度 (而非O(N²))
- 数值稳定的在线softmax
- 支持因果mask和attention mask
- 性能目标: 比naive attention快2x以上

### Challenge 7: Sparse Matrix Operations
**题目**: 稀疏矩阵算子优化

```cpp
// CSR格式稀疏矩阵乘法
__global__ void sparse_sgemm_csr(
    const float* values, const int* col_indices, const int* row_ptr,
    const float* dense_matrix, float* result,
    int M, int N, int K, int nnz
);

// 稀疏矩阵向量乘法
__global__ void spmv_csr(
    const float* values, const int* col_indices, const int* row_ptr,
    const float* vector, float* result, int M, int nnz
);
```

**Benchmark数据集**:
- 真实世界稀疏矩阵 (从SuiteSparse获取)
- 不同稀疏度: 0.1%, 1%, 10%
- 与cuSPARSE性能对比

### Challenge 8: Fused Kernel Master
**题目**: 融合算子设计大师

```cpp
// 设计融合的LayerNorm + GELU + Linear
__global__ void fused_layernorm_gelu_linear(
    const float* input,           // [batch, seq_len, hidden_dim]
    const float* ln_weight,       // [hidden_dim]
    const float* ln_bias,         // [hidden_dim] 
    const float* linear_weight,   // [hidden_dim, output_dim]
    const float* linear_bias,     // [output_dim]
    float* output,                // [batch, seq_len, output_dim]
    LayerNormParams ln_params,
    LinearParams linear_params
);
```

**优化目标**:
- 最小化内存访问次数
- 最大化算术强度
- 与分离实现的性能对比

---

## 🟣 Expert级别挑战

### Challenge 9: Auto-Tuning GEMM
**题目**: 自动调优的矩阵乘法库

```cpp
// 实现能自动选择最优配置的GEMM
class AutoTunedGEMM {
public:
    // 自动benchmark并选择最优kernel
    void auto_tune(int M, int N, int K);
    
    // 执行优化后的GEMM
    void execute(const float* A, const float* B, float* C, 
                int M, int N, int K, float alpha, float beta);
                
private:
    // 多种不同的kernel实现
    std::vector<GemmKernel> kernel_variants_;
    
    // 性能数据库
    PerformanceDB perf_db_;
};
```

**技术挑战**:
- 支持20+种不同的优化策略
- 运行时自动选择最优配置
- 构建性能预测模型
- 支持多GPU架构适配

### Challenge 10: Distributed GPU Computing
**题目**: 多GPU协作计算框架

```cpp
// 跨多个GPU的大规模矩阵乘法
class MultiGPUGemm {
public:
    void distributed_gemm(
        const float* A, const float* B, float* C,
        int M, int N, int K, int num_gpus
    );
    
private:
    void partition_work(int M, int N, int K, int num_gpus);
    void execute_distributed();
    void allreduce_results();
};
```

**系统要求**:
- 支持2-8个GPU协作
- 自动负载均衡
- 通信开销最小化
- 扩展性分析报告

---

## 🎯 实战比赛模拟

### Weekly Challenge Format

每周发布新挑战，模拟真实竞赛环境：

```bash
# 比赛模拟脚本
#!/bin/bash

echo "🏁 CUDA算子优化挑战赛 Week $1"
echo "题目: $2"
echo "时间限制: 48小时"
echo "============================================"

# 1. 下载测试数据和baseline
wget https://challenge-data/week$1/testdata.tar.gz
wget https://challenge-data/week$1/baseline_results.json

# 2. 编译测试
nvcc -O3 -arch=native solution.cu -o solution

# 3. 性能测试
./solution < testdata/input1.txt > output1.txt
./benchmark_tool solution

# 4. 正确性验证
python verify_correctness.py output1.txt expected1.txt

# 5. 排行榜提交
curl -X POST https://leaderboard/submit \
     -F "solution=@solution.cu" \
     -F "performance=@benchmark_results.json"
```

### Leaderboard System

```python
# 排行榜计算逻辑
class LeaderboardCalculator:
    def calculate_score(self, submission):
        # 1. 正确性检验 (必须100%正确)
        correctness = verify_correctness(submission.output)
        if not correctness:
            return 0
            
        # 2. 性能得分 (与baseline对比)
        perf_ratio = submission.performance / baseline_performance
        perf_score = min(100, perf_ratio * 100)
        
        # 3. 代码质量得分
        quality_score = analyze_code_quality(submission.code)
        
        # 4. 综合得分
        final_score = (
            correctness * 0.4 +      # 40%
            perf_score * 0.4 +       # 40% 
            quality_score * 0.2      # 20%
        )
        
        return final_score
```

### 学习进度追踪

```cpp
// 个人学习档案
struct LearningProfile {
    int total_challenges_attempted = 0;
    int challenges_solved = 0;
    float average_performance_ratio = 0.0f;
    
    std::map<std::string, float> skill_ratings = {
        {"memory_optimization", 0.0f},
        {"algorithm_design", 0.0f},
        {"parallel_thinking", 0.0f},
        {"debugging_skills", 0.0f}
    };
    
    std::vector<Achievement> unlocked_achievements;
};
```

---

## 📈 与课程内容的对应关系

| Challenge | 对应Course | 核心技能 |
|-----------|------------|----------|
| Vector Ops | Course 1-2 | 基础kernel编写 |
| Transpose | Course 7 | 内存访问优化 |
| Reduction | Course 4 | warp primitives |
| GEMM | Course 5 | 共享内存tiling |
| Convolution | Course 5扩展 | 复杂算法设计 |
| Flash Attention | Course 9 | 内存高效算法 |
| Sparse Ops | 综合应用 | 不规则访问优化 |
| Fused Kernels | Course 8扩展 | 融合优化策略 |
| Auto-tuning | 系统级优化 | 自适应算法 |
| Multi-GPU | 分布式计算 | 系统架构设计 |

---

## 🚀 开始挑战！

```bash
# 创建挑战环境
cd /home/maxiaosong/work_space/cuda_learning/cuda_code/course10
mkdir leetgpu_challenges
cd leetgpu_challenges

# 开始第一个挑战
git clone https://github.com/your-repo/cuda-challenges.git
cd cuda-challenges/challenge-001-vector-ops

# 查看题目
cat README.md
cat test_cases.json

# 开始编码！
cp template.cu solution.cu
# 编辑 solution.cu 实现算法

# 本地测试
make test
make benchmark

# 查看排行榜
make leaderboard
```

**这种竞技式检验方式的优势**:
1. **实战导向**: 真实的性能优化场景
2. **持续激励**: 排行榜和竞争机制
3. **全面评估**: 不仅考虑正确性，更重视性能
4. **社区学习**: 可以学习他人的优化技巧

LeetGPU.com这样的平台确实是检验CUDA学习成果的**绝佳方式**！🏆