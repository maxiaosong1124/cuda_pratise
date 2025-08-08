# SGEMM V2 - 共享内存滑动窗口版本 计算逻辑图

## 核心算法

**目标**：使用滑动窗口技术优化共享内存使用，支持任意大小的K维度

## 假设参数
- 矩阵A: 4×8, 矩阵B: 8×4, 矩阵C: 4×4  
- BLOCK_SIZE = 2 (为了简化图示)
- K = 8 (需要4个窗口: 8/2 = 4)

## 滑动窗口机制详解

### 共享内存布局 (固定大小)
```
A_shared[BLOCK_SIZE][BLOCK_SIZE] = A_shared[2][2]
B_shared[BLOCK_SIZE][BLOCK_SIZE] = B_shared[2][2]

每个窗口只能存储 2×2 的数据块
```

### 窗口划分策略
```
K维度划分 (K=8, BLOCK_SIZE=2):
窗口0: s=0, 处理 k ∈ [0,1]
窗口1: s=2, 处理 k ∈ [2,3]  
窗口2: s=4, 处理 k ∈ [4,5]
窗口3: s=6, 处理 k ∈ [6,7]
```

## 详细计算过程

### 以Block(0,0)计算C[0:2][0:2]为例

#### 窗口0 (s=0): 处理k∈[0,1]

##### 数据加载阶段
```
A矩阵加载 (A[block_row][s:s+BLOCK_SIZE]):
T(0,0): A_shared[0][0] = A[0][0]
T(0,1): A_shared[0][1] = A[0][1]  
T(1,0): A_shared[1][0] = A[1][0]
T(1,1): A_shared[1][1] = A[1][1]

B矩阵加载 (B[s:s+BLOCK_SIZE][block_col]):
T(0,0): B_shared[0][0] = B[0][0]
T(0,1): B_shared[0][1] = B[0][1]
T(1,0): B_shared[1][0] = B[1][0]  
T(1,1): B_shared[1][1] = B[1][1]

共享内存状态:
A_shared:     B_shared:
┌─────┬─────┐ ┌─────┬─────┐
│A[0,0]│A[0,1]│ │B[0,0]│B[0,1]│
├─────┼─────┤ ├─────┼─────┤
│A[1,0]│A[1,1]│ │B[1,0]│B[1,1]│
└─────┴─────┘ └─────┴─────┘
```

##### 计算阶段
```
__syncthreads(); // 等待数据加载完成

每个线程计算当前窗口的贡献:
T(0,0) → temp += A_shared[0][0]*B_shared[0][0] + A_shared[0][1]*B_shared[1][0]
T(0,1) → temp += A_shared[0][0]*B_shared[0][1] + A_shared[0][1]*B_shared[1][1]
T(1,0) → temp += A_shared[1][0]*B_shared[0][0] + A_shared[1][1]*B_shared[1][0]
T(1,1) → temp += A_shared[1][0]*B_shared[0][1] + A_shared[1][1]*B_shared[1][1]

__syncthreads(); // 为下一个窗口做准备
```

#### 窗口1 (s=2): 处理k∈[2,3]

##### 数据加载阶段  
```
A矩阵加载 (A[block_row][2:4]):
T(0,0): A_shared[0][0] = A[0][2]  // 复用A_shared空间
T(0,1): A_shared[0][1] = A[0][3]
T(1,0): A_shared[1][0] = A[1][2]
T(1,1): A_shared[1][1] = A[1][3]

B矩阵加载 (B[2:4][block_col]):
T(0,0): B_shared[0][0] = B[2][0]  // 复用B_shared空间
T(0,1): B_shared[0][1] = B[2][1] 
T(1,0): B_shared[1][0] = B[3][0]
T(1,1): B_shared[1][1] = B[3][1]

共享内存状态:
A_shared:     B_shared:
┌─────┬─────┐ ┌─────┬─────┐
│A[0,2]│A[0,3]│ │B[2,0]│B[2,1]│
├─────┼─────┤ ├─────┼─────┤  
│A[1,2]│A[1,3]│ │B[3,0]│B[3,1]│
└─────┴─────┘ └─────┴─────┘
```

##### 计算阶段
```
__syncthreads();

累加到之前的temp结果:
T(0,0) → temp += A_shared[0][0]*B_shared[0][0] + A_shared[0][1]*B_shared[1][0]
T(0,1) → temp += A_shared[0][0]*B_shared[0][1] + A_shared[0][1]*B_shared[1][1]
T(1,0) → temp += A_shared[1][0]*B_shared[0][0] + A_shared[1][1]*B_shared[1][0]
T(1,1) → temp += A_shared[1][0]*B_shared[0][1] + A_shared[1][1]*B_shared[1][1]

__syncthreads();
```

#### 窗口2和窗口3
```
重复相同的加载-计算过程，直到处理完所有K维度
最终每个线程的temp累积了所有窗口的贡献
```

## 滑动窗口的核心逻辑

### 窗口迭代循环
```
for(int s = 0; s < K; s += BLOCK_SIZE) {
    // 步骤1: 加载当前窗口数据
    load_window_data(s);
    __syncthreads();
    
    // 步骤2: 计算当前窗口贡献  
    compute_window_contribution();
    __syncthreads();
}
```

### 数据加载映射
```
窗口s时的加载模式:

A矩阵加载:
A_shared[threadIdx.y][threadIdx.x] = A[(blockIdx.y*BLOCK_SIZE + threadIdx.y) * K + (s + threadIdx.x)]

B矩阵加载:
B_shared[threadIdx.y][threadIdx.x] = B[(s + threadIdx.y) * N + (blockIdx.x*BLOCK_SIZE + threadIdx.x)]
```

### 累加计算
```
每个窗口内的计算:
for(int k = 0; k < BLOCK_SIZE; ++k) {
    temp += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
}
```

## 算法特点分析

### V2相比V1的关键改进

#### 内存使用对比
```
V1版本:
- 共享内存大小: BLOCK_SIZE × K
- K值限制: 受共享内存容量约束  
- 适用场景: K值较小的矩阵

V2版本:
- 共享内存大小: 2 × BLOCK_SIZE × BLOCK_SIZE (固定)
- K值限制: 无限制，通过滑动窗口处理
- 适用场景: 任意大小的矩阵
```

#### 执行流程对比
```
V1: 一次性加载 → 计算 → 完成
V2: 多轮次: 加载窗口1 → 计算 → 加载窗口2 → 计算 → ... → 完成
```

### 性能特点

#### 优点
- **内存效率高**: 固定大小的共享内存，支持任意K值
- **扩展性强**: 不受K维度大小限制
- **数据局部性好**: 每个窗口内数据复用率高
- **内存合并访问**: 连续内存访问模式

#### 缺点
- **同步开销增加**: 每个窗口需要两次线程同步
- **循环开销**: 增加了外层循环的控制开销
- **寄存器压力**: 需要保持累加器状态

### 内存访问分析

```
滑动窗口的内存访问统计:
┌────────────────┬──────────────┬─────────────────┐
│   访问类型     │   访问次数   │      特点       │
├────────────────┼──────────────┼─────────────────┤
│ 全局内存读取   │   M×K+K×N    │ 每个元素读1次   │
│ 共享内存读取   │  2×M×N×K     │ 低延迟访问      │
│ 线程同步次数   │2×⌈K/BLOCK⌉  │ 窗口切换开销    │
│ 寄存器使用     │    M×N       │ 累加器temp      │
└────────────────┴──────────────┴─────────────────┘

窗口数量: ⌈K / BLOCK_SIZE⌉
每窗口开销: 2次同步 + BLOCK_SIZE²次计算
```

## 代码核心部分

```cuda
template <unsigned int BLOCK_SIZE>
__global__ void gpu_sgemm(float* A, float* B, float* C, 
                          const int M, const int N, const int K)
{
    // 固定大小的共享内存
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;  // 累加器初始化
    
    // 滑动窗口循环
    for(int s = 0; s < K; s += BLOCK_SIZE) {
        
        // 加载A矩阵窗口
        if ((blockIdx.y * BLOCK_SIZE + threadIdx.y) < M && 
            (s + threadIdx.x) < K) {
            A_shared[threadIdx.y][threadIdx.x] = 
                A[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * K + (s + threadIdx.x)];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // 加载B矩阵窗口  
        if ((s + threadIdx.y) < K && 
            (blockIdx.x * BLOCK_SIZE + threadIdx.x) < N) {
            B_shared[threadIdx.y][threadIdx.x] = 
                B[(s + threadIdx.y) * N + (blockIdx.x * BLOCK_SIZE + threadIdx.x)];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();  // 等待所有线程完成加载
        
        // 计算当前窗口的贡献
        for(int k = 0; k < BLOCK_SIZE; ++k) {
            temp += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }
        
        __syncthreads();  // 为下一次加载做准备
    }
    
    // 写入最终结果
    if (m < M && n < N) {
        C[m * N + n] = temp;
    }
}
```

## 算法复杂度

```
时间复杂度: O(M×N×K / P), P为并行线程数
空间复杂度: O(BLOCK_SIZE²) - 固定共享内存
同步次数: 2×⌈K/BLOCK_SIZE⌉ 
全局内存访问: O(M×K + K×N + M×N)
共享内存访问: O(M×N×K)
循环迭代次数: ⌈K/BLOCK_SIZE⌉
```

## 适用场景

1. **大规模矩阵**: K值很大，超出V1版本的内存限制
2. **通用计算**: 不需要事先知道K的具体大小
3. **内存受限环境**: 共享内存容量有限的场景
4. **生产环境**: 需要处理各种尺寸矩阵的应用

## 性能分析

### 理论性能
```
相比V0版本:
- 全局内存访问减少: 约K倍
- 内存延迟降低: 20倍 (共享内存 vs 全局内存)

相比V1版本:
- 通用性提升: 支持任意K值
- 同步开销增加: 与窗口数量成正比
- 实际性能: 略低于V1 (当K较小时)
```

### 优化效果
```
最适合的场景: K >> BLOCK_SIZE
性能提升幅度: 2-4倍 (相比V0)
内存带宽利用率: 60-80%
```

## 进一步优化方向

```
V2 → V3: 向量化优化
- Float4加载指令
- 提升内存带宽利用率
- 减少内存事务数量

其他可能的优化:
- 双缓冲技术: 重叠计算和数据加载
- Warp级优化: 利用warp内线程的协作
- 寄存器优化: 减少共享内存访问
```

## 总结

V2版本实现了滑动窗口优化，解决了V1版本的核心限制：

- **通用性大幅提升**: 支持任意大小的矩阵乘法
- **内存使用效率**: 固定的共享内存占用
- **算法鲁棒性**: 不依赖于特定的矩阵尺寸
- **实用价值高**: 可用于生产环境的通用实现

虽然引入了额外的同步开销，但换来了算法的通用性和扩展性，是GPU矩阵乘法优化中的重要里程碑。
