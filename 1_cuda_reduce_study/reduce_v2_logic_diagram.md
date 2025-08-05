# CUDA Reduce V2 计算逻辑图

## 核心算法：无分支分歧的归约操作

这个版本的reduce kernel通过避免分支分歧来提高性能。以下是详细的计算逻辑图：

### 假设参数
- `THREAD_PER_BLOCK = 8` (为了简化图示，实际代码中是256)
- 一个block处理8个元素：[a0, a1, a2, a3, a4, a5, a6, a7]

### 步骤详解

#### 初始化阶段
```
线程ID:   0   1   2   3   4   5   6   7
原始数据: a0  a1  a2  a3  a4  a5  a6  a7
        ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
shared: [a0][a1][a2][a3][a4][a5][a6][a7]
```

#### 迭代1: i = 1 (步长为1)
```
活跃线程: threadIdx.x < blockDim.x / (1 * 2) = 4
即线程 0, 1, 2, 3 参与计算

线程0: index = 0 * 2 * 1 = 0, shared[0] += shared[1]
线程1: index = 1 * 2 * 1 = 2, shared[2] += shared[3]  
线程2: index = 2 * 2 * 1 = 4, shared[4] += shared[5]
线程3: index = 3 * 2 * 1 = 6, shared[6] += shared[7]

计算后:
shared: [a0+a1][a1][a2+a3][a3][a4+a5][a5][a6+a7][a7]
        ↑           ↑           ↑           ↑
        更新        更新        更新        更新
```

#### 迭代2: i = 2 (步长为2)
```
活跃线程: threadIdx.x < blockDim.x / (2 * 2) = 2
即线程 0, 1 参与计算

线程0: index = 0 * 2 * 2 = 0, shared[0] += shared[2]
线程1: index = 1 * 2 * 2 = 4, shared[4] += shared[6]

计算后:
shared: [a0+a1+a2+a3][a1][a2+a3][a3][a4+a5+a6+a7][a5][a6+a7][a7]
        ↑                                 ↑
        更新                              更新
```

#### 迭代3: i = 4 (步长为4)
```
活跃线程: threadIdx.x < blockDim.x / (4 * 2) = 1
即只有线程 0 参与计算

线程0: index = 0 * 2 * 4 = 0, shared[0] += shared[4]

计算后:
shared: [a0+a1+a2+a3+a4+a5+a6+a7][a1][a2+a3][a3][a4+a5+a6+a7][a5][a6+a7][a7]
        ↑
        最终结果
```

#### 最终输出
```
线程0将shared[0]的结果写入d_output[blockIdx.x]
```

### 关键特性

1. **无分支分歧**: 
   - 条件判断 `threadIdx.x < blockDim.x / (i * 2)` 确保连续的线程ID参与计算
   - 避免了像 `threadIdx.x % (i * 2) == 0` 这样会造成分支分歧的条件

2. **内存访问模式**:
   - 使用shared memory提高访问速度
   - 访问模式是连续的，减少bank conflict

3. **线程利用率**:
   - 每次迭代减少一半的活跃线程
   - 迭代1: 4个线程活跃
   - 迭代2: 2个线程活跃  
   - 迭代3: 1个线程活跃

### 视觉化表示

```
迭代次数    活跃线程数    计算模式
   1           4        [0+1] [2+3] [4+5] [6+7]
   2           2        [01+23] [45+67]
   3           1        [0123+4567]
```

### 算法复杂度
- 时间复杂度: O(log n) 其中n是THREAD_PER_BLOCK
- 空间复杂度: O(n) 用于shared memory
- 同步次数: log₂(THREAD_PER_BLOCK) 次 `__syncthreads()`

这种设计相比于有分支分歧的版本，具有更好的warp利用率和更少的线程空闲时间。
