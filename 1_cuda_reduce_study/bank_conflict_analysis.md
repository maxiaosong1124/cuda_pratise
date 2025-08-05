# CUDA Shared Memory Bank Conflict 详解

## Shared Memory Bank 结构

CUDA的shared memory被组织成32个bank，每个bank宽度为4字节（32位）。对于连续的内存地址：

```
地址:    0    4    8   12   16   20   24   28   32   36   40   44  ...
Bank:    0    1    2    3    4    5    6    7    0    1    2    3  ...
```

**关键规律**: `bank_id = (address / 4) % 32`

对于float类型数据（4字节）：
- `shared[0]` 在 Bank 0
- `shared[1]` 在 Bank 1  
- `shared[2]` 在 Bank 2
- ...
- `shared[31]` 在 Bank 31
- `shared[32]` 在 Bank 0（循环）

## Bank Conflict 发生条件

当一个warp（32个线程）中的多个线程**同时访问同一个bank的不同地址**时，就会发生bank conflict，导致访问序列化。

### 无冲突情况：
1. 所有线程访问不同bank
2. 所有线程访问同一bank的同一地址（广播）

### 有冲突情况：
多个线程访问同一bank的不同地址

## V2 vs V3 的Bank Conflict分析

### V2的访问模式（可能有Bank Conflict）

假设THREAD_PER_BLOCK = 32，观察前32个线程的访问模式：

#### V2的迭代过程：
```c
for(int i = 1; i < blockDim.x; i *= 2) {
    if(threadIdx.x < blockDim.x / (i * 2)) {
        int index = threadIdx.x * 2 * i;
        shared[index] += shared[index + i];
    }
}
```

**迭代1 (i=1)**: 线程0-15活跃
```
线程0: shared[0] += shared[1]   // Bank 0, Bank 1
线程1: shared[2] += shared[3]   // Bank 2, Bank 3  
线程2: shared[4] += shared[5]   // Bank 4, Bank 5
...
线程15: shared[30] += shared[31] // Bank 30, Bank 31
```
✅ **无冲突**: 每个线程访问不同的bank

**迭代2 (i=2)**: 线程0-7活跃
```
线程0: shared[0] += shared[2]   // Bank 0, Bank 2
线程1: shared[4] += shared[6]   // Bank 4, Bank 6
线程2: shared[8] += shared[10]  // Bank 8, Bank 10
...
线程7: shared[28] += shared[30] // Bank 28, Bank 30
```
✅ **无冲突**: 每个线程访问不同的bank

**迭代3 (i=4)**: 线程0-3活跃
```
线程0: shared[0] += shared[4]   // Bank 0, Bank 4
线程1: shared[8] += shared[12]  // Bank 8, Bank 12
线程2: shared[16] += shared[20] // Bank 16, Bank 20
线程3: shared[24] += shared[28] // Bank 24, Bank 28
```
✅ **无冲突**: 每个线程访问不同的bank

**但是！** 当THREAD_PER_BLOCK > 32时，V2可能出现问题：

假设THREAD_PER_BLOCK = 64，迭代4 (i=8)，线程0-3活跃：
```
线程0: shared[0] += shared[8]    // Bank 0, Bank 8
线程1: shared[16] += shared[24]  // Bank 16, Bank 24  
线程2: shared[32] += shared[40]  // Bank 0, Bank 8  ❌
线程3: shared[48] += shared[56]  // Bank 16, Bank 24 ❌
```
🚫 **有冲突**: 线程0和线程2都访问Bank 0和Bank 8！

### V3的访问模式（无Bank Conflict）

#### V3的迭代过程：
```c
for(int i = blockDim.x / 2; i > 0; i /= 2) {
    if(threadIdx.x < i) {
        shared[threadIdx.x] += shared[threadIdx.x + i];
    }
}
```

**迭代1 (i=16)**: 线程0-15活跃
```
线程0: shared[0] += shared[16]   // Bank 0, Bank 16
线程1: shared[1] += shared[17]   // Bank 1, Bank 17
线程2: shared[2] += shared[18]   // Bank 2, Bank 18
...
线程15: shared[15] += shared[31] // Bank 15, Bank 31
```
✅ **无冲突**: 每个线程访问不同的bank

**迭代2 (i=8)**: 线程0-7活跃
```
线程0: shared[0] += shared[8]    // Bank 0, Bank 8
线程1: shared[1] += shared[9]    // Bank 1, Bank 9
线程2: shared[2] += shared[10]   // Bank 2, Bank 10
...
线程7: shared[7] += shared[15]   // Bank 7, Bank 15
```
✅ **无冲突**: 每个线程访问不同的bank

**关键优势**: V3中，活跃线程的threadIdx.x总是连续的：0, 1, 2, 3...
这保证了：
- `shared[threadIdx.x]` 访问连续的bank
- `shared[threadIdx.x + i]` 也访问连续但不重叠的bank

## 为什么V3更好？

1. **简单的访问模式**: `threadIdx.x` 直接对应bank编号
2. **连续线程活跃**: 活跃线程ID总是从0开始连续
3. **步长设计**: 步长i保证了两个访问地址映射到不同bank
4. **可预测性**: 容易分析和验证无冲突

## 总结

V3通过以下设计避免bank conflict：
- 使用连续的线程ID（0到i-1）
- 访问模式简单：`shared[threadIdx.x]` 和 `shared[threadIdx.x + i]`
- 步长i的设计确保两个地址不会映射到同一bank

这种设计不仅避免了bank conflict，还让代码更简洁、更容易理解和优化。
