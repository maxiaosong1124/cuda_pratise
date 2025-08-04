cmake_minimum_required(VERSION 3.22)
project(CudaReduce LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)


# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(my_reduce_v0_global_memory
  my_reduce_v0_global_memory.cu
)

add_executable(my_reduce_v1_shared_memory
  my_reduce_v1_shared_memory.cu
)

add_executable(my_reduce_v2_no_divergence_branch
  my_reduce_v2_no_divergence_branch.cu
)

add_executable(my_reduce_v3_no_bank_conflict
  my_reduce_v3_no_bank_conflict.cu
)

add_executable(my_reduce_v4_add_during_load_plan_a
  my_reduce_v4_add_during_load_plan_a.cu
)

add_executable(my_reduce_v4_add_during_load_plan_b
  my_reduce_v4_add_during_load_plan_b.cu
)

add_executable(my_reduce_v5_unroll_last_warp
  my_reduce_v5_unroll_last_warp.cu
)

add_executable(my_reduce_v6_completely_unroll
  my_reduce_v6_completely_unroll.cu
)

add_executable(my_reduce_v7_multi_add
  my_reduce_v7_multi_add.cu
)

add_executable(my_reduce_v8_shuffle
  my_reduce_v8_shuffle.cu
)


# Link libraries
target_link_libraries(my_reduce_v0_global_memory
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)
target_link_libraries(my_reduce_v1_shared_memory
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)

target_link_libraries(my_reduce_v2_no_divergence_branch
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)

target_link_libraries(my_reduce_v3_no_bank_conflict
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)

target_link_libraries(my_reduce_v4_add_during_load_plan_a
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)

target_link_libraries(my_reduce_v5_unroll_last_warp
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)

target_link_libraries(my_reduce_v6_completely_unroll
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)

target_link_libraries(my_reduce_v7_multi_add
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)

target_link_libraries(my_reduce_v8_shuffle
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)


# Compile options for all build types

target_compile_options(my_reduce_v0_global_memory
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)

target_compile_options(my_reduce_v1_shared_memory
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)

target_compile_options(my_reduce_v2_no_divergence_branch
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)


target_compile_options(my_reduce_v3_no_bank_conflict
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)

target_compile_options(my_reduce_v4_add_during_load_plan_a
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)


target_compile_options(my_reduce_v5_unroll_last_warp
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)

target_compile_options(my_reduce_v6_completely_unroll
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)

target_compile_options(my_reduce_v7_multi_add
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)

target_compile_options(my_reduce_v8_shuffle
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)


# Set CUDA architecture for all targets

target_compile_options(my_reduce_v0_global_memory
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

target_compile_options(my_reduce_v1_shared_memory
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

target_compile_options(my_reduce_v2_no_divergence_branch
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

target_compile_options(my_reduce_v3_no_bank_conflict
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

target_compile_options(my_reduce_v4_add_during_load_plan_a
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

target_compile_options(my_reduce_v5_unroll_last_warp
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

target_compile_options(my_reduce_v6_completely_unroll
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

target_compile_options(my_reduce_v7_multi_add
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

target_compile_options(my_reduce_v8_shuffle
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)
