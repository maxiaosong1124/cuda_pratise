cmake_minimum_required(VERSION 3.22)
project(SGEMM LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(sgemm_v1_naive
  sgemm_v1_naive.cu
)
# Link libraries
target_link_libraries(sgemm_v1_naive
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)
target_compile_options(sgemm_v1_naive
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)
# Set CUDA architecture flag explicitly (match reduce directory style)
target_compile_options(sgemm_v1_naive
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)


# Compile options (match reduce directory style)

add_executable(sgemm_v2_mem_coaliescing
  sgemm_v2_mem_coaliescing.cu
)
target_link_libraries(sgemm_v2_mem_coaliescing
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)
target_compile_options(sgemm_v2_mem_coaliescing
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)
# Set CUDA architecture flag explicitly (match reduce directory style)
target_compile_options(sgemm_v2_mem_coaliescing
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

add_executable(sgemm_v3_shared_mem
  sgemm_v3_shared_mem.cu
)
target_link_libraries(sgemm_v3_shared_mem
  PRIVATE
    CUDA::cudart ${CUDA_cublas_LIBRARY}
)
target_compile_options(sgemm_v3_shared_mem
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
)
# Set CUDA architecture flag explicitly (match reduce directory style)
target_compile_options(sgemm_v3_shared_mem
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: -arch=sm_86>
)

