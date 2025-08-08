cmake_minimum_required(VERSION 3.22)
project(CudaSgemm LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# 创建可执行文件
add_executable(my_sgemm_v0_global_mem my_sgemm_v0_global_mem.cu)
add_executable(my_sgemm_v1_shared_mem my_sgemm_v1_shared_mem.cu)
add_executable(my_sgemm_v2_shared_mem_sliding_window my_sgemm_v2_shared_mem_sliding_window.cu)
add_executable(my_sgemm_v3_using_float4 my_sgemm_v3_using_float4.cu)
add_executable(my_sgemm_v4_register_cache my_sgemm_v4_register_cache.cu)
# 链接库和设置编译选项
target_link_libraries(my_sgemm_v0_global_mem PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v0_global_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

target_link_libraries(my_sgemm_v1_shared_mem PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v1_shared_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

target_link_libraries(my_sgemm_v2_shared_mem_sliding_window PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v2_shared_mem_sliding_window PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

target_link_libraries(my_sgemm_v3_using_float4 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v3_using_float4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

target_link_libraries(my_sgemm_v4_register_cache PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v4_register_cache PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

