cmake_minimum_required(VERSION 3.22)
project(CudaSgemm LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# 创建可执行文件
add_executable(my_sgemm_v0_global_mem my_sgemm_v0_global_mem.cu)
add_executable(my_sgemm_v1_shared_mem my_sgemm_v1_shared_mem.cu)
add_executable(my_sgemm_v2_shared_mem_sliding_window my_sgemm_v2_shared_mem_sliding_window.cu)
add_executable(my_sgemm_v3_increase_work_of_per_thread my_sgemm_v3_increase_work_of_per_thread.cu)
# 链接库和设置编译选项
target_link_libraries(my_sgemm_v0_global_mem PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v0_global_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

target_link_libraries(my_sgemm_v1_shared_mem PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v1_shared_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

target_link_libraries(my_sgemm_v2_shared_mem_sliding_window PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v2_shared_mem_sliding_window PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

target_link_libraries(my_sgemm_v3_increase_work_of_per_thread PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(my_sgemm_v3_increase_work_of_per_thread PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g -arch=sm_86>)

