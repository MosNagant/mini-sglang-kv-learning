cmake_minimum_required(VERSION 3.24)
project(cuda_starter LANGUAGES CXX CUDA)

add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES 89
)

enable_testing()
add_test(NAME gpu_vector_add COMMAND vector_add)
set_tests_properties(gpu_vector_add PROPERTIES TIMEOUT 60)
