cmake_minimum_required(VERSION 3.10)
project(gamma LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE on)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE RELEASE)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/local/cuda/include)

add_library(link STATIC
    graph/match_gpu.h graph/match_gpu.cu
    kernels/indexing.h kernels/indexing.cu
    kernels/gamma_enumeration.h kernels/gamma_enumeration.cu
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/cuda_helpers.h
    utils/automorphism.h utils/automorphism.cpp)

set_target_properties(link PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(link PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    -Xptxas=\"-v\"
    -DTHRUST_IGNORE_CUB_VERSION_CHECK
    >)

add_executable(gamma
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    utils/config.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main.cpp)

set_target_properties(gamma PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(gamma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    -Xptxas=\"-v\"
    -DTHRUST_IGNORE_CUB_VERSION_CHECK
    >  $<$<COMPILE_LANGUAGE:CXX>:-Wall>)

target_link_libraries(gamma PUBLIC link)
