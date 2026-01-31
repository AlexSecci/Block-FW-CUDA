cmake_minimum_required(VERSION 4.2)
project(FloydWarshallCUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

include_directories(include)

find_package(CUDAToolkit REQUIRED)


file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")

add_executable(FloydWarshall ${SOURCES})
target_link_libraries(FloydWarshall PRIVATE CUDA::cudart)

add_executable(DataGenerator tools/data_generator.cpp)


# Hardware optimization for RTX 3070 Ti (Ampere, Compute Capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Optimization Flags
# -O3: Maximum optimization (heavy inlining, unrolling, etc.)
# -lineinfo: Generate line-number information for device code (useful for Nsight Compute profiling)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
endif()
