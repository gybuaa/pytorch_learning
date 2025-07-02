cmake_minimum_required(VERSION 3.18)
project(voxelization_cpu2 LANGUAGES CXX CUDA)

# 1. 查找依赖包
find_package(Torch REQUIRED COMPONENTS CXX CUDA)
find_package(Python3 REQUIRED COMPONENTS Development)  # 修改为Python3

# 2. 添加库和可执行文件
# 先编译库（如果有）
# add_library(voxelization_cpu2 SHARED src/voxelization_cpu2.cpp)
# target_link_libraries(voxelization_cpu2 PRIVATE "${TORCH_LIBRARIES}")

# 直接编译可执行文件
add_executable(voxelization_cpu2 src/voxelization_cpu2.cpp)

# 3. 链接库
target_link_libraries(voxelization_cpu2 PRIVATE 
    "${TORCH_LIBRARIES}"
    Python3::Python  # 修改为Python3
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 3. 设置编译标准
set_target_properties(voxelization_cpu2 PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)

# 4. 设置CUDA架构（根据你的GPU调整）
set(CUDA_ARCH CACHE STRING "CUDA Architecture")
if(CUDA_ARCH)
    target_compile_options(voxelization_cpu2 PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}>
    )
endif()