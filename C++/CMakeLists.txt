cmake_minimum_required(VERSION 3.14)
project(face_recognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 LibTorch 和 OpenCV 路径
set(LIBTORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc16/lib")

# 查找 LibTorch
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
find_package(Torch REQUIRED)

# 设置 Torch 的库目录，以便链接所有需要的库
set(TORCH_LIBRARIES 
    ${LIBTORCH_PATH}/lib/torch_cpu.lib
    ${LIBTORCH_PATH}/lib/torch.lib
    ${LIBTORCH_PATH}/lib/c10.lib
    ${LIBTORCH_PATH}/lib/fmt.lib
    ${LIBTORCH_PATH}/lib/kineto.lib
    ${LIBTORCH_PATH}/lib/libprotobuf.lib
    ${LIBTORCH_PATH}/lib/dnnl.lib
    ${LIBTORCH_PATH}/lib/XNNPACK.lib
    ${LIBTORCH_PATH}/lib/cpuinfo.lib
    ${LIBTORCH_PATH}/lib/pthreadpool.lib
    ${LIBTORCH_PATH}/lib/sleef.lib
    ${LIBTORCH_PATH}/lib/microkernels-prod.lib
    ${LIBTORCH_PATH}/lib/libittnotify.lib
)

# 设置 OpenCV 路径
set(OpenCV_STATIC_LIBS OFF)
find_package(OpenCV REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build" NO_DEFAULT_PATH)

# 添加 ViT 可执行文件
add_executable(test_vit test_vit.cpp)

# 包含目录
target_include_directories(test_vit PRIVATE 
    ${LIBTORCH_PATH}/include
    ${LIBTORCH_PATH}/include/torch/csrc/api/include
    ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(test_vit 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# Windows 特定设置
if(MSVC)
    target_compile_options(test_vit PRIVATE /W4 /O2)
    target_link_directories(test_vit PRIVATE ${LIBTORCH_PATH}/lib)
else()
    target_compile_options(test_vit PRIVATE -Wall -O2)
endif()

# 设置输出目录
set_target_properties(test_vit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 添加 Facenet Showcase 可执行文件
add_executable(showcase showcase.cpp)

# 包含目录
target_include_directories(showcase PRIVATE 
    ${LIBTORCH_PATH}/include
    ${LIBTORCH_PATH}/include/torch/csrc/api/include
    ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(showcase 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# Windows 特定设置
if(MSVC)
    target_compile_options(showcase PRIVATE /W4 /O2)
    target_link_directories(showcase PRIVATE ${LIBTORCH_PATH}/lib)
else()
    target_compile_options(showcase PRIVATE -Wall -O2)
endif()

# 设置输出目录
set_target_properties(showcase PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 复制 DLL 到输出目录 (Windows)
if(WIN32)
    add_custom_command(TARGET test_vit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${LIBTORCH_PATH}/bin"
        "${CMAKE_BINARY_DIR}/bin"
    )
    add_custom_command(TARGET test_vit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc16/bin"
        "${CMAKE_BINARY_DIR}/bin"
    )
    
    add_custom_command(TARGET showcase POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${LIBTORCH_PATH}/bin"
        "${CMAKE_BINARY_DIR}/bin"
    )
    add_custom_command(TARGET showcase POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc16/bin"
        "${CMAKE_BINARY_DIR}/bin"
    )
endif()
