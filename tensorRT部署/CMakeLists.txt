cmake_minimum_required(VERSION 3.10)
project(test)
find_package(OpenCV  REQUIRED)
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")

set(CUDA_INCLUDE_DIRS  /usr/local/cuda-11.7/targets/x86_64-linux/include)
set(CUDA_LIBRARIES  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudart.so
                    /usr/lib/x86_64-linux-gnu/libnvinfer.so
                    /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
)
set(EXECUTABLE_OUTPUT_PATH ../)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(inference "tensorRTdeploy.cpp")
target_link_libraries(inference ${OpenCV_LIBS} ${CUDA_LIBRARIES})
add_executable(export "export2engine.cpp")
target_link_libraries(export ${OpenCV_LIBS} ${CUDA_LIBRARIES})

