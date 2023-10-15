cmake_minimum_required(VERSION 3.1)
project(yolotest)

#find the opencv and the qt5
find_package(OpenCV  REQUIRED)
#onnxruntime
set(ONNXRUNTIME_ROOT_PATH /home/bacid/文档/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/providers/cuda)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ../)
add_executable(yolotest "yolodetect.cpp")
target_link_libraries(yolotest ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

