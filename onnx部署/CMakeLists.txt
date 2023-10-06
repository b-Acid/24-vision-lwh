# cmake needs this line

cmake_minimum_required(VERSION 3.9.1)
 
# Define project name
project(yourproject)

# Find OpenCV
find_package(OpenCV)
#find_package(CUDA REQUIRED)


# Add includes
#include_directories(${CUDA_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Enable C++11
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CUDA_STANDARD 11)   
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)




set(ONNXRUNTIME_INCLUDE_DIRS /home/bacid/文档/onnxruntime/include/)
set(ONNXRUNTIME_LIBS /home/bacid/文档/onnxruntime/lib/libonnxruntime.so)

include_directories(${ONNXRUNTIME_INCLUDE_DIRS})


# Link your application with other libraries

add_executable(${PROJECT_NAME} "test.cpp")
#target_link_libraries(${PROJECT_NAME}  ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME}  ${ONNXRUNTIME_LIBS} ${OpenCV_LIBS})

