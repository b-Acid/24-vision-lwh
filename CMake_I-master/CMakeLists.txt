# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.10)
#在此处设置项目名称
project(MAIN)


#在此处指定头文件路径
find_package(OpenCV)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modules/A1/include 
${CMAKE_CURRENT_SOURCE_DIR}/modules/A2/include 
${CMAKE_CURRENT_SOURCE_DIR}/modules/M1/include
${CMAKE_CURRENT_SOURCE_DIR}/modules/M2/include
${CMAKE_CURRENT_SOURCE_DIR}/common/kalman/include
${CMAKE_CURRENT_SOURCE_DIR}/common/math/include
${OpenCV_INCLUDE_DIRS}
)

####### 以下禁止修改 #######
add_subdirectory(common)   #
add_subdirectory(modules)  #
####### 以上禁止修改 #######



# 在此处生成可执行程序
link_libraries(common modules)
add_executable(main main.cpp)
target_link_libraries(test PUBLIC ${OpenCV_LIBS})


