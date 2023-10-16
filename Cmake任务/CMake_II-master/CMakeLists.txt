# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
#在此处设置项目名称
project(MAIN)

# # # # 以下内容禁止修改 # # # #
option(BUILD_A "build modules a" ON)
option(BUILD_B "build modules b" OFF)
option(BUILD_TESTS "build tests" OFF)

if(BUILD_TESTS)
    enable_testing()
endif()

if(BUILD_A)
    add_definitions(-DWITH_A)
endif()

if(BUILD_B)
    add_definitions(-DWITH_B)
endif()




#在此处指定头文件路径
find_package(OpenCV)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/common/rmath/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common/singleton/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/assembly1/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/assembly2/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/module1/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/module2/include

        ${OpenCV_INCLUDE_DIRS}
)


add_subdirectory(modules)
add_subdirectory(common)
# 小提示: module2 需要用到 pthread，即 target_link_libraries(xxx pthread)
# # # # 以上内容禁止修改 # # # #




# 在此处生成可执行程序
add_executable(client client.cpp)
add_executable(server server.cpp)
add_executable(assembly1test ${CMAKE_CURRENT_SOURCE_DIR}/modules/assembly1/test/assembly1_test.cpp)
target_link_libraries(client common modules ${OpenCV_LIBS} pthread)
target_link_libraries(server common modules ${OpenCV_LIBS} pthread)
target_link_libraries(assembly1test common modules ${OpenCV_LIBS} pthread)
add_test(NAME assembly1_test COMMAND assembly1test)



