cmake_minimum_required(VERSION 3.10)

project(COMMON)


set(lib_src ${CMAKE_CURRENT_SOURCE_DIR}/rmath/src/rmath.cpp)

add_library(common ${lib_src})

set_target_properties(common PROPERTIES LINKER_LANGUAGE CXX)