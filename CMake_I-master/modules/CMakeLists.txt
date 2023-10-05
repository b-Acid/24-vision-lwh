cmake_minimum_required(VERSION 3.10)

project(MODULES)

set(lib_src ${CMAKE_CURRENT_SOURCE_DIR}/A1/src/A11.cpp ${CMAKE_CURRENT_SOURCE_DIR}/A1/src/A12.cpp ${CMAKE_CURRENT_SOURCE_DIR}/A1/src/A13.cpp ${CMAKE_CURRENT_SOURCE_DIR}/A2/src/A2.cpp ${CMAKE_CURRENT_SOURCE_DIR}/M1/src/M1.cpp ${CMAKE_CURRENT_SOURCE_DIR}/M2/src/M2.cpp)

add_library(modules ${lib_src})

set_target_properties(modules PROPERTIES LINKER_LANGUAGE CXX)




