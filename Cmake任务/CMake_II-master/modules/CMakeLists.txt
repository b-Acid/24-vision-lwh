cmake_minimum_required(VERSION 3.10)

project(MODULES)

set(lib_src ${CMAKE_CURRENT_SOURCE_DIR}/module1/src/module1a/m1.cpp 
            ${CMAKE_CURRENT_SOURCE_DIR}/module1/src/module1a/m2.cpp 
            ${CMAKE_CURRENT_SOURCE_DIR}/module1/src/module1a/m3.cpp 
            ${CMAKE_CURRENT_SOURCE_DIR}/module1/src/module1a/m4.cpp 
            ${CMAKE_CURRENT_SOURCE_DIR}/module1/src/module1b/module1b.cpp 
            ${CMAKE_CURRENT_SOURCE_DIR}/module2/src/client.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/module2/src/object.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/module2/src/open62541.c
            ${CMAKE_CURRENT_SOURCE_DIR}/module2/src/server.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/module2/src/variable.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/assembly1/src/assembly1.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/assembly2/src/assembly2.cpp)

add_library(modules ${lib_src})
set_target_properties(modules PROPERTIES LINKER_LANGUAGE CXX)





