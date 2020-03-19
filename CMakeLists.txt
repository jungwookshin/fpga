cmake_minimum_required (VERSION 2.8.11)
project (base)

set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) 

if(APPLE)
	find_library(COREFOUNDATION_LIBRARY CoreFoundation )
endif()


include_directories (
    ${PROJECT_SOURCE_DIR},
    ./
)

add_executable(fpga fpga.cpp)

install(PROGRAMS fpga DESTINATION .)

