cmake_minimum_required(VERSION 3.10)
project(MicroGrad)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Define paths for sources and headers
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# Collect all source files and header files
file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE HEADERS "${INCLUDE_DIR}/*.h")

# Include the header directory
include_directories(${INCLUDE_DIR})

# Group files for IDE organization
source_group("Sources" FILES ${SOURCES})
source_group("Headers" FILES ${HEADERS})

# Add the executable
add_executable(MicroGrad ${SOURCES} ${HEADERS})