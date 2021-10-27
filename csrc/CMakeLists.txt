cmake_minimum_required(VERSION 2.8)
project(chaos_brain_ext)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O0 ${OpenMP_CXX_FLAGS}")
set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(PythonInterp REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/pybind11/)

set(SOURCEFILES mc/cards.cpp mc/samples.cpp mc/simulator.cpp mc/tables.cpp mc/tools.cpp)

pybind11_add_module(mc MODULE pybind.cpp ${SOURCEFILES})
