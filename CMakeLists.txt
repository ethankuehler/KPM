cmake_minimum_required(VERSION 3.24)
project(KPM)

set(CMAKE_CXX_STANDARD 17)

link_directories("/opt/homebrew/lib")
include_directories("/opt/homebrew/include")
include_directories("/opt/homebrew/include/eigen3")

find_package (Eigen3 3.4 REQUIRED)
find_package(Spectra REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(KPM KPM.cpp)
link_libraries(KPM Eigen3 OpenMP)

