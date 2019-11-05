cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(lass VERSION 0.0.1 LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(SOURCE_FILES src/utils.cc)

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GFLAGS REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Found CCache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

include_directories(SYSTEM 
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_subdirectory(${PROJECT_SOURCE_DIR}/src/exec)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_library(lass SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MapManager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MapManagerNG.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mesh_sampling.cc
)

target_compile_options(lass PUBLIC
    $<$<CONFIG:RELEASE>:-O3>
)

target_compile_options(lass PUBLIC
    $<$<CONFIG:DEBUG>:-Og -g -ggdb>
)

target_include_directories(lass
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)


target_link_libraries(lass
    PUBLIC
    ${PCL_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    stdc++fs
)
