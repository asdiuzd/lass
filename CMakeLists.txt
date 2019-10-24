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
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

# add_executable(cloud_viewer main.cc NVMProcessor.cc util.cc PlanarMap.cc)
# add_executable(test_extrinsics test_extrinsics.cc NVMProcessor.cc util.cc PlanarMap.cc)
# add_executable(test_raycasting test_raycasting.cc NVMProcessor.cc util.cc PlanarMap.cc)
# add_executable(test_build_meshes test_build_meshes.cc NVMProcessor.cc util.cc PlanarMap.cc)
# add_executable(test_pointcloud_processing test_pointcloud_processing.cc ${SOURCE_FILES})

# target_link_libraries(cloud_viewer ${PCL_LIBRARIES})
# target_link_libraries(test_extrinsics ${PCL_LIBRARIES})
# target_link_libraries(test_raycasting ${PCL_LIBRARIES})
# target_link_libraries(test_build_meshes ${PCL_LIBRARIES})
# target_link_libraries(test_pointcloud_processing ${PCL_LIBRARIES})

# target_include_directories(test_pointcloud_processing
#     SYSTEM PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
# )

target_compile_options(test_pointcloud_processing PUBLIC
    $<$<CONFIG:RELEASE>:-O3>
)

target_compile_options(test_pointcloud_processing PUBLIC
    $<$<CONFIG:DEBUG>:-Og -g -ggdb>
)
