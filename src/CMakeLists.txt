add_executable(test_pointcloud_processing test_pointcloud_processing.cc utils.cc)
target_link_libraries(test_pointcloud_processing ${PCL_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} stdc++fs)

add_executable(test_mapmanager test_mapmanager.cc utils.cc MapManager.cc)
target_link_libraries(test_mapmanager ${PCL_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} stdc++fs)