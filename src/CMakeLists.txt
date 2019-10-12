add_executable(test_pointcloud_processing test_pointcloud_processing.cc utils.cc)
target_link_libraries(test_pointcloud_processing ${PCL_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} stdc++fs)

add_executable(test_mapmanager test_mapmanager.cc utils.cc MapManager.cc)
target_link_libraries(test_mapmanager ${PCL_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} stdc++fs)

add_executable(test_extrinsics test_extrinsics.cc utils.cc MapManager.cc)
target_link_libraries(test_extrinsics ${PCL_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} stdc++fs)

add_executable(test_raycasting test_raycasting.cc utils.cc MapManager.cc)
target_link_libraries(test_raycasting ${PCL_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} stdc++fs)

add_executable(test_tsdf test_tsdf.cc utils.cc MapManager.cc)
target_link_libraries(test_tsdf ${PCL_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} stdc++fs)

add_executable(test_LODR test_LODR.cc utils.cc)
target_link_libraries(test_LODR ${PCL_LIBRARIES} ${GFLAGS_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} stdc++fs)