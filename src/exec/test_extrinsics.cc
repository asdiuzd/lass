#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/concave_hull.h>
#include <Eigen/Geometry>

#include "utils.h"
#include "MapManager.h"

// #include <pcl/filters/statistical_outlier_removal.h>
// #include <pcl/filters/radius_outlier_removal.h>
// #include <pcl/filters/extract_indices.h>


typedef pcl::PointXYZRGB PointType;
using namespace std;
using namespace pcl;
using namespace pcl::visualization;
using namespace lass;

void test_load_extrinsics(int argc, char** argv) {
    /*
        argv[1] -- info file
        argv[2] -- list file
        argv[3] -- name of dir
     */
    const string info_fn = argv[1];  
    const string list_fn = argv[2];
    const string dir     = argv[3];

    Eigen::Matrix4f left, rear, right;
    Eigen::Matrix3f intrinsics;
    vector<Eigen::Matrix4f> es;
    vector<string>  image_fns;
    vector<int>     camera_types;
    PointCloud<PointXYZRGB>::Ptr pcd(new PointCloud<PointXYZRGB>);

    intrinsics << 400, 0, 500.10765, 0, 400, 511.461426, 0, 0, 1;

    load_info_file(info_fn.c_str(), es);
    load_list_file(list_fn.c_str(), es.size(), image_fns, camera_types);

    LOG(INFO) << "number of cameras: " << es.size() << endl;

    auto mm = make_unique<MapManager>(dir);
    mm->dye_through_semantics();
    mm->filter_outliers();
    mm->update_view();

    Eigen::Vector3f o, d, u;

    pcd->points.resize(es.size() * 5);
    for (int idx = 0; idx < es.size(); idx++) {
        auto &e = es[idx];
        Eigen::Vector3f c_nvm = e.block(0, 3, 3, 1), t_e;
        Eigen::Matrix3f R_nvm = e.block(0, 0, 3, 3), R_e;

        c_nvm(1) *= -1; 
        c_nvm(2) *= -1;
        R_nvm(0, 1) *= -1; 
        R_nvm(0, 2) *= -1;
        R_nvm(1, 0) *= -1; 
        R_nvm(2, 0) *= -1;

        c_nvm = -R_nvm.transpose() * c_nvm;
        d = R_nvm.transpose() * Eigen::Vector3f::UnitZ();

        auto &pt = pcd->points[idx * 5];
        pt.r = pt.g = pt.b = 200;
        pt.x = c_nvm(0);
        pt.y = c_nvm(1);
        pt.z = c_nvm(2);
        for (int s_idx = 1; s_idx < 5; s_idx++) {
            auto &pt = pcd->points[idx * 5 + s_idx];
            pt.r = pt.g = pt.b = 0;
            switch (camera_types[idx]) {
            case 0:
                pt.r = 200;
                break;
            case 1:
                pt.g = 200;
                break;
            case 2:
                pt.b = 200;
                break;
            default:
                break;
            }
            pt.x = c_nvm(0) + d(0) * s_idx * 0.2;
            pt.y = c_nvm(1) + d(1) * s_idx * 0.2;
            pt.z = c_nvm(2) + d(2) * s_idx * 0.2;
        }
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pcd);
    mm->m_viewer->addPointCloud(pcd, rgb, "camera poses");
    mm->m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "camera poses");
    mm->show_point_cloud();
    // int idx = 0;
    // for (auto &e: es) { // e = [R|c]
    //     Eigen::Vector3f c_nvm = e.block(0, 3, 3, 1), t_e;
    //     Eigen::Matrix3f R_nvm = e.block(0, 0, 3, 3), R_e;
    //     cout << image_fns[idx] << endl;

    //     c_nvm(1) *= -1; c_nvm(2) *= -1;

    //     R_nvm(0, 1) *= -1; R_nvm(0, 2) *= -1;
    //     R_nvm(1, 0) *= -1; R_nvm(2, 0) *= -1;

    //     o = - R_nvm.transpose() * c_nvm;      //< camera to world
    //     e.block(0, 0, 3, 3) = R_nvm.transpose().eval();
    //     e.block(0, 3, 3, 1) = o;

    //     mm->m_viewer->setCameraParameters(intrinsics, e);
    //     // pm->m_viewer->saveScreenshot(image_fns[idx]);
    //     mm->show_point_cloud();
    //     // std::this_thread::sleep_for(std::chrono::microseconds(1000000));

    //     idx++;
    // }
    // mm->show_point_cloud();
}

int main(int argc, char** argv) {
    test_load_extrinsics(argc, argv);
}