#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <experimental/filesystem>
#include <pcl/io/ply_io.h>

#include "utils.h"
#include "MapManager.h"

using namespace std;
using namespace lass;
using namespace pcl;
namespace fs = std::experimental::filesystem;

void processing(shared_ptr<MapManager>& mm) {
    mm->m_index_of_landmark = PointIndices::Ptr{new PointIndices};
    mm->m_index_of_landmark->indices.resize(mm->m_pcd->points.size());
    for (int idx = 0; idx < mm->m_pcd->points.size(); idx++) {
        mm->m_index_of_landmark->indices[idx] = idx;
    }
    mm->update_view();
    mm->show_point_cloud();

    mm->supervoxel_landmark_clustering(0.015, 0.4, 1.0, 0.0, 0.0);
    mm->set_view_target_pcd(true);
    mm->update_view();
    mm->show_point_cloud();
    
    mm->filter_minor_segmentations(30);
    mm->update_view();
    mm->show_point_cloud();

}

void test_raycasting_7scenes(int argc, char** argv) {
    /*
        argv[1] - base_path
        argv[2] - scene
     */
    const char * ply_path = argv[1];
    const char * base_path = argv[2];
    const char * scene = argv[3];

    const int scale = 1;
    const int width = 640 / scale, height = 480 / scale;
    const float fx = 585 / scale, fy = 585 / scale;
    const float cx = 320 / scale, cy = 240 / scale;
    const camera_intrinsics intrsinsics{
        .cx = cx, .cy = cy, .fx = fx, .fy = fy, .width = width, .height = height
    };

    vector<Eigen::Matrix4f> training_es, test_es;

    load_7scenes_poses(base_path, scene, training_es, test_es);

    auto mm = make_shared<MapManager>();
    mm->load_ply_pcl(ply_path);
    processing(mm);


    PointCloud<PointXYZL>::Ptr pcd{new PointCloud<PointXYZL>};
    cv::Mat save_img(cv::Size(width, height), CV_8UC3);

    for (auto& e : training_es) {
        mm->raycasting_target_pcd(e, intrsinsics, pcd, 0.02, false);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                auto& pt = pcd->points[j * width + i];
                auto& c = save_img.at<cv::Vec3b>(j, i);

                if (pt.label == 0) {
                    c[0] = c[1] = c[2] = 0;
                } else {
                    GroundColorMix(c[0], c[1], c[2], normalize_value(pt.label, 0, mm->max_target_label));
                }
            }
        }

        cv::imshow("show", save_img);
        cv::waitKey(0);
        // fillHoles(save_img);
        // cv::imshow("show2", save_img);
        // cv::waitKey(0);
        // cv::imwrite(image_fn, save_img);
        // cv::imshow("show", save_img);
        // cv::waitKey(0);
    }
}

int main(int argc, char** argv) {
    /*
        argv[1] -- dataset path
        argv[2] -- mhd file
     */
    // fs::path p(argv[1]);
    // string mhd_fn = (p / argv[2]).string();
    // mhd_structure file_data;
    // load_mhd_file(mhd_fn.c_str(), file_data);
    // cout << (p / file_data.data_file) << endl;
    // auto mm = make_shared<MapManager>();
    // mm->load_ply_pcl(argv[1]);
    // mm->m_index_of_landmark = PointIndices::Ptr{new PointIndices};
    // mm->m_index_of_landmark->indices.resize(mm->m_pcd->points.size());
    // for (int idx = 0; idx < mm->m_pcd->points.size(); idx++) {
    //     mm->m_index_of_landmark->indices[idx] = idx;
    // }
    // mm->update_view();
    // mm->show_point_cloud();

    // mm->supervoxel_landmark_clustering(0.02, 0.4, 1.0, 1.0, 1.0);
    // mm->set_view_target_pcd(true);
    // mm->update_view();
    // mm->show_point_cloud();
    
    // mm->filter_minor_segmentations(30);
    // mm->update_view();
    // mm->show_point_cloud();

    test_raycasting_7scenes(argc, argv);
}