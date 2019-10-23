#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
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
#include <experimental/filesystem>
#include <Eigen/Geometry>
#include "utils.h"
#include "MapManager.h"
#include "json.h"

using namespace std;
using namespace pcl;
using namespace lass;
using namespace pcl::visualization;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

void processing(shared_ptr<MapManager>& mm) {
    mm->filter_outliers(2, 10);
    mm->filter_landmarks_through_background();
    mm->supervoxel_landmark_clustering(0.7f);

    mm->set_view_type(1);
    mm->update_view();
    mm->show_point_cloud();

    mm->filter_supervoxels_through_background();
    mm->filter_points_near_cameras(2.0);
    mm->update_view();
    mm->show_point_cloud();

    mm->filter_minor_segmentations(30);
    mm->update_view();
    mm->show_point_cloud();

    mm->assign_supervoxel_label_to_filtered_pcd();
    mm->set_view_type(2);
    mm->update_view();
    mm->show_point_cloud();
}

void test_raycasting_robotcar(int argc, char** argv) {
    /*
        argv[1] -- info file
        argv[2] -- list file
        argv[3] -- dir
     */
    // initialize intrinsics and load data
    Eigen::Matrix3f intrinsics;
    const int scale = 4;
    const int width = 1024 / scale, height = 1024 / scale;
    const float fx = 400 / scale, fy = 400 / scale;
    const double cx_left = 500.107605 / scale, cy_left = 511.461426 / scale;
    const double cx_rear = 508.222931 / scale, cy_rear = 498.187378 / scale;
    const double cx_right = 502.503754 / scale, cy_right = 490.259033 / scale;
    const camera_intrinsics left_intrsinsics{
        .cx = cx_left, .cy = cy_left, .fx = fx, .fy = fy, .width = width, .height = height
    };
    const camera_intrinsics rear_intrsinsics{
        .cx = cx_rear, .cy = cy_rear, .fx = fx, .fy = fy, .width = width, .height = height
    };
    const camera_intrinsics right_intrsinsics{
        .cx = cx_right, .cy = cy_right, .fx = fx, .fy = fy, .width = width, .height = height
    };

    const char* info_fn = argv[1];
    const char* list_fn = argv[2];
    vector<Eigen::Matrix4f> es;
    vector<string>  image_fns;
    vector<int>     camera_types;
    load_info_file(info_fn, es);
    load_list_file(list_fn, es.size(), image_fns, camera_types);
    LOG(INFO) << "cameras: " << es.size() << endl;


    // process pointcloud
    auto mm = make_shared<MapManager>(argv[3]);
    mm->m_show_camera_extrinsics = true;
    mm->m_camera_extrinsics = es;
    mm->m_camera_types = camera_types;
    processing(mm);

    /* scope: json file output */ 
    {
        vector<PointXYZRGB>    centers{static_cast<unsigned long>(mm->max_target_label), PointXYZRGB{0, 0, 0}};
        vector<int>         point_counter(static_cast<unsigned long>(mm->max_target_label), 0);

        LOG(INFO) << centers.size() << endl;
        LOG(INFO) << point_counter.size() << endl;
        LOG(INFO) << mm->max_target_label << endl;
        LOG(INFO) << "process center and RGB" << endl;
        for (auto& pt: mm->m_target_pcd->points) {
            auto& label = pt.label;
            centers[label].x += pt.x;
            centers[label].y += pt.y;
            centers[label].z += pt.z;
            point_counter[label]++;
        }

        centers[0].r = centers[0].g = centers[0].b = 0;
        LOG(INFO) << "center of label 0: " << centers[0] << endl;
        json j_label;
        j_label.push_back(
            {centers[0].x, centers[0].y, centers[0].z, centers[0].r, centers[0].g, centers[0].b}
        );

        for (int idx = 1; idx < mm->max_target_label; idx++) {
            auto& center = centers[idx];
            center.x /= point_counter[idx];
            center.y /= point_counter[idx];
            center.z /= point_counter[idx];
            GroundColorMix(center.r, center.g, center.b, normalize_value(idx, 0, mm->max_target_label));
            j_label.push_back(
                {center.x, center.y, center.z, center.r, center.g, center.b}
            );
        }

        ofstream o_label{argv[4]};
        json j_es;
        o_label << std::setw(4) << j_label;

        LOG(INFO) << "process extrinsics" << endl;
        for (int idx = 0; idx < es.size(); idx++) {
            Eigen::Matrix4f e = es[idx];
            auto& camera_type = camera_types[idx];
            auto& image_fn = image_fns[idx];

            e(1, 3) *= -1; e(2, 3) *= -1;
            e(0, 1) *= -1; e(0, 2) *= -1;
            e(1, 0) *= -1; e(2, 0) *= -1;

            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 4; c++) {
                    j_es[image_fn.c_str()].push_back(e(r, c));
                }
            }
        }

        ofstream o_es{argv[5]};
        o_es << std::setw(4) << j_es;


        // train test split
        std::vector<std::string> train_list, test_list;
        for (int idx = 0; idx < es.size(); idx++) {
            auto& e = es[idx];
            auto& camera_type = camera_types[idx];
            auto& image_fn = image_fns[idx];
            if (idx % 10 >= 8) {
                test_list.push_back(image_fn);
            } else {
                train_list.push_back(image_fn);
            }
        }
        ofstream o_train_list{"train_list.json"};
        json j_train_list = train_list;
        o_train_list << std::setw(4) << j_train_list;
        ofstream o_test_list{"test_list.json"};
        json j_test_list = test_list;
        o_test_list << std::setw(4) << j_test_list;

        LOG(INFO) << "finished output json" << endl;
    }
    
    /* scope: raycasting */ 
    {
        mm->prepare_octree_for_target_pcd(0.5f);

        Eigen::Vector3f o, d, u;
        PointCloud<PointXYZL>::Ptr pcd{new PointCloud<PointXYZL>};
        cv::Mat save_img(cv::Size(width, height), CV_8UC3);
        cv::Vec3b color;

        int res = system("mkdir -p left right rear");
        for (int idx = 0; idx < es.size(); idx++) {
            Eigen::Matrix4f e = es[idx];
            auto& camera_type = camera_types[idx];
            auto& image_fn = image_fns[idx];

            e(1, 3) *= -1; e(2, 3) *= -1;
            e(0, 1) *= -1; e(0, 2) *= -1;
            e(1, 0) *= -1; e(2, 0) *= -1;

            switch (camera_type) {
            case 0:
                mm->raycasting_pcd(e, left_intrsinsics, pcd, "labeled");
                break;
            
            case 1:
                mm->raycasting_pcd(e, rear_intrsinsics, pcd, "labeled");
                break;
            
            case 2:
                mm->raycasting_pcd(e, right_intrsinsics, pcd, "labeled");
                break;

            default:
                CHECK(0) << "Alert!" << endl;
                break;
            }

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
            // LOG(INFO) << "image fn: " << image_fn << endl;
            fprintf(stdout, "\rProgress: %d / %zu", idx, es.size());
            fflush(stdout);

            cv::imwrite(image_fn, save_img);
            // cv::imshow("show", save_img);
            // cv::waitKey(0);
        }
    }

    
}

int main(int argc, char** argv) {
    test_raycasting_robotcar(argc, argv);
}
