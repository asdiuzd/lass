#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <cstdio>
#include <experimental/filesystem>
#include <pcl/io/ply_io.h>

#include "json.h"
#include "utils.h"
#include "MapManager.h"
#include <bits/stdc++.h>

using namespace std;
using namespace lass;
using namespace pcl;
namespace fs = std::experimental::filesystem;
using json = nlohmann::json;

void generate_centers(json& j, shared_ptr<MapManager>& mm) {
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
    j.push_back(
        {centers[0].x, centers[0].y, centers[0].z, centers[0].r, centers[0].g, centers[0].b}
    );

    for (int idx = 1; idx < mm->max_target_label; idx++) {
        auto& center = centers[idx];
        center.x /= point_counter[idx];
        center.y /= point_counter[idx];
        center.z /= point_counter[idx];
        GroundColorMix(center.r, center.g, center.b, normalize_value(idx, 0, mm->max_target_label));
        {
            // debug scope
            // make sure each color map to only one label
            uint32_t unique_key = (uint32_t(center.r) << 16) + (uint32_t(center.g) << 8) + uint32_t(center.b);
            static std::map<uint32_t, uint32_t> color_map;
            if (color_map.count(unique_key) > 0) {
                CHECK(color_map[unique_key] == idx);
            } else {
                color_map[unique_key] = idx;
            }     
        }
        j.push_back(
            {center.x, center.y, center.z, center.r, center.g, center.b}
        );
    }
}

void generate_es(json& j, vector<string>& image_fns, vector<Eigen::Matrix4f>& es) {
    for (int idx = 0; idx < es.size(); idx++) {
        auto& e = es[idx];
        auto& image_fn = image_fns[idx];

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                j[image_fn.c_str()].push_back(e(r, c));
            }
        }
    }

}

void process_path(fs::path& fn, const string& target_path, string& output_fn) {
    fs::path target_fn = target_path;

    vector<fs::path> path_parts;
    for (const auto& p : fn) {
        path_parts.emplace_back(p);
    }

    target_fn = target_fn / path_parts[path_parts.size() - 2];

    if (!fs::exists(target_fn)) {
        fs::create_directory(target_fn);
    }

    target_fn /= path_parts[path_parts.size() - 1].stem().stem();
    target_fn += ".segmentation.png";
    output_fn = target_fn.string();
}

void processing(shared_ptr<MapManager>& mm) {
    mm->m_index_of_landmark = PointIndices::Ptr{new PointIndices};
    mm->m_index_of_landmark->indices.resize(mm->m_pcd->points.size());
    for (int idx = 0; idx < mm->m_pcd->points.size(); idx++) {
        mm->m_index_of_landmark->indices[idx] = idx;
    }
    mm->update_view();
    mm->show_point_cloud();

    mm->supervoxel_landmark_clustering(0.015, 0.4, 1.0, 0.0, 0.0);
    // mm->set_view_target_pcd(true);
    // mm->update_view();
    // mm->show_point_cloud();
    
    mm->filter_minor_segmentations(30);
    mm->m_pcd = mm->extract_points_from_supervoxel();
    mm->update_view();
    mm->show_point_cloud();

}

void visualize_centers(json& j_output, shared_ptr<MapManager>& mm) {
    PointCloud<PointXYZL>::Ptr center_cloud{new PointCloud<PointXYZL>};
    for (int idx = 0; idx < j_output["centers"].size(); idx++) {
        PointXYZL pt;
        pt.x = j_output["centers"][idx][0].get<float>();
        pt.y = j_output["centers"][idx][1].get<float>();
        pt.z = j_output["centers"][idx][2].get<float>();
        pt.label = mm->max_target_label + 1;
        center_cloud->points.emplace_back(pt);
    }
    mm->m_viewer->addPointCloud(center_cloud, "centers");
    mm->m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "centers");
    mm->show_point_cloud();
}

void test_raycasting_7scenes(int argc, char** argv) {
    /*
        argv[1] - json fn
     */
    json j_input;
    const char * json_fn = argv[1];
    CHECK(fs::exists(json_fn)) << "json does not exist: " << json_fn << endl;
    ifstream j_in(json_fn);
    j_in >> j_input;

    const string ply_path = j_input["model"].get<string>();
    const string base_path = j_input["base_path"].get<string>();
    const string scene = j_input["scene"].get<string>();
    string target_path = (fs::path(j_input["target_path"].get<string>()) / scene).string();
    const string parameters_output_fn = fs::path(target_path) / j_input["parameters_fn"].get<string>();
    json j_output;
    // const char * target_path = target_path_str.c_str();

    CHECK(fs::exists(ply_path)) << "model does not exist: " << ply_path << endl;
    CHECK(fs::exists(base_path)) << "path does not exist: " << base_path << endl;
    if (!fs::exists(target_path)) {
        // fs::create_directory(target_path);
        string cmd = "mkdir -p " + target_path;
        if (system(cmd.c_str()) == -1) {
            cout << "failed to create " << cmd << endl;
            return;
        }
    }
    CHECK(fs::exists(target_path)) << "path does not exist: " << target_path << endl;

    const int scale = 1;
    const int width = 640 / scale, height = 480 / scale;
    // WARNING(ybbbbt): focal length for RGB: 520, for depth: 585
    float rgb_focal = 520;
    const float fx = rgb_focal / scale, fy = rgb_focal / scale;
    const float cx = 320 / scale, cy = 240 / scale;
    const camera_intrinsics intrsinsics{
        .cx = cx, .cy = cy, .fx = fx, .fy = fy, .width = width, .height = height
    };

    vector<Eigen::Matrix4f> es;
    vector<string> fns, relative_fns;

    // load camera poses and file names
    load_7scenes_poses(base_path, scene, es, fns, relative_fns);

    // process map
    auto mm = make_shared<MapManager>();
#if 1  // set 1 to disable viewer, for batch generation
    mm->m_disable_viewer = true;
#endif
    mm->load_ply_pcl(ply_path);
    processing(mm);
    mm->prepare_octree_for_target_pcd(0.02);

    // generate json
    generate_centers(j_output["centers"], mm);
    generate_es(j_output["camera_poses"], relative_fns, es);
    ofstream jout(parameters_output_fn);
    jout << j_output.dump(4);

    // visualize_centers(j_output, mm);

    PointCloud<PointXYZL>::Ptr pcd{new PointCloud<PointXYZL>};
    cv::Mat save_img(cv::Size(width, height), CV_8UC3);
    vector<string> image_fns;

    vector<int>    label_mapping(mm->max_target_label);
    for (int idx = 0; idx < mm->max_target_label; idx++) {
        label_mapping[idx] = idx;
    }
    shuffle(label_mapping.begin(), label_mapping.end(), default_random_engine(0));

    for (int idx = 0; idx < es.size(); idx++) {
        Eigen::Matrix4f e = es[idx];
        fs::path fn = fns[idx];
        string image_fn;
        process_path(fn, target_path, image_fn);
        cout << fn << endl;
        cout << target_path << endl;
        cout << image_fn << endl;

        // ybbbbt: dirty fix for new interface
        mm->m_labeled_pcd = mm->m_target_pcd;
        // fix 7 scenes gt
        e.block<3, 1>(0, 3) = e.block<3, 1>(0, 3) - e.block<3, 3>(0, 0).transpose() * Eigen::Vector3f(0.0245, 0, 0);
        mm->raycasting_pcd(e, intrsinsics, pcd, std::vector<pcl::PointXYZRGB>(), false);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                auto& pt = pcd->points[j * width + i];
                auto& c = save_img.at<cv::Vec3b>(j, i);
                auto label = label_mapping[pt.label];
                if (pt.label == 0) {
                    c[0] = c[1] = c[2] = 0;
                } else {
                    GroundColorMix(c[0], c[1], c[2], normalize_value(pt.label, 0, mm->max_target_label));
                }
            }
        }

        // cv::imwrite(image_fn, save_img);
        // cv::imshow("show", save_img);
        // cv::imwrite("show.png", save_img);
        cv::imwrite(image_fn, save_img);
        // cv::waitKey(0);
    }
}

void test_normalize_rotations(int argc, char ** argv) {
    /*
        argv[1] - json fn
     */
    json j_input;
    const char * json_fn = argv[1];
    CHECK(fs::exists(json_fn)) << "json does not exist: " << json_fn << endl;
    ifstream j_in(json_fn);
    j_in >> j_input;

    const string ply_path = j_input["model"].get<string>();
    const string base_path = j_input["base_path"].get<string>();
    const string scene = j_input["scene"].get<string>();
    string target_path = (fs::path(j_input["target_path"].get<string>()) / scene).string();
    const string parameters_output_fn = fs::path(target_path) / j_input["parameters_fn"].get<string>();
    json j_output;
    // const char * target_path = target_path_str.c_str();

    CHECK(fs::exists(ply_path)) << "model does not exist: " << ply_path << endl;
    CHECK(fs::exists(base_path)) << "path does not exist: " << base_path << endl;
    if (!fs::exists(target_path)) {
        // fs::create_directory(target_path);
        string cmd = "mkdir -p " + target_path;
        if (system(cmd.c_str()) == -1) {
            cout << "failed to create " << cmd << endl;
            return;
        }
    }
    CHECK(fs::exists(target_path)) << "path does not exist: " << target_path << endl;

    normalize_7scenes_poses(base_path, scene);
}

int main(int argc, char** argv) {
    /*
        argv[1] -- dataset path
        argv[2] -- mhd file
     */

    test_raycasting_7scenes(argc, argv);
    // test_normalize_rotations(argc, argv);
}
