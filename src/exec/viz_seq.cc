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
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <pcl/visualization/common/common.h>
#include "mesh_sampling.h"

#include "json.h"
#include "utils.h"
#include "MapManager.h"
#include <bits/stdc++.h>

namespace {

bool g_disable_viewer = true;

using namespace std;
using namespace lass;
using namespace pcl;
using namespace cv;
namespace fs = std::experimental::filesystem;
using json = nlohmann::json;

void generate_centers(json& j, vector<PointXYZL>& centers) {
    LOG(INFO) << centers.size() << endl;

    // 0 is invalid
    LOG(INFO) << "center of label 0: " << centers[0] << endl;
    j.push_back(
        {centers[0].x, centers[0].y, centers[0].z, 0, 0, 0}
    );

    for (int idx = 1; idx < centers.size(); idx++) {
        auto& center = centers[idx];
        CHECK(idx == center.label || center.label == 0) << "idx = " << idx << ", label = " << center.label << endl;
        unsigned char r, g, b;
        // GroundColorMix(r, g, b, normalize_value(idx, 0, centers.size()));
        label_to_rgb(r,g , b, idx);
        {
            // debug scope
            // make sure each color map to only one label
            uint32_t unique_key = (uint32_t(r) << 16) + (uint32_t(g) << 8) + uint32_t(b);
            static std::map<uint32_t, uint32_t> color_map;
            if (color_map.count(unique_key) > 0) {
                CHECK(color_map[unique_key] == idx);
            } else {
                color_map[unique_key] = idx;
            }     
        }
        j.push_back(
            {center.x, center.y, center.z, r, g, b}
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

void processing(std::shared_ptr<MapManager>& mm, float voxel_resolution = 0.015, float seed_resolution = 0.4) {
    mm->m_index_of_landmark = PointIndices::Ptr{new PointIndices};
    mm->m_index_of_landmark->indices.resize(mm->m_pcd->points.size());
    for (int idx = 0; idx < mm->m_pcd->points.size(); idx++) {
        mm->m_index_of_landmark->indices[idx] = idx;
    }
    
    if (!g_disable_viewer) {
        mm->update_view();
        mm->show_point_cloud();
    }

    mm->supervoxel_landmark_clustering(voxel_resolution, seed_resolution, 1.0, 0.0, 0.1);
    // mm->set_view_target_pcd(true);
    if (!g_disable_viewer) {
        mm->update_view();
        mm->show_point_cloud();
    }
    
    mm->filter_minor_segmentations(30);
    mm->m_pcd = mm->extract_points_from_supervoxel();
    if (!g_disable_viewer) {
        mm->update_view();
        mm->show_point_cloud();
    }
}

void visualize_centers(json& j_output, std::shared_ptr<MapManager>& mm) {
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
    // float rgb_focal = 525;
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
    auto mm = std::make_shared<MapManager>();
#if 1  // set 1 to disable viewer, for batch generation
    mm->m_disable_viewer = true;
#endif
    mm->load_ply_pcl(ply_path);
    processing(mm);
    mm->prepare_octree_for_target_pcd(0.02);

    // generate json
    // generate_centers(j_output["centers"], mm);
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
        // e.block<3, 1>(0, 3) = e.block<3, 1>(0, 3) + e.block<3, 3>(0, 0).inverse() * Eigen::Vector3f(0.006880049706, -0.00333539999278, -0.0223485151692);
        e.block<3, 1>(0, 3) = e.block<3, 1>(0, 3) - e.block<3, 3>(0, 0).inverse() * Eigen::Vector3f(0.0245, 0, 0);
        mm->raycasting_pcd(e, intrsinsics, pcd, std::vector<pcl::PointXYZRGB>(), false);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                auto& pt = pcd->points[j * width + i];
                auto& c = save_img.at<cv::Vec3b>(j, i);
                auto label = label_mapping[pt.label];
                if (pt.label == 0) {
                    c[0] = c[1] = c[2] = 0;
                } else {
                    // // GroundColorMix(c[0], c[1], c[2], normalize_value(pt.label, 0, mm->max_target_label));
                    
                    label_to_rgb(c[0], c[1], c[2], pt.label);
                }
            }
        }

        string rgb_img_fn = fn.parent_path() / (fn.filename().stem().stem().string() + ".color.png");
        cout << rgb_img_fn << endl;
        auto rgb_img = cv::imread(rgb_img_fn);
        // cv::imwrite(image_fn, save_img);
        rgb_img = rgb_img / 2 + save_img / 2;
        cv::imshow("show", save_img);
        cv::imshow("rgb", rgb_img);
        // cv::imwrite("show.png", save_img);
        // cv::imwrite(image_fn, save_img);
        cv::waitKey(0);
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

void test_ply(int argc, char** argv) {
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
    const float voxel_resolution = j_input["voxel_resolution"].get<float>();
    const float seed_resolution = j_input["seed_resolution"].get<float>();
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
    // float rgb_focal = 525;
    float rgb_focal = 520;
    const float fx = rgb_focal / scale, fy = rgb_focal / scale;
    const float cx = 320 / scale, cy = 240 / scale;
    const camera_intrinsics intrsinsics{
        .cx = cx, .cy = cy, .fx = fx, .fy = fy, .width = width, .height = height
    };

    vector<Eigen::Matrix4f> es;
    vector<string> fns, relative_fns;

    PolygonMesh mesh;
    io::loadPLYFile(ply_path, mesh);

    const int sample_number = 10000000;

    auto mm = std::make_shared<MapManager>();
    mm->load_and_sample_ply(ply_path, sample_number);
    processing(mm, voxel_resolution, seed_resolution);

    mm->prepare_octree_for_target_pcd(0.02);

    PointCloud<PointXYZL>::Ptr pcd{new PointCloud<PointXYZL>};
    cv::Mat save_img(cv::Size(width, height), CV_8UC3);
    vector<string> image_fns;

    // load camera poses and file names
    load_7scenes_poses(base_path, scene, es, fns, relative_fns, true, false);

    // es.resize(10);

    vector<float> scores(mm->max_target_label, 0);
    PointXYZL default_center;
    default_center.x = default_center.y = default_center.z = default_center.label = 0;
    vector<PointXYZL> centers(mm->max_target_label, default_center);
    for (int idx = 0; idx < es.size(); idx++) {
        LOG_IF(INFO, idx % 10 == 0) << idx << "/" << es.size() << endl;
        Eigen::Matrix4f e = es[idx];
        fs::path fn = fns[idx];
        string image_fn;
        process_path(fn, target_path, image_fn);

        // ybbbbt: dirty fix for new interface
        mm->m_labeled_pcd = mm->m_target_pcd;
        // fix 7 scenes gt
        e.block<3, 1>(0, 3) = e.block<3, 1>(0, 3) - e.block<3, 3>(0, 0).inverse() * Eigen::Vector3f(0.0245, 0, 0);
        mm->raycasting_pcd(e, intrsinsics, pcd, std::vector<pcl::PointXYZRGB>(), false);
        update_candidate_list(pcd, scores, centers, width);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                auto& pt = pcd->points[j * width + i];
                auto& c = save_img.at<cv::Vec3b>(j, i);
                if (pt.label == 0) {
                    c[0] = c[1] = c[2] = 0;
                } else {
                // GroundColorMix(c[0], c[1], c[2], normalize_value(pt.label, 0, mm->max_target_label));
                label_to_rgb(c[0], c[1], c[2], pt.label);
                }
            }
        }

        // string rgb_img_fn = fn.parent_path() / (fn.filename().stem().stem().string() + ".color.png");
        // cout << rgb_img_fn << endl;
        // auto rgb_img = cv::imread(rgb_img_fn);
        // cv::imwrite(image_fn, save_img);
        // rgb_img = rgb_img / 2 + save_img / 2;
        // cv::imshow("show", save_img);
        // cv::imshow("rgb", rgb_img);
        // cv::imwrite("show.png", save_img);
        cv::imwrite(image_fn, save_img);
        // cv::waitKey(0);
    }

    generate_centers(j_output["centers"], centers);
    generate_es(j_output["camera_poses"], relative_fns, es);

    // generate json
    ofstream jout(parameters_output_fn);
    jout << j_output.dump(4);

    // mm->update_view();
    // mm->update_centers_to_viewer(centers);
    // mm->show_point_cloud();

    load_7scenes_poses(base_path, scene, es, fns, relative_fns, false, true);

    for (int idx = 0; idx < es.size(); idx++) {
        LOG_IF(INFO, idx % 10 == 0) << idx << "/" << es.size() << endl;
        Eigen::Matrix4f e = es[idx];
        fs::path fn = fns[idx];
        string image_fn;
        process_path(fn, target_path, image_fn);

        // ybbbbt: dirty fix for new interface
        mm->m_labeled_pcd = mm->m_target_pcd;
        // fix 7 scenes gt
        e.block<3, 1>(0, 3) = e.block<3, 1>(0, 3) - e.block<3, 3>(0, 0).inverse() * Eigen::Vector3f(0.0245, 0, 0);
        mm->raycasting_pcd(e, intrsinsics, pcd, std::vector<pcl::PointXYZRGB>(), false);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                auto& pt = pcd->points[j * width + i];
                auto& c = save_img.at<cv::Vec3b>(j, i);
                if (pt.label == 0) {
                    c[0] = c[1] = c[2] = 0;
                } else {
                    // GroundColorMix(c[0], c[1], c[2], normalize_value(pt.label, 0, mm->max_target_label));
                    label_to_rgb(c[0], c[1], c[2], pt.label);
                }
            }
        }

        // string rgb_img_fn = fn.parent_path() / (fn.filename().stem().stem().string() + ".color.png");
        // cout << rgb_img_fn << endl;
        // auto rgb_img = cv::imread(rgb_img_fn);
        // cv::imwrite(image_fn, save_img);
        // rgb_img = rgb_img / 2 + save_img / 2;
        // cv::imshow("show", save_img);
        // cv::imshow("rgb", rgb_img);
        // cv::imwrite("show.png", save_img);
        cv::imwrite(image_fn, save_img);
        // cv::waitKey(0);
    }

}

void associate_sc_with_label() {

}

inline void load_sc_data(float* sc_data_ptr, const std::string& fn, int width, int height) {
    CHECK(fs::exists(fn)) << fn << " not exists" << endl;

    ifstream ifsc(fn, ios::binary);
    ifsc.read((char *)(sc_data_ptr), sizeof(float) * 3 * width * height);
}

void test_ply_with_scene_coordinate(int argc, char** argv) {
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
    const float voxel_resolution = j_input["voxel_resolution"].get<float>();
    const float seed_resolution = j_input["seed_resolution"].get<float>();
    const string sc_data_dir = j_input["sc_data_dir"].get<std::string>();
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
    // float rgb_focal = 525;
    float rgb_focal = 520;
    const float fx = rgb_focal / scale, fy = rgb_focal / scale;
    const float cx = 320 / scale, cy = 240 / scale;
    const camera_intrinsics intrsinsics{
        .cx = cx, .cy = cy, .fx = fx, .fy = fy, .width = width, .height = height
    };
    float *sc_data_ptr = new float[width * height * 3 * sizeof(float)];

    vector<Eigen::Matrix4f> es;
    vector<string> fns, relative_fns;

    PolygonMesh mesh;
    io::loadPLYFile(ply_path, mesh);

    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPolygonMesh(mesh,"meshes",0);
    // viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    viewer->setShowFPS(false);

    while (!viewer->wasStopped ()){
        viewer->spinOnce (100);
    }

    const int sample_number = 10000000;

    auto mm = std::make_shared<MapManager>();
    mm->load_and_sample_ply(ply_path, sample_number);
    processing(mm, voxel_resolution, seed_resolution);

    auto labeled_pcd = mm->m_target_pcd;
    pcl::KdTreeFLANN<pcl::PointXYZL> tree;
    tree.setInputCloud(labeled_pcd);
    std::vector<int> point_indices(1);
    std::vector<float> distances(1);
    PointXYZL point;
    cout << "labeled pcd size = " << labeled_pcd->points.size() << endl;

    PointCloud<PointXYZL>::Ptr pcd{new PointCloud<PointXYZL>};

    pcd->points.resize(width * height);
    // cout << "pcd size 1  = " << pcd->points.size() << endl;

    cv::Mat save_img(cv::Size(width, height), CV_8UC3);
    vector<string> image_fns;

    // load camera poses and file names
    load_7scenes_poses(base_path, scene, es, fns, relative_fns, false, true);


    vector<float> scores(mm->max_target_label, 0);
    PointXYZL default_center;
    default_center.x = default_center.y = default_center.z = default_center.label = 0;
    vector<PointXYZL> centers(mm->max_target_label, default_center);
    for (int idx = 0; idx < es.size(); idx++) {
        LOG_IF(INFO, idx % 10 == 0) << idx << "/" << es.size() << endl;
        Eigen::Matrix4f e = es[idx];
        fs::path fn = fns[idx];
        string image_fn;
        std::string sc_fn = sc_data_dir + "/" + relative_fns[idx];
        sc_fn.replace(sc_fn.end() - 3, sc_fn.end(), "bin");
        cout << "fn = " << fn << endl;
        process_path(fn, target_path, image_fn);
        cout << "image fn = " << image_fn << endl;
        load_sc_data(sc_data_ptr, sc_fn, width, height);
        cout << "pcd size 3  = " << pcd->points.size() << endl;
        cout << "sc_fn = " <<  sc_fn << endl;

        // fix 7 scenes gt
        // mm->raycasting_pcd(e, intrsinsics, pcd, std::vector<pcl::PointXYZRGB>(), false);
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                int idx = j * width + i;
                // cout << "idx = " << idx << endl;
                auto& point = pcd->points[idx];
                // cout << "pcd size = " << pcd->points.size() << endl;
                point.x = sc_data_ptr[idx * 3 + 0];
                point.y = sc_data_ptr[idx * 3 + 1];
                point.z = sc_data_ptr[idx * 3 + 2];
                // cout << 2 << endl;

                if (point.x == 0 && point.y == 0 && point.z == 0) {
                    point.label = 0;
                } else {
                    if (tree.nearestKSearch(point, 1, point_indices, distances) > 0) {
                        point.label = labeled_pcd->points[point_indices[0]].label;
                    } else {
                        LOG(INFO) << "error: can not find nearest neighbor" << endl;
                        exit(0);
                    }
                }

            }
        }
        viewer->addPointCloud(pcd);
        viewer->spinOnce (100);
        viewer->saveScreenshot(image_fn);
        viewer->removePointCloud();
        // visualize_pcd(pcd);
    }
}
}

int main(int argc, char** argv) {
    /*
        argv[1] -- dataset path
        argv[2] -- mhd file
     */

    // test_raycasting_7scenes(argc, argv);
    // test_ply(argc, argv);
    test_ply_with_scene_coordinate(argc, argv);
    // test_normalize_rotations(argc, argv);
    // test_rendered_depth();
    // PointCloud<PointXYZRGB>::Ptr pcd{new PointCloud<PointXYZRGB>};
    // load_and_sample_obj(argv[1], 40000000, pcd);
    // visualize_pcd(pcd);
}
