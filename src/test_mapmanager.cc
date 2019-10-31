#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>

#include <experimental/filesystem>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/io/pcd_io.h>
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
#include "json.h"
#include "MapManager.h"

typedef pcl::PointXYZRGB PointType;
using namespace std;
using namespace pcl;
using namespace pcl::visualization;
using namespace lass;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

void test_clustering(int argc, char** argv);
void test_io(int argc, char** argv);
void test_landmark(int argc, char** argv);
void test_background(int argc, char** argv);
void test_filter(int argc, char** argv);

int main(int argc, char** argv) {
    
    // test_io(argc, argv);
    test_landmark(argc, argv);
    // test_background(argc, argv);
    // test_filter(argc, argv);
    // test_clustering(argc, argv);
}

void test_clustering(int argc, char** argv) {
    CHECK(fs::exists(argv[1])) << argv[1] << " does not exist" << endl;

    auto mm = make_unique<MapManager>(argv[1]);
    mm->filter_outliers(2, 15);
    // mm->euclidean_landmark_clustering();
    mm->supervoxel_landmark_clustering();
    mm->set_view_type(1);
    mm->update_view();
    mm->show_point_cloud();

    mm->filter_minor_segmentations(30);
    mm->update_view();
    mm->show_point_cloud();
}

void test_filter(int argc, char** argv) {
    /*
        argv[1] - input dir
    */
    CHECK(fs::exists(argv[1])) << argv[1] << " does not exist" << endl;

    auto mm = make_unique<MapManager>(argv[1]);
    mm->update_view();
    mm->show_point_cloud();
    mm->filter_outliers();
    mm->update_view();
    mm->show_point_cloud();
    visualize_pcd(mm->extract_background());
}

void test_io(int argc, char** argv) {
    /*
        argv[1] - input dir
        argv[2] - output dir
        argv[3] - remained semantic json name
     */
    CHECK(fs::exists(argv[1])) << argv[1] << " does not exist" << endl;
    CHECK(fs::exists(argv[2])) << argv[2] << " does not exist" << endl;
    CHECK(fs::exists(argv[2])) << argv[2] << " does not exist" << endl;

    const string ifn(argv[1]);
    const string ofn(argv[2]);
    const string remain_semantic(argv[3]);

    auto mm = make_unique<MapManager>(ifn);
    mm->update_view();
    mm->show_point_cloud();

    mm->filter_useless_semantics_from_json(remain_semantic);
    mm->dye_through_semantics();
    mm->update_view();
    mm->show_point_cloud();

    mm->export_to_dir(ofn);
}

void test_landmark(int argc, char** argv) {
    /*
        argv[1] - input dir
     */
    CHECK(fs::exists(argv[1])) << argv[1] << " does not exist" << endl;
 
    const string ifn(argv[1]);
    auto mm = make_unique<MapManager>(ifn);
    PointCloud<PointXYZRGB>::Ptr new_pcd{new PointCloud<PointXYZRGB>};
    vector<int> new_landmark_label;
    mm->update_view();
    mm->show_point_cloud();

    mm->figure_out_landmarks_annotation();
    mm->dye_through_landmarks();
    mm->update_view();
    mm->show_point_cloud();

    auto background = mm->extract_background();
    auto landmarks = mm->extract_landmarks();
    // visualize_pcd(landmarks);
    const double r = 2, k = 15, stddev = 1.0;
    const int meanK = 15;

    // auto filtered = filter_outliers_via_stats(landmarks, stddev, meanK, false);
    // auto outliers = filter_outliers_via_stats(landmarks, stddev, meanK, true);
    // visualize_pcd(filtered);
    // visualize_pcd(outliers);

    // auto filtered2 = filter_outliers_via_stats(filtered, stddev, meanK, false);
    // auto outliers2 = filter_outliers_via_stats(filtered, stddev, meanK, true);
    // visualize_pcd(filtered2);
    // visualize_pcd(outliers2);
    auto filtered = filter_outliers_via_radius(landmarks, r, k, false);
    auto outliers = filter_outliers_via_radius(landmarks, r, k, true);
    visualize_pcd(filtered);
    visualize_pcd(outliers);

    auto filtered_landmarks = filter_outliers_via_radius(filtered, r, k, false);
    auto outliers_landmarks = filter_outliers_via_radius(outliers, r, k, true);
    visualize_pcd(filtered_landmarks);
    visualize_pcd(outliers_landmarks);


    auto filtered_background = filter_outliers_via_radius(background, r, k, false);
    auto outliers_background = filter_outliers_via_radius(background, r, k, true);
    new_pcd->points.insert(new_pcd->points.end(), filtered_landmarks->points.begin(),filtered_landmarks->points.end());
    new_pcd->points.insert(new_pcd->points.end(), filtered_background->points.begin(),filtered_background->points.end());
    new_landmark_label.resize(new_pcd->points.size(), LandmarkType::LANDMARK);
    for (int idx = filtered_landmarks->points.size(); idx < new_pcd->points.size(); idx++) {
        new_landmark_label[idx] = LandmarkType::BACKGROUND;
    }
    mm->m_pcd = new_pcd;
    mm->m_render_label = new_landmark_label;
    mm->filter_landmarks_through_background();
    mm->dye_through_render();
    mm->update_view();
    mm->show_point_cloud();
    // auto filtered2 = filter_outliers_via_stats(filtered, stddev, meanK, false);
    // auto outliers2 = filter_outliers_via_stats(filtered, stddev, meanK, true);
    // visualize_pcd(filtered2);
    // visualize_pcd(outliers2);

    // auto filtered2 = filter_outliers_via_radius(filtered, r, k, false);
    // auto outliers2 = filter_outliers_via_radius(filtered, r, k, true);
    // visualize_pcd(filtered2);
    // visualize_pcd(outliers2);

    // auto filtered3 = filter_outliers_via_radius(filtered2, r, k, false);
    // auto outliers3 = filter_outliers_via_radius(filtered2, r, k, true);
    // visualize_pcd(filtered3);
    // visualize_pcd(outliers3);
}

void test_background(int argc, char** argv) {
    /*
        argv[1] - input dir
     */
    CHECK(fs::exists(argv[1])) << argv[1] << " does not exist" << endl;
 
    const string ifn(argv[1]);
    auto mm = make_unique<MapManager>(ifn);
    mm->update_view();
    // mm->show_point_cloud();

    LOG(INFO) << 1 << endl;
    mm->figure_out_landmarks_annotation();
    LOG(INFO) << 1 << endl;
    mm->dye_through_semantics();
    LOG(INFO) << 1 << endl;
    mm->update_view();
    // mm->show_point_cloud();

    LOG(INFO) << 2 << endl;
    auto background = mm->extract_background();
    // visualize_pcd(landmarks);

    auto filtered = filter_outliers_via_radius(background, 2, 5, false);
    auto outliers = filter_outliers_via_radius(background, 2, 5, true);
    visualize_pcd(filtered);
    visualize_pcd(outliers);

    auto removed = mm->extract_removed();
    visualize_pcd(removed);

    auto unknown = mm->extract_unknown();
    visualize_pcd(unknown);
    // auto filtered2 = filter_outliers_via_radius(filtered, 2, 5, false);
    // auto outliers2 = filter_outliers_via_radius(filtered, 2, 5, true);
    // visualize_pcd(filtered2);
    // visualize_pcd(outliers2);

    // auto filtered3 = filter_outliers_via_radius(filtered2, 2, 5, false);
    // auto outliers3 = filter_outliers_via_radius(filtered2, 2, 5, true);
    // visualize_pcd(filtered3);
    // visualize_pcd(outliers3);
}
