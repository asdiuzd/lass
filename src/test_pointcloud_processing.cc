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

typedef pcl::PointXYZRGB PointType;
using namespace std;
using namespace pcl;
using namespace pcl::visualization;
using namespace lass;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

void test_nvm_to_pcd(int argc, char** argv);
void test_semantic_segmentation(int argc, char** argv);
void test_view_segmentation(int argc, char **argv);

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    // test_nvm_to_pcd(argc, argv);
    // test_semantic_segmentation(argc, argv);
    test_view_segmentation(argc, argv);
}

void test_nvm_to_pcd(int argc, char** argv) {
    /*
        argv[1] - nvm file name
        argv[2] - pcd file name
     */

    transfer_nvm_to_pcd(argv[1], argv[2], true);
}

void test_semantic_segmentation(int argc, char** argv) {
    /*
        argv[1] - nvm file name
        argv[2] - list file name
        argv[3] - semantic annotation dir
        argv[4] - pcd file name (to be saved)
     */
    CHECK(fs::exists(argv[1])) << argv[1] << " does not exist" << endl;
    CHECK(fs::exists(argv[2])) << argv[2] << " does not exist" << endl;
    CHECK(fs::exists(argv[3])) << argv[3] << " does not exist" << endl;
    CHECK(fs::exists(argv[4])) << argv[4] << " does not exist" << endl;

    string nvm_fn(argv[1]), list_fn(argv[2]), annotation_dir(argv[3]), pcd_fn(argv[4]);

    std::vector<CameraF> cameras;
    std::vector<Point3DF> points;
    std::vector<Point2D> measurements;
    std::vector<int> pidx;
    std::vector<int> cidx;
    std::vector<std::string> names;
    std::vector<int> ptc;
    std::vector<int> points_semantics;

    std::vector<string> image_fns;
    vector<int> camera_types;

    PointCloud<pcl::PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>());

    fs::path p(annotation_dir);

    load_nvm_file(nvm_fn.c_str(), cameras, points, measurements, pidx, cidx, names, ptc);
    load_list_file(list_fn.c_str(), cameras.size(), image_fns, camera_types);

    for (auto& fn: image_fns) {
        fn = (p / fn).string();
    }
    cout << "image filenames example: " << image_fns[0] << endl;

    transfer_nvm_to_pcd(points, cloud, false);

    points_semantics.resize(points.size(), -1);
    annotate_point_cloud(annotation_dir.c_str(), image_fns, measurements, pidx, cidx, points_semantics);

    int max_s = *max_element(points_semantics.begin(), points_semantics.end());
    int min_s = *min_element(points_semantics.begin(), points_semantics.end());

    LOG(INFO) << "max semantic value: " << max_s << endl;
    LOG(INFO) << "min semantic value: " << min_s << endl;

    ofstream of("parameters.txt");

    of << points_semantics.size() << endl;
    for (int i = 0; i < points.size(); i++) {
        auto& point = cloud->points[i];
        of << points_semantics[i] << endl;
        if (points_semantics[i] == -1) {
            point.r = point.g = point.b = 10;
        } else {
            lass::GroundColorMix(point.r, point.g, point.b, normalize_value(points_semantics[i], 0, max_s), 0, 255);
        }
    }


    pcl::PCDWriter writer;
    writer.writeBinaryCompressed<pcl::PointXYZRGB> (pcd_fn.c_str(), *cloud);
    visualize_pcd(cloud);
}

void test_view_segmentation(int argc, char **argv) {
    /*
        argv[1] -- input point cloud .pcd
        argv[2] -- original semantic names json file
        argv[3] -- remained semantic names json file
        argv[4] -- parameters file
     */
    CHECK(fs::exists(argv[1])) << argv[1] << " does not exist" << endl;
    CHECK(fs::exists(argv[2])) << argv[2] << " does not exist" << endl;
    CHECK(fs::exists(argv[3])) << argv[3] << " does not exist" << endl;
    CHECK(fs::exists(argv[4])) << argv[4] << " does not exist" << endl;

    LOG(INFO) << "read pcd" << endl;   
    PCDReader reader;
    PointCloud<PointXYZRGB>::Ptr pcd(new PointCloud<PointXYZRGB>());
    vector<int> semantic_labels;
    reader.read<PointXYZRGB>(argv[1], *pcd);

    LOG(INFO) << "read json" << endl;   
    json j_original_semantic_names, j_remained_semantic_names;
    ifstream o_if(argv[2]), r_if(argv[3]);
    vector<string> original_names, remained_names;
    o_if >> j_original_semantic_names;
    r_if >> j_remained_semantic_names;

    original_names.resize(j_original_semantic_names.size());
    remained_names.resize(j_remained_semantic_names.size());

    for (int idx = 0; idx < original_names.size(); idx++) {
        original_names[idx] = j_original_semantic_names[idx].get<string>();
    }
    for (int idx = 0; idx < remained_names.size(); idx++) {
        remained_names[idx] = j_remained_semantic_names[idx].get<string>();
    }

    ifstream p_if(argv[4]);
    while(!p_if.eof()) {
        int label;
        p_if >> label;
        semantic_labels.push_back(label);
    }
    // retrieve_semantic_label_via_color(pcd, original_names.size(), semantic_labels);
    filter_useless_semantics(semantic_labels, original_names, remained_names);

    int label_number = remained_names.size();
    for (int idx = 0; idx < pcd->points.size(); idx++) {
        auto& point = pcd->points[idx];
        if (semantic_labels[idx] == -1) {
            point.r = point.g = point.b = 20;
        } else {
            GroundColorMix(point.r, point.g, point.b, normalize_value(semantic_labels[idx], 0, label_number), 0, 255);
        }
    }

    ofstream p_of("p.txt");
    p_of << semantic_labels.size() << endl;    
    for (auto& label:semantic_labels) {
        p_of << label << endl;
    }

    LOG(INFO) << "start visualization" << endl;
    visualize_pcd(pcd);
}
