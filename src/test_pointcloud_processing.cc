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

typedef pcl::PointXYZRGB PointType;
using namespace std;
using namespace pcl;
using namespace pcl::visualization;
using namespace lass;
namespace fs = std::experimental::filesystem;

void test_nvm_to_pcd(int argc, char** argv);
void test_semantic_segmentation(int argc, char** argv);

int main(int argc, char** argv) {
    // test_nvm_to_pcd(argc, argv);
    test_semantic_segmentation(argc, argv);
}

void test_nvm_to_pcd(int argc, char** argv) {
    /*
        argv[1] - nvm file name
        argv[2] - pcd file name
     */
    // CHECK(argc < 3) << "./test nvm_file_name saved_pcd_file_name" << endl;

    transfer_nvm_to_pcd(argv[1], argv[2], true);
}

void test_semantic_segmentation(int argc, char** argv) {
    /*
        argv[1] - nvm file name
        argv[2] - list file name
        argv[3] - semantic annotation dir
        argv[4] - pcd file name (to be saved)
     */
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

    for (int i = 0; i < points.size(); i++) {
        auto& point = cloud->points[i];
        if (points_semantics[i] == -1) {
            point.r = point.g = point.b = 10;
        } else {
            lass::GroundColorMix(point.r, point.g, point.b, normalize_value(points_semantics[i], min_s, max_s));
        }
    }

    pcl::PCDWriter writer;
    writer.writeBinaryCompressed<pcl::PointXYZRGB> (pcd_fn.c_str(), *cloud);
    visualize_pcd(cloud);
}