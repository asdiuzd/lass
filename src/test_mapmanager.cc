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
    mm->update_view();
    mm->show_point_cloud();

    mm->dye_through_landmarks_semantics();
    mm->update_view();
    mm->show_point_cloud();
}

int main(int argc, char** argv) {
    
    // test_io(argc, argv);
    test_landmark(argc, argv);
}