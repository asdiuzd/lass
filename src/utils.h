#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <Eigen/Geometry>

#include <glog/logging.h>

#include "Camera.h"
#include "MapManager.h"

using Clock = std::chrono::high_resolution_clock;
#define print_var(x) std::cout << #x << " " << x << std::endl;

namespace lass {

class camera_intrinsics;

typedef struct mhd_structure {
    int ndims, dimsize, offsetx, offsety, offsetz;
    double element_spacing;
    std::string data_file;
} mhd_structure;

// show point cloud
void visualize_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string vn = std::string("simple"));
void visualize_pcd(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud, const std::string vn = std::string("simple"));
void visualize_pcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const std::string vn = std::string("simple"));
void visualize_pcd(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, const std::string vn = std::string("simple"));

void load_and_sample_obj(const std::string& fn, const int sample_number, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcd);
// load camera poses
bool load_info_file(const char *fn, std::vector<Eigen::Matrix4f>& es);

// load image name-index correspondences
bool load_list_file(const char *fn, int n_cameras, std::vector<std::string>& image_fns, std::vector<int>& camera_types, bool camera_type_valid=true);

// load bin xyz
bool load_bin_file(const char *fn, pcl::PointCloud<pcl::PointXYZ>::Ptr& pts);
/*
    load nvm point cloud
    cameras     - camera pose
    points      - 3D points
    measurements    - 2D measurements on each image
    pidx        - the index of 3D points, pidx[i], that 2D measurements, measurements[i], belong to.   
    cidx        - the index of cameras, cidx[i], that 2D measurements, measurements[i], belong to.
 */
bool load_nvm_file(const char *fn, std::vector<CameraF>& cameras, std::vector<Point3DF>& points, std::vector<Point2D>& measurements, std::vector<int>& pidx, std::vector<int>& cidx, std::vector<std::string> &names, std::vector<int>& ptc);

bool load_mhd_file(const char *fn, mhd_structure& data);
// read nvm from file ifn and write pcd to file ofn
bool load_sequences(const char *fn, std::vector<std::string>& seqs);
bool load_7scenes_poses(const std::string base_path, const std::string scene, std::vector<Eigen::Matrix4f>& es, std::vector<std::string>& fns, std::vector<std::string>& relative_fns, bool gettraining= true, bool gettest= true);
bool normalize_7scenes_poses(const std::string base_path, const std::string scene);

bool transfer_nvm_to_pcd(const char *ifn, const char *ofn, const bool visualize=false);
// transfer nvm to pcd
bool transfer_nvm_to_pcd(std::vector<Point3DF>& nvm, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, const bool visualize=false);

bool annotate_point_cloud(const char *annotation_dir, std::vector<std::string>& image_fns, std::vector<Point2D>& measurements, std::vector<int>& pidx, std::vector<int>& cidx, std::vector<int>&  point_semantics);
// get color with given min, max, x
bool filter_useless_semantics(std::vector<int>& original_labels, std::vector<std::string>& original_semantics, std::vector<std::string>& remained_semantics);
bool retrieve_semantic_label_via_color(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcd, int label_numbers, std::vector<int>& labels);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_outliers_via_radius(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, float radius=1, int k=10, bool set_negative=false);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_outliers_via_stats(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, float stddev=1, int meanK=10, bool set_negative=false);

template<typename T>
inline T euclidean_distance(T x, T y, T z) {
    return sqrt(x*x + y*y + z*z);
}

void Dilate(pcl::PointCloud<pcl::PointXYZL>::Ptr &pcd, std::vector<double> &depth, int r, const camera_intrinsics& intrinsics, float scale = 4);
void Erode(pcl::PointCloud<pcl::PointXYZL>::Ptr &pcd, std::vector<double> &depth, int r, const camera_intrinsics& intrinsics, float scale = 4);
void depth_based_DE(pcl::PointCloud<pcl::PointXYZL>::Ptr &pcd, std::vector<double> &depth, const camera_intrinsics& intrinsics, int stride = 7, float scale = 4);
void fillHoles(cv::Mat &img);
void fillHoles_fast(cv::Mat &img);

std::vector<int> grid_segmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, double grid_resolution=5.0);

void GroundColorMix(unsigned char &r, unsigned char &g, unsigned char &b, double x, double min=0, double max=255);
inline double normalize_value(double value, double min, double max) {
    return (value - min) / (max - min);
}

inline void label_to_rgb(uchar &r, uchar &g, uchar &b, uint32_t label) {
    // r = label / (256 * 256);
    // set r to constant value, for better appearance
    r = 200;
    CHECK(label < 256 * 256);
    g = (label / 256) % 256;
    b = label % 256;
}

inline void hash_colormap(uchar &r, uchar &g, uchar &b, size_t label) {
    size_t h = label * 6364136223846793005u + 1442695040888963407;
    r = uchar(h & 0xFF);
    g = uchar((h >> 4) & 0xFF);
    b = uchar((h >> 8) & 0xFF);
}

inline float rand01() {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_real_distribution<float> rand01_d(0, 1);
    return rand01_d(rng);
}

void filter_few_colors(cv::Mat &img, int few_color_threshold = 36);
void add_camera_trajectory_to_viewer(std::shared_ptr<pcl::visualization::PCLVisualizer> viewer, const std::vector<Eigen::Matrix4f> &Twcs, int point_size = 5);

void update_candidate_list(pcl::PointCloud<pcl::PointXYZL>::Ptr& pcd, std::vector<float>& scores, std::vector<pcl::PointXYZL>& centers, int width);
}

#endif
