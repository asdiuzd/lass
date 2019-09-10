#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <algorithm>

#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <Eigen/Geometry>

#include <glog/logging.h>

#include "Camera.h"


namespace lass {

// show point cloud
void visualize_pcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const std::string vn = std::string("simple"));

// load camera poses
bool load_info_file(const char *fn, std::vector<Eigen::Matrix4f>& es);

// load image name-index correspondences
bool load_list_file(const char *fn, int n_cameras, std::vector<std::string>& image_fns, std::vector<int>& camera_types);

/*
    load nvm point cloud
    cameras     - camera pose
    points      - 3D points
    measurements    - 2D measurements on each image
    pidx        - the index of 3D points, pidx[i], that 2D measurements, measurements[i], belong to.   
    cidx        - the index of cameras, cidx[i], that 2D measurements, measurements[i], belong to.
 */
bool load_nvm_file(const char *fn, std::vector<CameraF>& cameras, std::vector<Point3DF>& points, std::vector<Point2D>& measurements, std::vector<int>& pidx, std::vector<int>& cidx, std::vector<std::string> names, std::vector<int>& ptc);

// read nvm from file ifn and write pcd to file ofn
bool transfer_nvm_to_pcd(const char *ifn, const char *ofn, const bool visualize=false);
// transfer nvm to pcd
bool transfer_nvm_to_pcd(std::vector<Point3DF>& nvm, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, const bool visualize=false);

bool annotate_point_cloud(const char *annotation_dir, std::vector<std::string>& image_fns, std::vector<Point2D>& measurements, std::vector<int>& pidx, std::vector<int>& cidx, std::vector<int>&  point_semantics);
// get color with given min, max, x
bool filter_useless_semantics(std::vector<int>& original_labels, std::vector<std::string>& original_semantics, std::vector<std::string>& remained_semantics);
bool retrieve_semantic_label_via_color(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcd, int label_numbers, std::vector<int>& labels);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_outliers_via_radius(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, float radius=1, int k=10, bool set_negative=false);

void GroundColorMix(unsigned char &r, unsigned char &g, unsigned char &b, double x, double min=0, double max=255);
inline double normalize_value(double value, double min, double max) {
    return (value - min) / (max - min);
}
}

#endif