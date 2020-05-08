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
using namespace Eigen;

void load_and_preserve_info_file(const char* fn, vector<Quaternionf>& qs, vector<Vector3f>& ts, vector<vector<float>>& is) {
    ifstream ifs(fn, std::ios::in | std::ios::binary);
    if (!ifs) {
        cout << "Cannot read file" << endl;
        exit(0);
    }

    uint32_t n_cameras;
    ifs.read((char*) &n_cameras, sizeof(u_int32_t));
    cout << "n_cameras = " << n_cameras << endl;
    qs.resize(n_cameras);
    ts.resize(n_cameras);
    is.resize(n_cameras, vector<float>(5));

    for (uint32_t i = 0; i < n_cameras; i++) {
        double focal_length, kappa_1, kappa_2;
        int32_t width, height;
        double *r = new double[9];
        double *t = new double[3];
        Eigen::Matrix3f rotation_matrix;

        ifs.read((char *)&focal_length, sizeof(double));
        ifs.read((char *)&kappa_1, sizeof(double));
        ifs.read((char *)&kappa_2, sizeof(double));
        ifs.read((char *)&width, sizeof(int32_t));
        ifs.read((char *)&height, sizeof(int32_t));
        ifs.read((char *)r, 9 * sizeof(double));
        ifs.read((char *)t, 3 * sizeof(double));

        auto& ii = is[i];
        ii[0] = focal_length;
        ii[1] = kappa_1;
        ii[2] = kappa_2;
        ii[3] = width;
        ii[4] = height;

        rotation_matrix <<    
                r[0], r[1], r[2],
                r[3], r[4], r[5],
                r[6], r[7], r[8];

        auto& qi = qs[i];
        auto& ti = ts[i];
        qi = Quaternionf(rotation_matrix);
        qi.normalize();
        qi.y() = -qi.y();
        qi.z() = -qi.z();

        ti[0] = t[0];
        ti[1] = -t[1];
        ti[2] = -t[2];
    }
}

void load_and_preserve_list_file(const char *fn, int n_cameras, vector<string>& image_fns) {
    cout << "Start load list file" << endl;
    cout << n_cameras << endl;
    cout << fn << endl;

    ifstream ifs(fn);

    if (!ifs) {
        cout << "Cannot read file" << endl;
        exit(0);
    }
    image_fns.resize(n_cameras);
    for (int idx = 0; idx < n_cameras; idx++) {
        // string img_fn;
        ifs >> image_fns[idx];
        // image_fns[idx] = img_fn;
    }
}

void export_to_txt(const char *fn, vector<string>& image_fns, vector<Vector3f>& ts, vector<Quaternionf>& qs, vector<vector<float>>& is) {
    ofstream ofs(fn);

    const int n_cameras = image_fns.size();
    CHECK(ts.size() == n_cameras) << "ts inconsistent" << endl;
    CHECK(qs.size() == n_cameras) << "qs inconsistent" << endl;
    CHECK(is.size() == n_cameras) << "is inconsistent" << endl;
     
    ofs << "image_file_name, x, y, z, qw, qx, qy, qz, focal, k1, k2, width, height" << endl << endl;

    for (int idx = 0; idx < n_cameras; idx++) {
        ofs << image_fns[idx] << ' ';
        ofs << ts[idx][0] << ' ' << ts[idx][1] << ' ' << ts[idx][2] << ' ';
        ofs << qs[idx].w() << ' ' << qs[idx].x() << ' ' << qs[idx].y() << ' ' << qs[idx].z() << ' ';
        for (int i = 0; i < 5; i++) {
            ofs << is[idx][i] << ' ';
        }
        ofs << endl;
    }
}

void test_extrinsics(const string ply_fn, vector<Eigen::Vector3f>& ts, vector<Eigen::Quaternionf>& qs) {
    vector<Eigen::Matrix4f> es;
    PointCloud<PointXYZRGB>::Ptr pcd(new PointCloud<PointXYZRGB>);
    PointXYZRGB min3d, max3d;
    PointXYZ position;

    auto mm = make_unique<MapManager>();
    mm->load_ply_pcl(ply_fn);
    getMinMax3D(*mm->m_pcd, min3d, max3d);
    position.x = (min3d.x + max3d.x) / 2;
    position.y = (min3d.y + max3d.y) / 2;
    position.z = (min3d.z + max3d.z) / 2;

    mm->update_view();
    mm->m_viewer->setCameraPosition(position.x, position.y, position.z, 1, 0, 0);

    Eigen::Vector3f o, d, u;

    pcd->points.resize(ts.size() * 5);
    for (int idx = 0; idx < ts.size(); idx++) {
        Eigen::Matrix4f e;

        e.block<3, 3>(0, 0) = qs[idx].matrix();
        e.block<3, 1>(0, 3) = ts[idx];

        Eigen::Vector3f c_nvm = e.block(0, 3, 3, 1), t_e;
        Eigen::Matrix3f R_nvm = e.block(0, 0, 3, 3), R_e;

        // c_nvm(1) *= -1; 
        // c_nvm(2) *= -1;
        // R_nvm(0, 1) *= -1; 
        // R_nvm(0, 2) *= -1;
        // R_nvm(1, 0) *= -1; 
        // R_nvm(2, 0) *= -1;

        c_nvm = -R_nvm.transpose() * c_nvm;
        d = R_nvm.transpose() * Eigen::Vector3f::UnitZ();

        auto &pt = pcd->points[idx * 5];
        pt.r = pt.g = pt.b = 200;
        pt.x = c_nvm(0);
        pt.y = c_nvm(1);
        pt.z = c_nvm(2);
        for (int s_idx = 1; s_idx < 5; s_idx++) {
            auto &pt = pcd->points[idx * 5 + s_idx];
            pt.g = pt.b = 0;
            pt.r = 200;

            pt.x = c_nvm(0) + d(0) * s_idx * 0.2;
            pt.y = c_nvm(1) + d(1) * s_idx * 0.2;
            pt.z = c_nvm(2) + d(2) * s_idx * 0.2;
        }
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pcd);
    mm->m_viewer->addPointCloud(pcd, rgb, "camera poses");
    mm->m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "camera poses");
    mm->show_point_cloud();
}

void test_transfer_list(int argc, char** argv) {
    const char* info_fn = argv[1];
    const char* list_fn = argv[2];
    const char* txt_fn  = argv[3];
    const char* ply_fn  = argv[4];

    vector<Eigen::Quaternionf>  qs;
    vector<Eigen::Vector3f>     ts;
    vector<vector<float>>     is;
    vector<string>  image_fns;
    vector<int>     camera_types;

    load_and_preserve_info_file(info_fn, qs, ts, is);
    load_and_preserve_list_file(list_fn, qs.size(), image_fns);
    export_to_txt(txt_fn, image_fns, ts, qs, is);
    test_extrinsics(ply_fn, ts, qs);
}


int main(int argc, char** argv) {
    
    test_transfer_list(argc, argv);
}
