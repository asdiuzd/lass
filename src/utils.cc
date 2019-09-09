#include<experimental/filesystem>
#include<opencv2/opencv.hpp>
#include<omp.h>

#include "utils.h"

using namespace std;
using namespace pcl;
using namespace cv;
namespace fs = std::experimental::filesystem;

namespace lass {

void visualize_pcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    pcl::visualization::CloudViewer viewer("simple");
    viewer.showCloud(cloud);
    while(!viewer.wasStopped()){}
}

bool load_info_file(const char *fn, vector<Eigen::Matrix4f>& extrinsics) {
    ifstream ifs(fn, std::ios::in | std::ios::binary);
    if (!ifs) {
        cout << "Cannot read file" << endl;
        exit(0);
    }

    uint32_t n_cameras;
    ifs.read((char*) &n_cameras, sizeof(u_int32_t));
    cout << "n_cameras = " << n_cameras << endl;
    extrinsics.resize(n_cameras);

    for (uint32_t i = 0; i < n_cameras; i++) {
        double focal_length, kappa_1, kappa_2;
        int32_t width, height;
        double *r = new double[9];
        double *t = new double[3];
        auto &e = extrinsics[i];

        ifs.read((char *)&focal_length, sizeof(double));
        ifs.read((char *)&kappa_1, sizeof(double));
        ifs.read((char *)&kappa_2, sizeof(double));
        ifs.read((char *)&width, sizeof(int32_t));
        ifs.read((char *)&height, sizeof(int32_t));
        ifs.read((char *)r, 9 * sizeof(double));
        ifs.read((char *)t, 3 * sizeof(double));

        e <<    r[0], r[1], r[2], t[0],
                r[3], r[4], r[5], t[1],
                r[6], r[7], r[8], t[2],
                0, 0, 0, 1;
    }
    return true;
}

bool load_list_file(const char *fn, int n_cameras, vector<string>& image_fns, vector<int>& camera_types) {
    cout << "Start load list file" << endl;
    cout << n_cameras << endl;
    cout << fn << endl;
    ifstream ifs(fn);

    if (!ifs) {
        cout << "Cannot read file" << endl;
        exit(0);
    }
    image_fns.resize(n_cameras);
    camera_types.resize(n_cameras);
    for (int idx = 0; idx < n_cameras; idx++) {
        string img_fn;
        ifs >> img_fn;
        fs::path p(img_fn);
        if (img_fn.find("left") != string::npos) {
            image_fns[idx] = string("left/") + fs::path(img_fn).filename().string();
            camera_types[idx] = 0;
        } else if (img_fn.find("rear") != string::npos) {
            image_fns[idx] = string("rear/") + fs::path(img_fn).filename().string();
            camera_types[idx] = 1;
        } else if (img_fn.find("right") != string::npos) {
            image_fns[idx] = string("right/") + fs::path(img_fn).filename().string();
            camera_types[idx] = 2;
        } else {
            cout << "Error !" << endl;
            exit(0);
        }
    }
}


bool load_nvm_file(const char *fn, std::vector<CameraF>& cameras, std::vector<Point3DF>& points, std::vector<Point2D>& measurements, std::vector<int>& pidx,
    std::vector<int>& cidx, std::vector<std::string> names, std::vector<int>& ptc
) {
    ifstream in(fn);
    int rotation_parameter_num = 4;
    bool format_r9t = false;
    std::string token;

    if (in.peek() == 'N') {
        in >> token;
        if (strstr(token.c_str(), "R9T")) {
            rotation_parameter_num = 9;
            format_r9t = true;
        }
    }

    int ncam = 0, npoint = 0, nproj = 0;
    in >> ncam;
    if (ncam <= 1) return false;

    cameras.resize(ncam);
    names.resize(ncam);
    for (int i = 0; i < ncam; i++) {
        double f, q[9], c[3], d[2];
        in >> token >> f;
        cameras[i].SetFocalLength(f);

        for (int j = 0; j < rotation_parameter_num; j++) {
            in >> q[j];
        }
        in >> c[0] >> c[1] >> c[2] >> d[0] >> d[1];
        if (format_r9t) {
            cameras[i].SetMatrixRotation(q);
            cameras[i].SetTranslation(c);
        } else {
            cameras[i].SetQuaternionRotation(q);
            cameras[i].SetCameraCenterAfterRotation(c);
        }
        cameras[i].SetNormalizedMeasurementDistortion(d[0]);
        names[i] = token;
    }
    
    cout << "cameras loaded" << std::endl;
    cout << "start to load points..." << std::endl;
    in >> npoint;
    if (npoint <= 0) return false;

    points.resize(npoint);
    for (int i = 0; i < npoint; i++) {
        float pt[3];
        int cc[3], npj;
        in  >> pt[0] >> pt[1] >> pt[2]
            >> cc[0] >> cc[1] >> cc[2] >> npj;
        
        for (int j = 0; j < npj; j++) {
            int cid, fid;
            float imx, imy;
            in >> cid >> fid >> imx >> imy;
            cidx.push_back(cid);
            pidx.push_back(i);
            measurements.push_back(Point2D(imx, imy));
            nproj++;
        }
        points[i].SetPoint(pt);
        ptc.insert(ptc.end(), cc, cc + 3); // what's cc?
    }

    cout << ncam << " cameras; " << npoint << " 3D points; " << nproj << " projections\n";

    return true;
}

bool transfer_nvm_to_pcd(const char *ifn, const char *ofn, const bool visualize) {
    std::vector<CameraF> cameras;
    std::vector<Point3DF> points;
    std::vector<Point2D> measurements;
    std::vector<int> pidx;
    std::vector<int> cidx;
    std::vector<std::string> names;
    std::vector<int> ptc;

    cout << "load nvm..." << endl;
    if(lass::load_nvm_file(ifn, cameras, points, measurements, pidx, cidx, names, ptc)) {
        cout << "Success!" << endl;
        PointCloud<pcl::PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>());
        cout << "transfering nvm to cloud..." << endl;
        if (lass::transfer_nvm_to_pcd(points, cloud, visualize)) {
            std::cout << "Success!" << endl;
            cloud->height = 1;
            cloud->width = cloud->points.size();
            pcl::PCDWriter writer;
            writer.writeBinaryCompressed<pcl::PointXYZRGB> (ofn, *cloud);
        } else {
            cout << "Failed to transfer" << endl;
            return false;
        }
    } else {
        cout << "Failed to load file: " << ifn << endl;
        return false;
    }
    return true;
}

bool transfer_nvm_to_pcd(std::vector<Point3DF>& nvm, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, bool visualize) {
    float maxz = -10000, minz = 10000;
    for (int i = 0; i < nvm.size(); i++) {
        pcl::PointXYZRGB point;
        point.x = nvm[i].xyz[0];
        point.y = nvm[i].xyz[1];
        point.z = nvm[i].xyz[2];
        point.r = 255;
        point.g = 255;
        point.b = 255;
        pcd->points.push_back(point);

        maxz = std::max(maxz, point.z);
        minz = std::min(minz, point.z);
    }
    if (!visualize) return true;

    for (int i = 0; i < nvm.size(); i++) {
        lass::GroundColorMix(pcd->points[i].r, pcd->points[i].g, pcd->points[i].b, pcd->points[i].z / (maxz - minz), 0, 255);
    }

    visualize_pcd(pcd);
    return true;
}

void GroundColorMix(unsigned char &r, unsigned char &g, unsigned char &b, double x, double min, double max)
{
	x = x * 360.0;
	double posSlope = (max - min) / 60;
	double negSlope = (min - max) / 60;
	if (x < 60)
	{
		r = (unsigned char)max;
		g = (unsigned char)(posSlope * x + min);
		b = (unsigned char)min;
		return;
	}
	else if (x < 120)
	{
		r = (unsigned char)(negSlope * x + 2 * max + min);
		g = (unsigned char)max;
		b = (unsigned char)min;
		return;
	}
	else if (x < 180)
	{
		r = (unsigned char)min;
		g = (unsigned char)max;
		b = (unsigned char)(posSlope * x - 2 * max + min);
		return;
	}
	else if (x < 240)
	{
		r = (unsigned char)min;
		g = (unsigned char)(negSlope * x + 4 * max + min);
		b = (unsigned char)max;
		return;
	}
	else if (x < 300)
	{
		r = (unsigned char)(posSlope * x - 4 * max + min);
		g = (unsigned char)min;
		b = (unsigned char)max;
		return;
	}
	else
	{
		r = (unsigned char)max;
		g = (unsigned char)min;
		b = (unsigned char)(negSlope * x + 6 * max);
		return;
	}
}

// double normalize_value(double value, double min, double max) {
// }

bool annotate_point_cloud(const char *annotation_dir, std::vector<std::string>& image_fns, std::vector<Point2D>& measurements, std::vector<int>& pidx, std::vector<int>& cidx, std::vector<int>&  point_semantics) {
    LOG(INFO) << "start annotate point cloud with semantic labels..." << endl;
    CHECK(measurements.size() == pidx.size()) << "the size of measurements is not equal to the size of pidx" << endl;
    CHECK(measurements.size() == cidx.size()) << "the size of measurements is not equal to the size of cidx" << endl;
    CHECK(point_semantics.size() != 0) << "point_semantics has not been initialized" << endl;

    int counter = 0;
    // int iteration_number = 100000;
    int iteration_number = measurements.size();
#pragma omp parallel for
    for (int idx = 0; idx < iteration_number; idx++) {
        // auto& img = images[cidx[idx]];
        counter++;
        LOG_IF(INFO, counter % 1000 == 0) << "processing " << counter << "/" << measurements.size() << " measurement..." << endl;
        Mat img = imread(image_fns[cidx[idx]], 0);
        auto& point = point_semantics[pidx[idx]];
        auto& proj = measurements[idx];
        int x = static_cast<int>(proj.x), y = static_cast<int>(proj.y);

        point = img.at<unsigned char>(y, x);
        // point = img.data[x * 1024 + y];
    }

    return true;
}

bool filter_useless_semantics(vector<int>& original_labels, vector<string>& original_semantics, vector<string>& remained_semantics) {
    vector<int> label_mapping, label_counter;
    label_mapping.resize(original_semantics.size(), -1);
    label_counter.resize(remained_semantics.size(), 0);

    for (int o_idx = 0; o_idx < original_semantics.size(); o_idx++) {
        for (int r_idx = 0; r_idx < remained_semantics.size(); r_idx++) {
            if (original_semantics[o_idx] == remained_semantics[r_idx]) {
                label_mapping[o_idx] = r_idx;
                break;
            }
        }
    }

    for (auto& label: original_labels) {
        label = label_mapping[label];
        if (label != -1) {
            label_counter[label]++;
        }
    }

    LOG(INFO) << "original semantic numbers = " << original_semantics.size() << endl;
    LOG(INFO) << "remained semantic numbers = " << remained_semantics.size() << endl;

    LOG(INFO) << "label numbers:" << endl;
    for (int idx = 0; idx < label_counter.size(); idx++) {
        LOG(INFO) << remained_semantics[idx] << ":\t" << label_counter[idx] << endl;
    }
    LOG(INFO) << "Finished" << endl;
}

bool retrieve_semantic_label_via_color(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcd, int label_numbers, std::vector<int>& labels) {
    vector<vector<unsigned char>> color_mapping;
    color_mapping.resize(label_numbers);

    labels.resize(pcd->points.size(), -1);

    for (int i = 0; i < label_numbers; i++) {
        const double value = normalize_value(i, 0, label_numbers);
        auto& color = color_mapping[i];
        color.resize(3);

        GroundColorMix(color[0], color[1], color[2], value, 0, 255);
    }

    for (int idx = 0; idx < pcd->points.size(); idx++) {
        auto& point = pcd->points[idx];
        auto& label = labels[idx];

        for (int i = 0; i < label_numbers; i++) {
            auto& color = color_mapping[i];

            if (point.r == color[0] && point.g == color[1] && point.b == color[2]) {
                label = i;
                break;
            }
        }
    }
}

}