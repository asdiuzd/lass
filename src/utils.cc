#include <pcl/filters/statistical_outlier_removal.h>
#include<experimental/filesystem>
#include<opencv2/opencv.hpp>
#include<omp.h>
#include<Eigen/Eigen>
#include <pcl/io/ply_io.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>

#include "utils.h"
#include "mesh_sampling.h"

using namespace std;
using namespace pcl;
using namespace cv;
namespace fs = std::experimental::filesystem;

namespace lass {

void visualize_pcd(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, const string vn) {
    pcl::visualization::CloudViewer viewer(vn.c_str());
    viewer.showCloud(cloud);
    while(!viewer.wasStopped()){}
}

void visualize_pcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const string vn) {
    pcl::visualization::CloudViewer viewer(vn.c_str());
    viewer.showCloud(cloud);
    while(!viewer.wasStopped()){}
}

void visualize_pcd(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud, const string vn) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer);                                // viewer
    // pcl::visualization::CloudViewer viewer(vn.c_str());
    viewer->addPointCloud(cloud, "target_pcd");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "target_pcd");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_pcd");
    while(!viewer->wasStopped()){
        viewer->spinOnce(100);
    }
}

void visualize_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const string vn) {
    pcl::visualization::CloudViewer viewer(vn.c_str());
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
        Eigen::Quaternionf q(e.block<3, 3>(0, 0));
        q.normalize();
        e.block<3, 3>(0, 0) = q.matrix();
    }
    return true;
}

bool load_list_file(const char *fn, int n_cameras, vector<string>& image_fns, vector<int>& camera_types, bool camera_type_valid) {
    cout << "Start load list file" << endl;
    cout << n_cameras << endl;
    cout << fn << endl;

    if (camera_type_valid) {
        LOG(INFO) << "camera type enabled" << endl;
    } else {
        LOG(INFO) << "camera type disabled" << endl;
    }

    ifstream ifs(fn);

    if (!ifs) {
        cout << "Cannot read file" << endl;
        exit(0);
    }
    image_fns.resize(n_cameras);
    if (camera_type_valid) {
        camera_types.resize(n_cameras);
    }
    for (int idx = 0; idx < n_cameras; idx++) {
        string img_fn;
        ifs >> img_fn;
        fs::path p(img_fn);
        if (camera_type_valid) {
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
        } else {
            image_fns[idx] = img_fn;
        }
    }
}

bool load_bin_file(const char *fn, pcl::PointCloud<PointXYZ>::Ptr& pts) {
    FILE* fp = fopen(fn, "rb");
    char buffer[8];
    int width, height;
    int ret = fread(&width, 4, 1, fp);
    ret = fread(&height, 4, 1, fp);
    CHECK(width == 640) << width << endl;
    CHECK(height == 480) << height << endl;

    pts->points.resize(width * height);
    for (auto& pt : pts->points) {
        ret = fread(&pt.x, 4, 1, fp);
        ret = fread(&pt.y, 4, 1, fp);
        ret = fread(&pt.z, 4, 1, fp);
    }
}

bool load_nvm_file(const char *fn, std::vector<CameraF>& cameras, std::vector<Point3DF>& points, std::vector<Point2D>& measurements, std::vector<int>& pidx,
    std::vector<int>& cidx, std::vector<std::string> &names, std::vector<int>& ptc
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

void load_and_sample_obj(const std::string& fn, const int sample_number, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcd) {
    PolygonMesh mesh;
    CHECK(io::loadPolygonFileOBJ(fn.c_str(), mesh) >= 0) << "can not load: " << fn << endl;

    vtkSmartPointer<vtkPolyData> vtkmesh;
    VTKUtils::convertToVTK(mesh, vtkmesh);
    PointCloud<PointXYZRGBNormal>::Ptr cloud(new PointCloud<PointXYZRGBNormal>);
    uniform_sampling(vtkmesh, sample_number, true, true, *cloud);

    pcd->points.resize(sample_number);
    for (auto idx = 0; idx < sample_number; idx++) {
        auto& pt1 = pcd->points[idx];
        auto& pt2 = cloud->points[idx];
        pt1.x = pt2.x;
        pt1.y = pt2.y;
        pt1.z = pt2.z;
        pt1.r = pt1.g = pt1.b = 220;
        // pt1.r = pt2.r;
        // pt1.g = pt2.g;
        // pt1.b = pt2.b;
    }

    PointXYZRGB minpt, maxpt;
    getMinMax3D(*pcd,minpt, maxpt);
    LOG(INFO) << "min pt = " << minpt << endl;
    LOG(INFO) << "max pt = " << maxpt << endl;
}

bool load_sequences(const char *fn, vector<string>& seqs) {
    ifstream in(fn);
    CHECK(fs::exists(fn)) << fn << " not exists" << endl;

    while (!in.eof()) {
        string seq;
        in >> seq;
        if (seq == "") {
            continue;
        }
        seqs.emplace_back(seq);
    }
}

bool load_7scenes_poses(const string base_path, const string scene, std::vector<Eigen::Matrix4f>& es, vector<string>& fns, std::vector<std::string>& relative_fns, bool gettraining, bool gettest) {
    fs::path base_path_str{base_path};
    string scene_str(scene), trainingset("TrainSplit.txt"), testset("TestSplit.txt");
    int frame_number;
    if (scene.compare("stairs") == 0) {
        frame_number = 500;
    } else {
        frame_number = 1000;
    }

    string training_fn = (base_path_str / scene_str / trainingset).string();
    string test_fn = (base_path_str / scene_str / testset).string();
    vector<string> seqs;


    if (gettraining) {
        load_sequences(training_fn.c_str(), seqs);
    }
    if (gettest) {
        load_sequences(test_fn.c_str(), seqs);
    }

    /*
        The matrix stored in file is camera-to-world.
        We need world-to-camera.
    */
    es.resize(0);
    fns.resize(0);
    relative_fns.resize(0);
    stringstream s;
    s.fill('0');
    fns.resize(0);
    for (auto& seq : seqs) {
        for (int idx = 0; idx < frame_number; idx++) {
            s.str("");
            s << "frame-" << setw(6) << idx << ".pose.txt";
            auto fn = (base_path_str / scene_str / seq / s.str()).string();
            ifstream in(fn);
            fns.emplace_back(fn);
            relative_fns.emplace_back(fs::path(seq) / s.str());

            Eigen::Matrix4f e;
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    in >> e(r, c);
                }
            }
            // Eigen::Quaternionf q(e.block<3, 3>(0, 0));
            // q.normalize();
            // e.block(0, 0, 3, 3) = q.conjugate().toRotationMatrix();
            // e.block(0, 3, 3, 1) = - (q.conjugate() * e.block(0, 3, 3, 1));
            e.block(0, 0, 3, 3) = e.block(0, 0, 3, 3).inverse();
            // e.block(0, 0, 3, 3) = e.block(0, 0, 3, 3).transpose();
            e.block(0, 3, 3, 1) = - (e.block(0, 0, 3, 3) * e.block(0, 3, 3, 1));
            es.emplace_back(e);
        }
    }
}

bool normalize_7scenes_poses(const string base_path, const string scene) {
    fs::path base_path_str{base_path};
    string scene_str(scene), trainingset("TrainSplit.txt"), testset("TestSplit.txt");
    int frame_number;
    if (scene.compare("stairs") == 0) {
        frame_number = 500;
    } else {
        frame_number = 1000;
    }

    string training_fn = (base_path_str / scene_str / trainingset).string();
    string test_fn = (base_path_str / scene_str / testset).string();
    vector<string> seqs;

    load_sequences(training_fn.c_str(), seqs);
    load_sequences(test_fn.c_str(), seqs);

    /*
        The matrix stored in file is camera-to-world.
        We need world-to-camera.
    */
    stringstream s, target_s;
    s.fill('0');
    target_s.fill('0');
    for (auto& seq : seqs) {
        for (int idx = 0; idx < frame_number; idx++) {
            s.str("");
            target_s.str("");
            s << "frame-" << setw(6) << idx << ".pose.txt";
            target_s << "frame-" << setw(6) << idx << ".normpose.txt";
            auto fn = (base_path_str / scene_str / seq / s.str()).string();
            auto target_fn = (base_path_str / scene_str / seq / target_s.str()).string();
            ifstream in(fn);
            ofstream target_on(target_fn);
            target_on << setiosflags(ios::scientific) <<setprecision(8);

            Eigen::Matrix4f e;
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    in >> e(r, c);
                }
            }
            Eigen::Quaternionf q(e.block<3, 3>(0, 0));
            q.normalize();
            e.block(0, 0, 3, 3) = q.toRotationMatrix();

            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    target_on << e(r, c) << "\t";
                }
                target_on << endl;
            }
        }
    }
}

bool load_mhd_file(const char *fn, mhd_structure& data) {
    ifstream in(fn);
    string token, equal;

    in >> token >> equal;
    CHECK(token.compare("NDims") == 0) << "NDims failed to read" << endl;
    CHECK(equal.compare("=") == 0) << "not equal" << endl;
    in >> data.ndims;

    in >> token >> equal;
    CHECK(token.compare("DimSize") == 0) << "DimSize failed to read" << endl;
    CHECK(equal.compare("=") == 0) << "not equal" << endl;
    in >> data.dimsize >> data.dimsize >> data.dimsize;

    in >> token >> equal;
    CHECK(token.compare("Offset") == 0) << "Offset failed to read" << endl;
    CHECK(equal.compare("=") == 0) << "not equal" << endl;
    in >> data.offsetx >> data.offsety >> data.offsetz;

    in >> token >> equal;
    CHECK(token.compare("ElementSpacing") == 0) << "ElementSpacing failed to read" << endl;
    CHECK(equal.compare("=") == 0) << "not equal" << endl;
    in >> data.element_spacing >> data.element_spacing >> data.element_spacing;

    in >> token >> equal;
    CHECK(token.compare("ElementType") == 0) << "ElementType failed to read" << endl;
    CHECK(equal.compare("=") == 0) << "not equal" << endl;
    in >> token;
    CHECK(token.compare("MET_SHORT") == 0) << "MET_SHORT failed to read" << endl;

    in >> token >> equal;
    CHECK(token.compare("ElementDataFile") == 0) << "ElementDataFile failed to read" << endl;
    CHECK(equal.compare("=") == 0) << "not equal" << endl;
    in >> data.data_file;
}

void Erode(PointCloud<PointXYZL>::Ptr &pcd, vector<double> &depth, int r, const camera_intrinsics& intrinsics, float scale) {
    int width = intrinsics.width / scale, height = intrinsics.height / scale;
    const double depth_min = 0, depth_max = 99999;
    PointCloud<PointXYZL>::Ptr m_pcd(new PointCloud<PointXYZL>);

    // cout << "erode: " << width << ", " << height << endl;

    m_pcd->points.resize(width * height);
    vector<double> m_depth(width * height, depth_max);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            int loc = -1;
            double max_d = 0;
            bool update = true;
            for (int o1 = -r; o1 <= r; ++o1) {
                for (int o2 = -r; o2 <= r; ++o2) {
                    int u = i + o1;
                    int v = j + o2;
                    if (o1 * o1 + o2 * o2 > r * r) {
                        continue;
                    }
                    if (u >= 0 && v >= 0 && u < width && v < height) {
                        if (depth[v * width + u] > max_d) {
                            max_d = depth[v * width + u];
                            loc = v * width + u;
                        }
                    }
                }
            }
            // cout << min_d << endl;
            auto center = j * width + i;
            auto &pt = m_pcd->points[center];
            pt.x = pcd->points[center].x;
            pt.y = pcd->points[center].y;
            pt.z = pcd->points[center].z;

            if (max_d > 0) {
                if (update == true) {
                    // cout << "updadte" << endl;
                    pt.label = pcd->points[loc].label;
                    m_depth[center] = depth[loc];
                } else {
                    pt.label = pcd->points[center].label;
                    m_depth[center] = depth[center];
                }
            } else {
                pt.label = 0; // 0 means none
            }
        }
    }
    depth = m_depth;
    pcd = m_pcd;
}

void Dilate(PointCloud<PointXYZL>::Ptr &pcd, vector<double> &depth, int r, const camera_intrinsics& intrinsics, float scale) {
    int width = intrinsics.width / scale, height = intrinsics.height / scale;
    const double depth_max = 999999;
    PointCloud<PointXYZL>::Ptr m_pcd(new PointCloud<PointXYZL>);
    m_pcd->points.resize(width * height);
    vector<double> m_depth(width * height, depth_max);

    // cout << "dilate: " << width << ", " << height << endl;

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            int loc = -1;
            double min_d = depth_max;
            bool update = false;
            for (int o1 = -r; o1 <= r; ++o1) {
                for (int o2 = -r; o2 <= r; ++o2) {
                    int u = i + o1;
                    int v = j + o2;
                    if (o1 * o1 + o2 * o2 > r * r) {
                        continue;
                    }
                    if (u >= 0 && v >= 0 && u < width && v < height) {
                        if (depth[v * width + u] < min_d) {
                            min_d = depth[v * width + u];
                            loc = v * width + u;
                            update = true;
                        }
                    }
                }
            }
            // cout << min_d << endl;
            auto center = j * width + i;
            auto &pt = m_pcd->points[center];
            pt.x = pcd->points[center].x;
            pt.y = pcd->points[center].y;
            pt.z = pcd->points[center].z;
            if (min_d > 0 && min_d < depth_max) {
                if (update == true) {
                    // cout << "updadte" << endl;
                    pt.label = pcd->points[loc].label;
                    m_depth[center] = depth[loc];
                }
                else {
                    pt.label = pcd->points[center].label;
                    m_depth[center] = depth[center];
                }
            } else {
                pt.label = 0;
            }
        }
    }
    depth = m_depth;
    pcd = m_pcd;
}

void depth_based_DE(PointCloud<PointXYZL>::Ptr &pcd, vector<double> &depth, const camera_intrinsics& intrinsics, int stride, float scale) {
    map<int, int> m_map;
    const int pixel_number = depth.size();
    const int threshold =  pixel_number * 0.0001;
    // LOG(INFO) << "threshold = " << threshold << endl;
    // LOG(INFO) << "pixel number = " << pixel_number << endl;

    for (int i = 0; i < pixel_number; i++) {
        auto& label = pcd->points[i].label;
        if (label == 0) {
            depth[i] = 999999;
        } 
        m_map[label]++;
    }
    for (int i = 0; i < pixel_number; ++i) {
        auto& label = pcd->points[i].label;
        if (m_map[label] <= threshold) {
            label = 0;
            depth[i] = 999999;
        }
    }
    Dilate(pcd, depth, stride, intrinsics, scale);
    Erode(pcd, depth, stride, intrinsics, scale);
    // m_map.clear();
    // for (int i = 0; i < pixel_number; ++i) {
    //     auto& label = pcd->points[i].label;
    //     if (label == 0) {
    //         depth[i] = 99999;
    //     }
    //     m_map[label]++;
    // }
    // for (int i = 0; i < depth.size(); ++i) {
    //     auto& label = pcd->points[i].label;
    //     if (m_map[label] <= threshold) {
    //         label = 0;
    //         depth[i] = 99999;
    //     }
    // }
}

void fillHoles_fast(cv::Mat &img) {
    cv::Mat img_filled = img.clone();
    const int min_conscultive_angle_thresh = 280;
    const int distance_thresh = 25;
    const int distance_thresh_2 = distance_thresh * distance_thresh;
    const int bin_num = 72;
    const int bin_angle = 360 / bin_num;
    assert(std::abs((360.0 / bin_num) - bin_angle) < 1.0e-5);

    std::vector<int> bins(bin_num);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            if (img.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {
                const int i_lower = std::max(0, i - distance_thresh);
                const int i_upper = std::min(img.rows - 1, i + distance_thresh);
                const int j_lower = std::max(0, j - distance_thresh);
                const int j_upper = std::min(img.cols - 1, j + distance_thresh);

                bins.assign(bins.size(), std::numeric_limits<int>::max());
                int min_distance_2 = std::numeric_limits<int>::max();
                std::pair<int, int> closest_pos;
                for (int ii = i_lower; ii < i_upper; ++ii) {
                    for (int jj = j_lower; jj < j_upper; ++jj) {
                        if (img.at<cv::Vec3b>(ii, jj) == cv::Vec3b(0, 0, 0)) continue;
                        int distance_2 = (ii - i) * (ii - i) + (jj - j) * (jj - j);
                        if (distance_2 > distance_thresh_2) continue;
                        int y = ii - i, x = jj - j;
                        double theta = (atan2(y, x) + M_PI) * 180 / M_PI;
                        int bin_idx = theta / bin_angle;
                        bin_idx = std::min(bin_idx, int(bins.size() - 1));
                        bins[bin_idx] = std::min(bins[bin_idx], distance_2);
                        if (distance_2 < min_distance_2) {
                            min_distance_2 = distance_2;
                            closest_pos = {ii, jj};
                        }
                    }
                }
                int cnt = 0;
                int max_cnt = 0;
                for (int i = 0; i < bin_num * 2; ++i) {
                    if (bins[i % bin_num] != std::numeric_limits<int>::max()) {
                        ++cnt;
                        max_cnt = std::max(cnt, max_cnt);
                    } else {
                        cnt = 0;
                    }
                }
                int max_angle = std::max(max_angle, max_cnt * bin_angle);
                if (max_angle > min_conscultive_angle_thresh) {
                    img_filled.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(closest_pos.first, closest_pos.second);
                }
            }
        }
    }
    img = img_filled;
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

PointCloud<PointXYZRGB>::Ptr filter_outliers_via_radius(PointCloud<PointXYZRGB>::Ptr pcd, float radius, int k, bool set_negative) {
    pcl::PointCloud<PointXYZRGB>::Ptr cloud = pcd;
    pcl::PointCloud<PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<PointXYZRGB>);

    LOG(INFO) << "Filtering outliers..." << std::endl;
    LOG(INFO) << radius << ", " << k << std::endl;
    // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    // sor.setInputCloud(cloud);
    // sor.setMeanK(k);
    // sor.setStddevMulThresh(radius);
    RadiusOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMinNeighborsInRadius(k);
    sor.setRadiusSearch(radius);

    // sor.setNegative(true);
    // sor.filter(m_index_of_pcd); // ???

    sor.setNegative(set_negative);
    sor.filter(*cloud_filtered);

    LOG(INFO) << "before filtering:\tpoint number = " << pcd->points.size() << endl;
    LOG(INFO) << "after filtering:\tpoint number = " << cloud_filtered->points.size() << endl;
    LOG(INFO) << "filtered number = " << pcd->points.size() - cloud_filtered->points.size() << endl;
    return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_outliers_via_stats(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, float stddev, int meanK, bool set_negative) {
    pcl::PointCloud<PointXYZRGB>::Ptr cloud = pcd;
    pcl::PointCloud<PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<PointXYZRGB>);

    LOG(INFO) << "Statistical filtering outliers..." << std::endl;
    LOG(INFO) << stddev << ", " << meanK << std::endl;

    StatisticalOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(meanK);
    sor.setStddevMulThresh(stddev);
    sor.setNegative(set_negative);
    sor.filter(*cloud_filtered);

    LOG(INFO) << "before filtering:\tpoint number = " << pcd->points.size() << endl;
    LOG(INFO) << "after filtering:\tpoint number = " << cloud_filtered->points.size() << endl;
    LOG(INFO) << "filtered number = " << pcd->points.size() - cloud_filtered->points.size() << endl;
    return cloud_filtered;
}

std::vector<int> grid_segmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd, double grid_resolution) {
    PointXYZ min_value{1000000, 1000000, 1000000}, max_value{-1000000, -1000000, -1000000}, range;
    vector<int> labels;
    labels.resize(pcd->points.size());

    for (auto& pt: pcd->points) {
        min_value.x = std::min(min_value.x, pt.x);
        min_value.y = std::min(min_value.y, pt.y);

        max_value.x = std::max(max_value.x, pt.x);
        max_value.y = std::max(max_value.y, pt.y);
    }

    range.x = max_value.x - min_value.x;
    range.y = max_value.y - min_value.y;

    int x_division = static_cast<int>(range.x / grid_resolution) + 1, y_division = static_cast<int>(range.y / grid_resolution) + 1;
    double x_resolution = range.x / x_division, y_resolution = range.y / y_division;
    for(int idx = 0; idx < labels.size(); idx++) {
        auto& pt = pcd->points[idx];

        labels[idx] = static_cast<int>(pt.x / x_resolution) * y_division + static_cast<int>(pt.y / y_resolution);
    }
}

void filter_few_colors(cv::Mat &img, int few_color_threshold) {
    // construct color count map
    std::map<int, int> color_count_map;
    for (int j = 0; j < img.rows; ++j) {
        for (int i = 0; i < img.cols; ++i) {
            cv::Vec3b &c = img.at<cv::Vec3b>(j, i);
            int color_key = c[0] * 255 * 255 + c[1] * 255 + c[2];
            if (color_count_map.count(color_key) == 0) {
                color_count_map[color_key] = 1;
            } else {
                color_count_map[color_key]++;
            }
        }
    }
    // find few color color_key
    std::set<int> few_color_keys;
    for (const auto &p : color_count_map) {
        if (p.second <= few_color_threshold) few_color_keys.insert(p.first);
    }

    // filter out few colors
    for (int j = 0; j < img.rows; ++j) {
        for (int i = 0; i < img.cols; ++i) {
            cv::Vec3b &c = img.at<cv::Vec3b>(j, i);
            int color_key = c[0] * 255 * 255 + c[1] * 255 + c[2];
            if (few_color_keys.count(color_key) > 0) {
                c[0] = c[1] = c[2] = 0;
            }
        }
    }
}

void add_camera_trajectory_to_viewer(std::shared_ptr<pcl::visualization::PCLVisualizer> viewer, const std::vector<Eigen::Matrix4f> &Twcs) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camera_points(new PointCloud<PointXYZRGB>());
    for (size_t i = 0; i < Twcs.size(); ++i) {
        Eigen::Matrix4f e = Twcs[i];
        Eigen::Vector3f pwc = e.block<3, 1>(0, 3);
        Eigen::Quaternionf qwc(e.block<3, 3>(0, 0));
        const int visualize_length = 5;
        for (int j = 0; j < visualize_length; ++j) {
            // 0->x red, 1->y green, 2->z blue
            // slam axis : x->right, y->down, z->forward
            for (int i_axis = 0; i_axis < 3; ++i_axis) {
                Eigen::Vector3f pt_axis;
                pt_axis.setZero();
                pt_axis(i_axis) = 1.0f;
                pt_axis = qwc * pt_axis * (0.1 * j) + pwc;
                Eigen::Vector3i color;
                color.setZero();
                color(i_axis) = 200;
                pcl::PointXYZRGB pt;
                pt.x = pt_axis.x();
                pt.y = pt_axis.y();
                pt.z = pt_axis.z();
                pt.r = color.x();
                pt.g = color.y();
                pt.b = color.z();
                camera_points->points.emplace_back(pt);
            }
        }
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> colors(camera_points);
    viewer->addPointCloud(camera_points, colors, "camera_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "camera_points");
}

void update_candidate_list(PointCloud<PointXYZL>::Ptr& pcd, vector<float>& scores, vector<PointXYZL>& centers, int width) {
    const int pt_number = pcd->points.size(), label_size = centers.size();
    vector<int> startrow(label_size, 10000), startcol(label_size, 10000), endrow(label_size, 0), endcol(label_size, 0);
    vector<bool> visited(label_size, false);

    for (int idx = 0; idx < pt_number; idx++) {
        auto& label = pcd->points[idx].label;
        visited[label] = true;

        int row = idx / width, col = idx % width;
        startrow[label] = std::min(startrow[label], row);
        startcol[label] = std::min(startcol[label], col);
        endrow[label] = std::max(endrow[label], row);
        endcol[label] = std::max(endcol[label], col);

    }
    for (int idx = 1; idx < label_size; idx++) {
        if (!visited[idx]) {
            continue;
        }
        float score = (endcol[idx] - startcol[idx]) * (endrow[idx] - startrow[idx]);

        if (score > scores[idx]) {
            int image_center_row = (endrow[idx] + startrow[idx]) / 2, image_center_col = (endcol[idx] + startcol[idx]) / 2;
            auto& pt = pcd->points[image_center_row * width + image_center_col];
            if (pt.label != idx) {
                float normdis = 10000000000;
                for (int row = startrow[idx]; row < endrow[idx]; row++) {
                    for (int col = startcol[idx]; col < endcol[idx]; col++) {
                        int i = row *  width + col;
                        if (pcd->points[i].label == idx) {
                            float newdis =(row - image_center_row) * (row - image_center_row) + (col - image_center_col) * (col - image_center_col) ;
                            if ( newdis < normdis) {
                                normdis = newdis;
                                centers[idx] = pcd->points[i];
                            }
                        }
                    }
                }
            } else {
                centers[idx] = pt;
            }
            scores[idx] = score;
        }
    }
}
}
