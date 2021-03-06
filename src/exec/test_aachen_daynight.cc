#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <map>
#include <sstream>
#include <random>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "MapManager.h"
#include "utils.h"

using namespace pcl;
using namespace lass;
using json = nlohmann::json;

namespace {
bool g_stop_view = false;
bool g_disbale_viewer = false;

const int resize_ratio = 2;

struct PoseData {
    std::string filename;
    // qwc
    Eigen::Quaternionf q;
    // pwc
    Eigen::Vector3f p;
    float focal;
    int width = 1920;
    int height = 1080;
};

struct Cluster {
    Cluster() {
        center.setZero();
        center_orig.setZero();
        max_visible_pixel = 0;
    }
    Eigen::Vector3f center;
    Eigen::Vector3f center_orig;
    int max_visible_pixel;
};

std::vector<Eigen::Matrix4f> g_Twcs_train;
std::vector<Eigen::Matrix4f> g_Twcs_test;
std::vector<Eigen::Matrix4f> g_Twcs_bad;
std::vector<Eigen::Matrix4f> g_Twcs_close_to_wall;
} // namespace

inline std::vector<PoseData> load_camera_data_from_nvm(const std::string &image_base_dir, const std::string &filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr *cloud = nullptr) {
    std::vector<PoseData> pose_data;
    std::vector<CameraF> cameras;
    std::vector<Point3DF> points;
    std::vector<Point2D> measurements;
    std::vector<int> pidx;
    std::vector<int> cidx;
    std::vector<std::string> names;
    std::vector<int> ptc;
    lass::load_nvm_file(filename.c_str(), cameras, points, measurements, pidx, cidx, names, ptc);
    CHECK(cameras.size() == names.size());
    CHECK(!cameras.empty());
    for (int i = 0; i < cameras.size(); ++i) {
        const auto &c = cameras[i];
        PoseData pd;
        Eigen::Matrix3f R;
        R << c.m[0][0], c.m[0][1], c.m[0][2],
             c.m[1][0], c.m[1][1], c.m[1][2],
             c.m[2][0], c.m[2][1], c.m[2][2];
        Eigen::Vector3f t;
        t << c.t[0], c.t[1], c.t[2];
        pd.q = R.transpose();
        pd.q.normalize();
        pd.p = -(pd.q * t);
        pd.focal = cameras[i].f;
        pd.filename = names[i];
        cv::Mat img = cv::imread(image_base_dir + "/" + pd.filename);
        pd.width = img.cols;
        pd.height = img.rows;
        pose_data.emplace_back(std::move(pd));
        printf("\r%d / %zu", i, cameras.size());
    }
    if (cloud) {
        LOG(WARNING) << __func__ << " point coordinate abs > 3000 would be removed!";
        for (int i = 0; i < points.size(); ++i) {
            pcl::PointXYZRGB pt;
            pt.x = points[i].xyz[0];
            pt.y = points[i].xyz[1];
            pt.z = points[i].xyz[2];
            float xxx = pt.x + pt.y + pt.z;
            if (std::isnan(xxx) || std::isinf(xxx)) continue;
            if (std::abs(pt.x) > 3000 || std::abs(pt.y) > 3000 || std::abs(pt.z) > 3000) continue;
            pt.r = pt.g = pt.b = 200;
            (*cloud)->points.push_back(pt);
        }
        print_var(points.size());
    }
    return pose_data;
}

inline void keyboard_callback(const pcl::visualization::KeyboardEvent &event, void *ptr) {
    LOG(INFO) << "key board event: " << event.getKeySym() << endl;
    if (event.getKeySym() == "n" && event.keyDown()) {
        g_stop_view = true;
    }
}

inline void visualize_rgb_points(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, bool visualize_trajectory = true) {
    if (g_disbale_viewer) return;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_color_handler(cloud);
    viewer->registerKeyboardCallback(keyboard_callback, nullptr);
    viewer->addPointCloud(cloud, cloud_color_handler, "base_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.5, "base_cloud");
    viewer->addCoordinateSystem(1.0);
    if (visualize_trajectory) {
        lass::add_camera_trajectory_to_viewer(viewer, g_Twcs_train, 2);
        lass::add_camera_trajectory_to_viewer(viewer, g_Twcs_test, 4);
        lass::add_camera_trajectory_to_viewer(viewer, g_Twcs_bad, -1);
        lass::add_camera_trajectory_to_viewer(viewer, g_Twcs_close_to_wall, -2);
    }
    while (!viewer->wasStopped() && !g_stop_view) {
        viewer->spinOnce(100);
    }
    g_stop_view = false;
}

inline void visualize_labeled_points(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud, std::vector<Cluster> *centers = nullptr) {
    if (g_disbale_viewer) return;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer);
    viewer->registerKeyboardCallback(keyboard_callback, nullptr);
    viewer->addPointCloud(cloud, "base_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.5, "base_cloud");
    viewer->addCoordinateSystem(1.0);
    if (centers) {
        pcl::PointCloud<PointXYZRGB>::Ptr centers_pcd(new pcl::PointCloud<PointXYZRGB>);
        for (const auto &c : *centers) {
            pcl::PointXYZRGB pt;
            pt.x = c.center.x();
            pt.y = c.center.y();
            pt.z = c.center.z();
            pt.r = 255;
            pt.g = pt.b = 0;
            centers_pcd->points.push_back(pt);
            pt.x = c.center_orig.x();
            pt.y = c.center_orig.y();
            pt.z = c.center_orig.z();
            pt.r = 0;
            pt.g = 0;
            pt.b = 255;
            centers_pcd->points.push_back(pt);
        }
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_color_handler(centers_pcd);
        viewer->addPointCloud(centers_pcd, cloud_color_handler, "center_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "center_cloud");
    }
    while (!viewer->wasStopped() && !g_stop_view) {
        viewer->spinOnce(100);
    }
    g_stop_view = false;
}

inline void visualize_label_points_from_rgb(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                            const std::vector<pcl::PointIndices> &cluster,
                                            const std::vector<int> &cluster_labels) {
    if (g_disbale_viewer) return;
    // assign to PointXYZL
    int outlier_clusters = 0;
    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_pcd(new pcl::PointCloud<pcl::PointXYZL>);
    labeled_pcd->points.reserve(cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto &pt_orig = cloud->points[i];
        pcl::PointXYZL pt;
        pt.x = pt_orig.x;
        pt.y = pt_orig.y;
        pt.z = pt_orig.z;
        if (cluster_labels[i] < 0) {
            outlier_clusters++;
            continue;
        }
        pt.label = cluster_labels[i];
        labeled_pcd->points.emplace_back(pt);
    }
    cout << "Total invalid clusters: " << outlier_clusters << endl;

    uint32_t label_idx = 0;
    // orig_label->compressed_label
    std::unordered_map<uint32_t, uint32_t> label_map;
    for (auto &p : labeled_pcd->points) {
        if (label_map.count(p.label) == 0) {
            label_map[p.label] = label_idx++;
        }
        p.label = label_map[p.label];
    }
    visualize_labeled_points(labeled_pcd);
}

// remove too small clusters, compress labels, compute and return cluster centers
inline std::vector<Cluster> reform_labeled_pcd(pcl::PointCloud<PointXYZL>::Ptr pcd, float small_cluster_thresh = 1.5) {
    // find too small clusters
    using min_max_pair = std::pair<Eigen::Vector3f, Eigen::Vector3f>;
    // label->min_max_pair
    std::unordered_map<uint32_t, min_max_pair> label_size;
    for (auto &p : pcd->points) {
        if (label_size.count(p.label) == 0) {
            min_max_pair minmax;
            minmax.first = {p.x, p.y, p.z};
            minmax.second = {p.x, p.y, p.z};
            label_size[p.label] = minmax;
        } else {
            auto &minmax = label_size[p.label];
            minmax.first = {
                std::min(minmax.first.x(), p.x),
                std::min(minmax.first.y(), p.y),
                std::min(minmax.first.z(), p.z)};
            minmax.second = {
                std::max(minmax.second.x(), p.x),
                std::max(minmax.second.y(), p.y),
                std::max(minmax.second.z(), p.z)};
        }
    }
    std::set<uint32_t> small_cluster_labels;
    for (const auto &p : label_size) {
        float cluster_size = (p.second.first - p.second.second).norm();
        if (cluster_size < small_cluster_thresh) small_cluster_labels.insert(p.first);
    }
    LOG(INFO) << "Find small clusters " << small_cluster_labels.size();
    // compress label and mark small clusters to 0
    // zero remains for invalid labels
    // remove point with label == 0
    pcl::PointIndices::Ptr outliers(new pcl::PointIndices());
    uint32_t label_idx = 1;
    std::unordered_map<uint32_t, uint32_t> label_map;
    for (int i = 0; i < pcd->points.size(); ++i) {
        auto &p = pcd->points[i];
        if (p.label == 0) {
            outliers->indices.push_back(i);
            continue;
        }
        if (small_cluster_labels.count(p.label) > 0) {
            p.label = 0;
            outliers->indices.push_back(i);
            continue;
        }
        if (label_map.count(p.label) == 0) {
            label_map[p.label] = label_idx++;
        }
        p.label = label_map[p.label];
    }
    pcl::ExtractIndices<pcl::PointXYZL> extract;
    extract.setInputCloud(pcd);
    extract.setIndices(outliers);
    extract.setNegative(true);
    extract.filter(*pcd);
    // compute centers
    std::vector<Cluster> centers(label_idx);
    std::vector<int> centers_counter(centers.size(), 0);
    for (const auto &p : pcd->points) {
        auto &c = centers[p.label];
        c.center += Eigen::Vector3f(p.x, p.y, p.z);
        centers_counter[p.label]++;
    }
    for (int i = 0; i < centers.size(); ++i) {
        int sz = std::max(1, centers_counter[i]);
        centers[i].center /= sz;
        centers[i].center_orig = centers[i].center;
    }
    return centers;
}

inline void point_process(const json &j_config, pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_pcd, pcl::PointCloud<pcl::PointXYZL>::Ptr &output_labeled_pcd, std::vector<Cluster> &centers) {
    auto &curr_pcd = input_pcd;
    // visualize_rgb_points(curr_pcd);

    pcl::PointCloud<PointXYZL>::Ptr curr_pcd_labeled(new pcl::PointCloud<PointXYZL>);
    curr_pcd_labeled->points.reserve(curr_pcd->points.size());
    // supervoxel clustering
    {
        LOG(INFO) << "start supervoxel clustering";
        SupervoxelClustering<PointXYZRGB> super(j_config["supervoxel"]["voxel_resolution"],
                                                j_config["supervoxel"]["seed_resolution"]);
        std::map<uint32_t, Supervoxel<PointXYZRGB>::Ptr> supervoxel_clusters;
        super.setInputCloud(curr_pcd);
        super.setColorImportance(0);
        super.setNormalImportance(0);
        super.setSpatialImportance(1);
        super.extract(supervoxel_clusters);

        LOG(INFO) << "super voxel cluster size = " << supervoxel_clusters.size() << endl;

        auto labeled_voxel_cloud = super.getLabeledVoxelCloud();

        visualize_labeled_points(labeled_voxel_cloud);

        centers = reform_labeled_pcd(labeled_voxel_cloud);
        LOG(INFO) << "final cluster size: " << centers.size();
        // visualize_labeled_points(labeled_voxel_cloud);

#if 0 // assign result via nearest neighbors
        pcl::KdTreeFLANN<pcl::PointXYZL> kdtree;
        kdtree.setInputCloud(labeled_voxel_cloud);
        int K = 1;
        std::vector<int> point_indices(K);
        std::vector<float> distances(K);
        for (int idx = 0; idx < curr_pcd->points.size(); ++idx) {
            const auto &orig_pt = curr_pcd->points[idx];
            pcl::PointXYZL pt;
            pt.x = orig_pt.x;
            pt.y = orig_pt.y;
            pt.z = orig_pt.z;
            if (kdtree.nearestKSearch(pt, K, point_indices, distances) > 0) {
                pt.label = labeled_voxel_cloud->points[point_indices[0]].label;
                curr_pcd_labeled->points.emplace_back(pt);
            }
        }
#else
        curr_pcd_labeled = labeled_voxel_cloud;
#endif
    }
    output_labeled_pcd = curr_pcd_labeled;
}

inline void filter_to_sparse_pcd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, json &j_config) {
    // uniform downsample
    {
        pcl::PointIndices::Ptr outliers(new pcl::PointIndices());
        float downsample_ratio = j_config["sparse"]["downsample_ratio"];
        for (int i = 0; i < cloud->points.size(); ++i) {
            if (rand01() > downsample_ratio) outliers->indices.push_back(i);
        }
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(outliers);
        extract.setNegative(true);
        extract.filter(*cloud);
    }
    // Statistical filtering outliers
    {
        const float meanK = j_config["sparse"]["statical_filter"]["mean_K"];
        const float stddev = j_config["sparse"]["statical_filter"]["stddev"];
        pcl::PointCloud<PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<PointXYZRGB>);

        LOG(INFO) << "Statistical filtering outliers..." << std::endl;
        LOG(INFO) << stddev << ", " << meanK << std::endl;

        StatisticalOutlierRemoval<PointXYZRGB> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(meanK);
        sor.setStddevMulThresh(stddev);
        // sor.setNegative(false);
        sor.filter(*cloud_filtered);

        LOG(INFO) << "before filtering:\tpoint number = " << cloud->points.size() << endl;
        LOG(INFO) << "after filtering:\tpoint number = " << cloud_filtered->points.size() << endl;
        LOG(INFO) << "filtered number = " << cloud->points.size() - cloud_filtered->points.size() << endl;
        cloud = cloud_filtered;
    }
}

inline void dump_parameters(const std::string &output_base_dir,
                            const std::vector<Cluster> &cluster_centers,
                            const std::vector<PoseData> &train_poses_twc,
                            const std::vector<PoseData> &test_poses_twc,
                            const std::vector<PoseData> &all_poses_twc) {
    const auto &centers = cluster_centers;
    json j_centers;
    j_centers.push_back(
        {centers[0].center.x(), centers[0].center.y(), centers[0].center.z()});

    for (int idx = 1; idx < centers.size(); idx++) {
        auto center = centers[idx];
        j_centers.push_back({center.center.x(), center.center.y(), center.center.z()});
    }
    ofstream o_id2centers{output_base_dir + "/id2centers.json"};
    o_id2centers << std::setw(4) << j_centers;

    json j_es;
    for (int idx = 0; idx < all_poses_twc.size(); idx++) {
        const auto &p = all_poses_twc[idx];
        Eigen::Matrix4f Tcw;
        Tcw.setIdentity();
        Tcw.block<3, 3>(0, 0) = p.q.conjugate().matrix();
        Tcw.block<3, 1>(0, 3) = -(p.q.conjugate() * p.p);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                j_es[p.filename].push_back(Tcw(r, c));
            }
        }
        j_es[p.filename].push_back(p.focal);
    }

    ofstream o_es(output_base_dir + "/out_extrinsics.json");
    o_es << std::setw(4) << j_es;

    // train test split
    std::vector<std::string> train_list, test_list;
    for (const auto &p : train_poses_twc) {
        train_list.push_back(p.filename);
    }
    for (const auto &p : test_poses_twc) {
        test_list.push_back(p.filename);
    }
    ofstream o_train_list{output_base_dir + "/train_list.json"};
    json j_train_list = train_list;
    o_train_list << std::setw(4) << j_train_list;
    ofstream o_test_list{output_base_dir + "/test_list.json"};
    json j_test_list = test_list;
    o_test_list << std::setw(4) << j_test_list;

    LOG(INFO) << "finished output json" << endl;
}

inline void adjust_cluster_centers_via_raycast_visibility(const std::vector<PoseData> &poses_twc, pcl::PointCloud<pcl::PointXYZL>::Ptr &labeled_pcd, std::vector<Cluster> &cluster_centers, double raycast_voxel_grid) {
    std::cout << __func__ << std::endl;
    

    PointCloud<PointXYZL>::Ptr image_pcd{new PointCloud<PointXYZL>};

    auto mm = std::make_unique<MapManager>();
    mm->m_target_pcd = labeled_pcd;
    mm->m_labeled_pcd = labeled_pcd;
    // mm->prepare_octree_for_target_pcd(0.3);
    // mm->prepare_octree_for_target_pcd(0.1); for greatcourt
    // mm->prepare_octree_for_target_pcd(0.1);
    mm->prepare_octree_for_target_pcd(raycast_voxel_grid);

    std::vector<pcl::PointXYZRGB> empty_centers;

    std::vector<int> center_counter(cluster_centers.size(), 0);
    std::vector<Eigen::Vector3f> center_sum(cluster_centers.size(), Eigen::Vector3f::Zero());

    for (int idx_pose = 0; idx_pose < poses_twc.size(); ++idx_pose) {
        const auto &p = poses_twc[idx_pose];

        // gt segmentation would be resized
        const int resize_ratio_visibility = 8;
        // set camera intrinsic
        camera_intrinsics K;
        K.cx = (p.width / 2) / resize_ratio_visibility;
        K.cy = (p.height / 2) / resize_ratio_visibility;
        K.width = p.width / resize_ratio_visibility;
        K.height = p.height / resize_ratio_visibility;
        // set focal
        float focal = p.focal;
        focal /= resize_ratio_visibility;
        K.fx = focal;
        K.fy = focal;
        Eigen::Matrix4f Tcw;
        Tcw.setIdentity();
        Tcw.block<3, 3>(0, 0) = p.q.conjugate().matrix();
        Tcw.block<3, 1>(0, 3) = -(p.q.conjugate() * p.p);
        PointCloud<PointXYZL>::Ptr visible_pcd{new PointCloud<PointXYZL>};
        mm->raycasting_pcd(Tcw, K, image_pcd, empty_centers, false, 3, 1.0, "labeled", &visible_pcd);
        //
        std::vector<int> cluster_visible_count(cluster_centers.size(), 0);
        std::vector<Eigen::Vector2i> cluster_pixel_location_sum(cluster_centers.size(), Eigen::Vector2i::Zero());
        for (int j = 0; j < K.height; j++) {
            for (int i = 0; i < K.width; i++) {
                auto &pt = image_pcd->points[j * K.width + i];
                if (pt.label == 0) continue;
                cluster_visible_count[pt.label]++;
                cluster_pixel_location_sum[pt.label] += Eigen::Vector2i(i, j);
            }
        }
        for (int i = 0; i < cluster_centers.size(); ++i) {
            auto &cluster = cluster_centers[i];
            if (cluster_visible_count[i] > cluster.max_visible_pixel) {
                Eigen::Vector2i center_2d = cluster_pixel_location_sum[i] / cluster_visible_count[i];
                // use mean depth from a small region
                int region_size = 3;
                int x_begin = std::max(0, center_2d.x() - region_size);
                int x_end = std::min(K.width - 1, center_2d.x() + region_size);
                int y_begin = std::max(0, center_2d.y() - region_size);
                int y_end = std::min(K.height - 1, center_2d.y() + region_size);
                float count = 0;
                Eigen::Vector3f pt_sum;
                pt_sum.setZero();
                for (int x = x_begin; x < x_end; ++x) {
                    for (int y = y_begin; y < y_end; ++y) {
                        auto &p = visible_pcd->points[y * K.width + x];
                        if (p.label != i) continue; // skip point which does not belong to current label
                        pt_sum += Eigen::Vector3f(p.x, p.y, p.z);
                        count++;
                    }
                }
                if (count > 0) {
                    cluster_centers[i].center = pt_sum / count;
                    cluster.max_visible_pixel = cluster_visible_count[i];
                }
            }
        }

        fprintf(stdout, "\r%d / %zu", idx_pose, poses_twc.size());
        fflush(stdout);
    }
}

inline void raycast_to_images(const json &j_config,
                              const std::string &output_base_dir,
                              const std::vector<PoseData> &poses_twc,
                              pcl::PointCloud<pcl::PointXYZL>::Ptr &labeled_pcd,
                              const std::vector<Cluster> &cluster_centers) {
    const float max_visible_distance = j_config.value("max_visible_distance", std::numeric_limits<float>::max());

    PointCloud<PointXYZL>::Ptr pcd{new PointCloud<PointXYZL>};

    auto mm = std::make_unique<MapManager>();
    mm->m_target_pcd = labeled_pcd;
    mm->m_labeled_pcd = labeled_pcd;
    mm->prepare_octree_for_target_pcd(0.3);

    // convert to pcl type cluster center
    std::vector<pcl::PointXYZRGB> pcl_centers;
    for (const auto &c : cluster_centers) {
        pcl::PointXYZRGB pt;
        pt.x = c.center.x();
        pt.y = c.center.y();
        pt.z = c.center.z();
        pcl_centers.emplace_back(pt);
    }

    for (int i = 0; i < poses_twc.size(); ++i) {
        const auto &p = poses_twc[i];
        // set camera intrinsic
        camera_intrinsics K;
        K.cx = (p.width / 2) / resize_ratio;
        K.cy = (p.height / 2) / resize_ratio;
        K.width = p.width / resize_ratio;
        K.height = p.height / resize_ratio;
        float focal = p.focal;
        focal /= resize_ratio;
        K.fx = focal;
        K.fy = focal;
        Eigen::Matrix4f Tcw;
        Tcw.setIdentity();
        Tcw.block<3, 3>(0, 0) = p.q.conjugate().matrix();
        Tcw.block<3, 1>(0, 3) = -(p.q.conjugate() * p.p);
        PointCloud<PointXYZL>::Ptr visible_pcd{new PointCloud<PointXYZL>};
        mm->raycasting_pcd(Tcw, K, pcd, pcl_centers, true, 3, 1.0, "labeled", &visible_pcd);
        cv::Mat save_img(cv::Size(K.width, K.height), CV_8UC3);
        save_img = 0;
        // draw to cv::Mat
        for (int j = 0; j < K.height; j++) {
            for (int i = 0; i < K.width; i++) {
                int idx = j * K.width + i;
                auto &pt = pcd->points[idx];
                auto &pt_3d = visible_pcd->points[idx];
                float distance = (Eigen::Vector3f(pt_3d.x, pt_3d.y, pt_3d.z) - p.p).norm();
                if (distance > max_visible_distance) continue;
                auto &c = save_img.at<cv::Vec3b>(j, i);
                if (pt.label == 0) {
                    c[0] = c[1] = c[2] = 0;
                } else {
                    lass::label_to_rgb(c[0], c[1], c[2], pt.label);
                    {
                        // debug scope
                        // make sure each color map to only one label
                        uint32_t unique_key = (uint32_t(c[0]) << 16) + (uint32_t(c[1]) << 8) + uint32_t(c[2]);
                        static std::map<uint32_t, uint32_t> color_map;
                        if (color_map.count(unique_key) > 0) {
                            CHECK(color_map[unique_key] == pt.label);
                        } else {
                            color_map[unique_key] = pt.label;
                        }
                    }
                }
            }
        }
        // cv::imshow("raycast", save_img);
        std::string output_filename = output_base_dir + "/" + p.filename;
        // create directory if not exist
        int ret = system(("mkdir -p " + output_filename.substr(0, output_filename.find_last_of("/"))).c_str());
        output_filename = output_filename.substr(0, output_filename.length() - 3) + "png";
        cv::imwrite(output_filename, save_img);
        // cv::waitKey(0);
        fprintf(stdout, "\r%d / %zu", i, poses_twc.size());
        fflush(stdout);
    }
}

void filter_out_bad_poses(const json &j_config, const std::string &data_base_dir, std::vector<PoseData> &poses, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
    if (j_config.count("bad_pose_list") == 0) return;
    std::ifstream ifs(data_base_dir + "/" + std::string(j_config["bad_pose_list"]));
    json j_bad_poses_list;
    ifs >> j_bad_poses_list;
    ifs.close();
    std::unordered_set<std::string> bad_poses_list;
    for (const auto &n : j_bad_poses_list) {
        bad_poses_list.insert(std::string(n));
    }
    // prune poses via bad_poses_list
    for (auto it = poses.begin(); it != poses.end(); ++it) {
        if (bad_poses_list.count(it->filename) > 0) {
            Eigen::Matrix4f Twc;
            Twc.setIdentity();
            Twc.block<3, 3>(0, 0) = it->q.matrix();
            Twc.block<3, 1>(0, 3) = it->p;
            g_Twcs_bad.push_back(Twc);
            // it = poses.erase(it);
        } else {
            // it++;
        }
    }
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud);
    int K = 1;
    std::vector<int> point_indices(K);
    std::vector<float> distances(K);
    // filter via distance to nearest point
    int count_near_wall = 0;
    json j_pose_near_wall_list;
    for (auto it = poses.begin(); it != poses.end(); ++it) {
        bool too_close_to_wall = false;
        pcl::PointXYZRGB pt;
        pt.x = it->p.x();
        pt.y = it->p.y();
        pt.z = it->p.z();
        if (kdtree.nearestKSearch(pt, K, point_indices, distances) > 0) {
            auto &p = cloud->points[point_indices[0]];
            float distance = (Eigen::Vector3f(p.x, p.y, p.z) - it->p).norm();
            if (distance < 0.5) too_close_to_wall = true;
        }
        if (too_close_to_wall) {
            Eigen::Matrix4f Twc;
            Twc.setIdentity();
            Twc.block<3, 3>(0, 0) = it->q.matrix();
            Twc.block<3, 1>(0, 3) = it->p;
            g_Twcs_close_to_wall.push_back(Twc);
            if (bad_poses_list.count(it->filename) == 0) j_bad_poses_list.push_back(it->filename);
            j_pose_near_wall_list.push_back(it->filename);
            count_near_wall++;
            // it = poses.erase(it);
        } else {
            // it++;
        }
    }
    LOG(INFO) << "find pose near wall " << count_near_wall;
    std::ofstream ofs;
    ofs.open(data_base_dir + "/bad_pose_full.json");
    ofs << std::setw(4) << j_bad_poses_list;
    ofs.close();
    ofs.open(data_base_dir + "/pose_near_wall.json");
    ofs << std::setw(4) << j_pose_near_wall_list;
    ofs.close();
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage ./test_aachen_daynight config_filename data_base_dir output_base_dir\n";
        exit(-1);
    }
    const std::string config_filename(argv[1]);
    const std::string data_base_dir(argv[2]);
    const std::string output_base_dir(argv[3]);
    // load config
    std::ifstream ifs(config_filename);
    json j_config;
    ifs >> j_config;
    ifs.close();
    // load data
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr curr_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr nvm_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::io::loadPLYFile(data_base_dir + "/" + std::string(j_config["ply"]), *curr_pcd);
    load_and_sample_obj(data_base_dir + "/" + std::string(j_config["obj"]), int(j_config["point_sample_size"]), curr_pcd);
    auto poses_twc_train = load_camera_data_from_nvm(data_base_dir + "/" + std::string(j_config["image_base_dir"]), data_base_dir + "/" + std::string(j_config["nvm"]), &nvm_pcd);
    decltype(poses_twc_train) poses_twc_test;
    double raycast_voxel_grid = j_config["raycast_voxel_grid"].get<double>();

    std::vector<PoseData> poses_twc_all = poses_twc_train;
    poses_twc_all.insert(poses_twc_all.end(), poses_twc_test.begin(), poses_twc_test.end());
    // filter_out_bad_poses(j_config, data_base_dir, poses_twc_all, curr_pcd);
    for (const auto &p : poses_twc_train) {
        Eigen::Matrix4f Twc;
        Twc.setIdentity();
        Twc.block<3, 3>(0, 0) = p.q.matrix();
        Twc.block<3, 1>(0, 3) = p.p;
        g_Twcs_train.push_back(Twc);
    }
    for (const auto &p : poses_twc_test) {
        Eigen::Matrix4f Twc;
        Twc.setIdentity();
        Twc.block<3, 3>(0, 0) = p.q.matrix();
        Twc.block<3, 1>(0, 3) = p.p;
        g_Twcs_test.push_back(Twc);
    }

    // visualize_rgb_points(nvm_pcd);
    // visualize_rgb_points(curr_pcd);
    // visualize_pcd(curr_pcd);

    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_pcd(new pcl::PointCloud<pcl::PointXYZL>);
    std::vector<Cluster> cluster_centers;
    point_process(j_config, curr_pcd, labeled_pcd, cluster_centers);

    adjust_cluster_centers_via_raycast_visibility(poses_twc_all, labeled_pcd, cluster_centers, raycast_voxel_grid);

    visualize_labeled_points(labeled_pcd, &cluster_centers);
    int ret = system(("mkdir -p " + output_base_dir).c_str());
    dump_parameters(output_base_dir, cluster_centers, poses_twc_train, poses_twc_test, poses_twc_all);
    raycast_to_images(j_config, output_base_dir, poses_twc_all, labeled_pcd, cluster_centers);
    LOG(INFO) << "Finish All";

    return 0;
}
