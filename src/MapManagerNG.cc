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
#include <thread>
#include <chrono>
#include <random>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "MapManager.h"

using namespace std;
using namespace pcl;
using namespace cv;
using json = nlohmann::json;
using namespace lass;

void MapManager::filter_outliers_via_statistics(float stddev, float mean_k) {
}

void MapManager::filter_and_clustering() {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr curr_pcd = extract_landmarks();
    
    // Statistical filtering outliers
    {
        const float meanK = 100.0;
        const float stddev = 1.0;
        pcl::PointCloud<PointXYZRGB>::Ptr cloud = curr_pcd;
        pcl::PointCloud<PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<PointXYZRGB>);

        LOG(INFO) << "Statistical filtering outliers..." << std::endl;
        LOG(INFO) << stddev << ", " << meanK << std::endl;

        StatisticalOutlierRemoval<PointXYZRGB> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(meanK);
        sor.setStddevMulThresh(stddev);
        // sor.setNegative(false);
        sor.filter(*cloud_filtered);

        LOG(INFO) << "before filtering:\tpoint number = " << curr_pcd->points.size() << endl;
        LOG(INFO) << "after filtering:\tpoint number = " << cloud_filtered->points.size() << endl;
        LOG(INFO) << "filtered number = " << curr_pcd->points.size() - cloud_filtered->points.size() << endl;
        curr_pcd = cloud_filtered;
    }
    // filter points near cameras
    {
        const float radius = 1.5;
        LOG(INFO) << __func__ << endl;
        LOG(INFO) << "filter points near cameras";
        if (m_camera_extrinsics.empty()) {
            LOG(ERROR) << "m_camera_extrinsics is empty, filter_points_near_cameras would do nothing!";
            return;
        }
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        kdtree.setInputCloud(curr_pcd);
        std::set<int> outlier_indices_set;
        for (Eigen::Matrix4f e : m_camera_extrinsics) {
            // apply rotation fix to extrinsics
            e(1, 3) *= -1; e(2, 3) *= -1;
            e(0, 1) *= -1; e(0, 2) *= -1;
            e(1, 0) *= -1; e(2, 0) *= -1;
            Eigen::Vector3f pcw = e.block<3, 1>(0, 3);
            Eigen::Quaternionf qcw(e.block<3, 3>(0, 0));
            Eigen::Vector3f pwc = -(qcw.conjugate() * pcw);
            std::vector<int> near_indices;
            std::vector<float> distances;
            pcl::PointXYZRGB pt;
            pt.x = pwc.x();
            pt.y = pwc.y();
            pt.z = pwc.z();
            if (kdtree.radiusSearch(pt, radius, near_indices, distances) > 0) {
                std::copy(near_indices.begin(), near_indices.end(),std::inserter(outlier_indices_set, outlier_indices_set.end()));
            }
        }
        // https://stackoverflow.com/questions/44921987/removing-points-from-a-pclpointcloudpclpointxyzrgb
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        pcl::PointIndices::Ptr outliers(new pcl::PointIndices());
        outliers->indices.insert(outliers->indices.end(), outlier_indices_set.begin(), outlier_indices_set.end());
        extract.setInputCloud(curr_pcd);
        extract.setIndices(outliers);
        extract.setNegative(true);
        extract.filter(*curr_pcd);
        LOG(INFO) << "outlier_indices size " << outlier_indices_set.size();
    }

    // euclidean clustering
    {
        LOG(INFO) << "euclidean clustering";
        LOG(INFO) << "Build Kdtree" << endl;
        search::KdTree<PointXYZRGB>::Ptr tree(new search::KdTree<PointXYZRGB>);
        tree->setInputCloud(curr_pcd);
        EuclideanClusterExtraction<PointXYZRGB> ec;
        cout << "Clustering extracting" << endl;
        ec.setClusterTolerance(0.7);
        ec.setMinClusterSize(100);
        ec.setMaxClusterSize(10000000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(curr_pcd);
        auto curr_indices = PointIndices::Ptr{new PointIndices};
        curr_indices->indices.resize(curr_pcd->points.size());
        for (int idx = 0; idx < curr_pcd->points.size(); idx++) {
            curr_indices->indices[idx] = idx;
        }
        ec.setIndices(curr_indices);
        ec.extract(m_clusters);
        cout << "Total: " << m_clusters.size() << " clusters" << endl;
        cout << "Labeling" << endl;
        
        m_cluster_label.clear();

        m_cluster_label.resize(curr_pcd->points.size(), -1);
        for (int label = 0; label < m_clusters.size(); label++) {
            auto &indices = m_clusters[label];
            for (auto &idx: indices.indices) {
                m_cluster_label[idx] = label;
            }
        }
    }

    // supervoxel for large clusters
    {
        int curr_valid_label = m_clusters.size();
        // cluster larger than cluster_size_th will be performed supervoxel
        const int cluster_size_th = 700;
        int count_process = 0;
        for (int i_cluster = 0; i_cluster < m_clusters.size(); ++i_cluster) {
            if (m_clusters[i_cluster].indices.size() >= cluster_size_th) {
                count_process++;
            }
        }
        LOG(INFO) << "We will process " << count_process << "clusters";

        for (int i_cluster = 0; i_cluster < m_clusters.size(); ++i_cluster) {
            const auto &indices = m_clusters[i_cluster].indices;
            // only process supervoxel over large clusters
            if (indices.size() < cluster_size_th) continue;
            SupervoxelClustering<PointXYZRGB> super(1.0, 42.0);
            map<uint32_t, Supervoxel<PointXYZRGB>::Ptr> supervoxel_clusters;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_pcd(new pcl::PointCloud<pcl::PointXYZRGB>());
            for (const auto &idx : indices) {
                local_pcd->points.emplace_back(curr_pcd->points[idx]);
            } 
            super.setInputCloud(local_pcd);
            super.setColorImportance(0);
            super.setNormalImportance(0);
            super.setSpatialImportance(1);
            super.extract(supervoxel_clusters);
            // TODO(ybbbbt): supervoxel 0 -> invalid
            if (supervoxel_clusters.empty()) continue;
            // LOG(INFO) << "super voxel cluster size = " << supervoxel_clusters.size() << endl;
            
            auto local_labeled_cloud = super.getLabeledVoxelCloud();

            // remove point with label == 0
            pcl::PointIndices::Ptr outliers(new pcl::PointIndices());
            for (int i = 0; i < local_labeled_cloud->points.size(); ++i) {
                if (local_labeled_cloud->points[i].label == 0) {
                    outliers->indices.push_back(i);
                    LOG(FATAL) << "Invalid label found in cluster level supervoxel!";
                    CHECK(false);
                }
            }
            pcl::ExtractIndices<pcl::PointXYZL> extract;
            extract.setInputCloud(local_labeled_cloud);
            extract.setIndices(outliers);
            extract.setNegative(true);
            extract.filter(*local_labeled_cloud);

            // assign result via nearest neighbors
            pcl::KdTreeFLANN<pcl::PointXYZL> kdtree;
            kdtree.setInputCloud(local_labeled_cloud);
            int K = 1;
            std::vector<int> point_indices(K);
            std::vector<float> distances(K);
            for (const auto &idx : indices) {
                const auto &orig_pt = curr_pcd->points[idx];
                pcl::PointXYZL pt;
                pt.x = orig_pt.x;
                pt.y = orig_pt.y;
                pt.z = orig_pt.z;
                if (kdtree.nearestKSearch(pt, K, point_indices, distances) > 0) {    
                    m_cluster_label[idx] = local_labeled_cloud->points[point_indices[0]].label + curr_valid_label;
                }
            }
            curr_valid_label += supervoxel_clusters.size();
        }
    }
    // TODO(ybbbbt): assign too small clusters to unlabeled
    // TODO(ybbbbt): assign unlabeled to nearest label
    {
        // auto orig_labeled_pts = new pcl::PointCloud<pcl::PointXYZL>();
        // assign to m_target_pcd
        int outlier_clusters = 0;
        m_target_pcd.reset(new pcl::PointCloud<pcl::PointXYZL>());
        m_target_pcd->points.reserve(curr_pcd->points.size());
        for (size_t i = 0; i < curr_pcd->points.size(); ++i) {
            const auto &pt_orig = curr_pcd->points[i];
            pcl::PointXYZL pt;
            pt.x = pt_orig.x;
            pt.y = pt_orig.y;
            pt.z = pt_orig.z;
            if (m_cluster_label[i] < 0) {
                outlier_clusters++;
                continue;
            }
            pt.label = m_cluster_label[i];
            m_target_pcd->points.emplace_back(pt);
        }
        cout << "Total invalid clusters: " << outlier_clusters << endl;
    }
    m_labeled_pcd = m_target_pcd;

    uint32_t label_idx = 0;
    // orig_label->compressed_label
    std::unordered_map<uint32_t, uint32_t> label_map;
    for (auto &p : m_labeled_pcd->points) {
        if (label_map.count(p.label) == 0) {
            label_map[p.label] = label_idx++;
        }
        p.label = label_map[p.label];
    }
    {
        // random labels
        std::vector<int> random_labels(label_map.size());
        int n(0);
        std::generate(std::begin(random_labels), std::end(random_labels), [&n] { return n++; });
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(random_labels), std::end(random_labels), rng);
        for (int i = 0; i < m_labeled_pcd->points.size(); ++i) {
            m_labeled_pcd->points[i].label = random_labels[m_labeled_pcd->points[i].label];
        }
    }
    LOG(INFO) << "finally, we got " << label_map.size() << " clusters";
    max_target_label = label_map.size();
}
