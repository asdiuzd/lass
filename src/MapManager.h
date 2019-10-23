#ifndef MAPMANAGER_H
#define MAPMANAGER_H

#include <pcl/io/ply_io.h>

#include "utils.h"
#include "json.h"

extern bool stop_view;
namespace lass {
typedef enum LandmarkType {
    UNKNOWN,
    REMOVED,
    LANDMARK,
    BACKGROUND,
    HIGHLIGHT,
    NUMBER
} LandmarkType;

typedef struct camera_intrinsics {
    float cx, cy, fx, fy;
    int width, height;
} camera_intrinsics;

class MapManager {
private:
    static const std::string model_name;
    static const std::string semantic_name;
    static const std::string parameters_name;

    void initialize_viewer();
    void update_camera_trajectory_to_viewer();
public:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  m_pcd;
    pcl::PointCloud<pcl::PointXYZL>::Ptr m_target_pcd;                                   // point cloud
    std::vector<bool>                       m_show_flag;
    std::vector<int>                        m_semantic_label, m_keypoints_label, m_render_label;
    std::vector<LandmarkType>               m_landmark_label;
    std::vector<pcl::PointIndices>          m_clusters;
    std::vector<int>                        m_cluster_label;
    pcl::PointIndices::Ptr                  m_index_of_landmark, m_index_of_background, m_index_of_removed, m_index_of_unknown;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZL> m_octree;

    bool m_use_flag = false;
    bool m_show_target_pcd = false;
    bool m_show_camera_extrinsics = false;  // visualize camera trajectory/extrinsics
    pcl::visualization::PCLVisualizer::Ptr m_viewer;                                // viewer
    int max_target_label;
    
    std::vector<std::string> m_semantic_names;
    std::vector<std::string> m_landmarks_semantic_names, m_background_semantic_names, m_removed_semantic_names;
    // std::vector<pcl::PointIndices> m_clusters;

    std::vector<Eigen::Matrix4f> m_camera_extrinsics;
    std::vector<int> m_camera_types;

    MapManager();
    MapManager(const std::string& dir); //< load from serialized directory
    ~MapManager() {
    }

    void filter_outliers(float radius=2, int k=5);
    void filter_landmarks_through_background(int knn=20, float ratio=0.5);
    void filter_supervoxels_through_background();
    void filter_useless_semantics_from_json(const std::string& fn);
    void filter_minor_segmentations(int number_threshold=20);
    void filter_points_near_cameras(float radius);

    void load_nvm_pcl(const std::string& fn);
    void load_pcd_pcl(const std::string& fn);
    void load_ply_pcl(const std::string& fn);
    void load_semantic_json(const std::string& fn);
    void load_parameters(const std::string& fn);
    void load_from_dir(const std::string& dir);
    void export_to_pcd(const std::string& fn);
    void export_to_semantic_json(const std::string& fn);
    void export_to_parameters(const std::string& fn);
    void export_to_dir(const std::string& dir);

    void figure_out_landmarks_annotation();
    void grid_landmark_clustering();
    void euclidean_landmark_clustering();
    void supervoxel_landmark_clustering(float voxel_resolution = 1.0, float seed_resolution = 21.0, float spatial_importance = 1, float color_importance = 0, float normal_importance = 0);

    void prepare_octree_for_target_pcd(float resolution = 1.0f);
    void raycasting_target_pcd(const Eigen::Matrix4f& extrinsics, const camera_intrinsics& intrinsics, pcl::PointCloud<pcl::PointXYZL>::Ptr& pcd, bool depthDE = true, int stride = 15, float scale = 1);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_landmarks();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_background();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_removed();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_unknown();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_points(pcl::PointIndices::Ptr indices);

    void set_view_target_pcd(bool flag) {
        m_show_target_pcd = flag;
    }

    void dye_through_semantics();
    void dye_through_clusters();
    void dye_through_landmarks();
    void dye_through_render();
    void update_view();
    void show_point_cloud();
};
}

#endif
