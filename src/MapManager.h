#ifndef MAPMANAGER_H
#define MAPMANAGER_H

#include "utils.h"
#include "json.h"

extern bool stop_view;
namespace lass {
typedef enum LandmarkType {
    UNKNOWN,
    REMOVED,
    LANDMARK,
    BACKGROUND,
    NUMBER
} LandmarkType;

class MapManager {
private:
    static const std::string model_name;
    static const std::string semantic_name;
    static const std::string parameters_name;

    void initialize_viewer();
public:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  m_pcd;                                   // point cloud
    std::vector<bool>                       m_show_flag;
    std::vector<int>                        m_semantic_label, m_keypoints_label;
    std::vector<LandmarkType>               m_landmark_label;
    std::vector<int>                        m_cluster_label;
    pcl::PointIndices::Ptr                  m_index_of_landmark, m_index_of_background, m_index_of_removed, m_index_of_unknown;

    bool m_use_flag = false;
    pcl::visualization::PCLVisualizer::Ptr m_viewer;                                // viewer
    
    std::vector<std::string> m_semantic_names;
    std::vector<std::string> m_landmarks_semantic_names, m_background_semantic_names, m_removed_semantic_names;
    // std::vector<pcl::PointIndices>} m_clusters;

    MapManager();
    MapManager(const std::string& dir); //< load from serialized directory
    ~MapManager() {}

    void filter_outliers(float radius=2, int k=5);
    void filter_useless_semantics_from_json(const std::string& fn);

    void load_nvm_pcl(const std::string& fn);
    void load_pcd_pcl(const std::string& fn);
    void load_semantic_json(const std::string& fn);
    void load_parameters(const std::string& fn);
    void load_from_dir(const std::string& dir);
    void export_to_pcd(const std::string& fn);
    void export_to_semantic_json(const std::string& fn);
    void export_to_parameters(const std::string& fn);
    void export_to_dir(const std::string& dir);

    void figure_out_landmarks_annotation();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_landmarks();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_background();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_removed();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_unknown();

    void dye_through_semantics();
    void dye_through_clusters();
    void dye_through_landmarks();
    void update_view();
    void show_point_cloud();
};
}

#endif