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
    std::vector<int>                        m_semantic_label;
    std::vector<int>                        m_cluster_label;

    pcl::visualization::PCLVisualizer::Ptr m_viewer;                                // viewer
    
    std::vector<std::string> m_semantic_names;
    std::vector<std::string> m_landmarks_semantic_names, m_background_semantic_names;
    // std::vector<pcl::PointIndices>} m_clusters;

    MapManager();
    MapManager(const std::string& dir); //< load from serialized directory
    ~MapManager() {}

    void filter_outliers(float radius=1, int k=100);
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

    void dye_through_semantics();
    void dye_through_clusters();
    void dye_through_landmarks_semantics();
    void update_view();
    void show_point_cloud();
};
}

#endif