#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sstream>
#include <thread>
#include <chrono>

#include "utils.h"
#include "MapManager.h"

using namespace std;
using namespace pcl;
using json = nlohmann::json;

namespace lass {

const std::string MapManager::model_name{"all.pcd"};
const std::string MapManager::semantic_name{"semantic_names.json"};
const std::string MapManager::parameters_name{"parameters.txt"};

bool stop_view = false;

void keyboardEvent(const pcl::visualization::KeyboardEvent &event, void *mm_void) {
    LOG(INFO) << "key board event: " << event.getKeySym() << endl;
    auto mm = static_cast<MapManager *>(mm_void);
    if (event.getKeySym() == "s" && event.keyDown()) {
        // pm->view_with_threshold(pm->m_mu_threshold, pm->m_sigma_threshold, pm->m_single_point_threshold);
        mm->dye_through_semantics();
    } else if (event.getKeySym() == "p" && event.keyDown()) {
        // pm->view_single_point_threshold(pm->m_single_point_threshold);
        mm->export_to_dir("./tmp");
    } else if (event.getKeySym() == "r" && event.keyDown()) {
        stop_view = true;
    } else if (event.getKeySym() == "t" && event.keyDown()) {
        LOG(INFO) << "Toggle use show flag" << endl;
        mm->m_use_flag = !mm->m_use_flag;
    }
}

void MapManager::initialize_viewer() {
    m_viewer->setBackgroundColor(0, 0, 0);
    m_viewer->addCoordinateSystem(1.0);
    m_viewer->initCameraParameters();
    m_viewer->registerKeyboardCallback(keyboardEvent, (void*)this);
}

void MapManager::update_view() {
    m_viewer->removeAllPointClouds();
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(m_pcd);
    m_viewer->addPointCloud<pcl::PointXYZRGB>(m_pcd, rgb, "cloud");
    m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
}

MapManager::MapManager():
    m_pcd(new PointCloud<PointXYZRGB>()),
    m_viewer(new visualization::PCLVisualizer()) {
    initialize_viewer();
}

MapManager::MapManager(const std::string& dir):
    m_pcd(new PointCloud<PointXYZRGB>()),
    m_viewer(new visualization::PCLVisualizer()) {
    LOG(INFO) << "MapManager c'tor from: " << dir << endl;
    initialize_viewer();
    load_from_dir(dir);
}

void MapManager::load_semantic_json(const std::string& fn) {
    json j;
    ifstream s_in(fn.c_str());
    s_in >> j;

    m_semantic_names.resize(j["overall_semantics"].size());
    for(int idx = 0; idx < j["overall_semantics"].size(); idx++) {
        m_semantic_names[idx] = j["overall_semantics"][idx].get<string>();
    }

    m_landmarks_semantic_names.resize(j["landmarks"].size());
    for(int idx = 0; idx < j["landmarks"].size(); idx++) {
        m_landmarks_semantic_names[idx] = j["landmarks"][idx].get<string>();
    }

    m_background_semantic_names.resize(j["background"].size());
    for(int idx = 0; idx < j["background"].size(); idx++) {
        m_background_semantic_names[idx] = j["background"][idx].get<string>();
    }

    m_removed_semantic_names.resize(j["removed"].size());
    for(int idx = 0; idx < j["removed"].size(); idx++) {
        m_removed_semantic_names[idx] = j["removed"][idx].get<string>();
    }
}

void MapManager::export_to_semantic_json(const std::string &fn) {
    json j;

    for(int idx = 0; idx < m_semantic_names.size(); idx++) {
        j["overall_semantics"].push_back(m_semantic_names[idx]);
    }
    for(int idx = 0; idx < m_landmarks_semantic_names.size(); idx++) {
        j["landmarks"].push_back(m_landmarks_semantic_names[idx]);
    }
    for(int idx = 0; idx < m_background_semantic_names.size(); idx++) {
        j["background"].push_back(m_background_semantic_names[idx]);
    }
    for(int idx = 0; idx < m_removed_semantic_names.size(); idx++) {
        j["removed"].push_back(m_removed_semantic_names[idx]);
    }

    ofstream s_on(fn.c_str());
    s_on << j;
}

void MapManager::load_parameters(const std::string& fn) {
    ifstream p_in(fn.c_str());
    int number;
    p_in >> number;
    m_semantic_label.resize(number);
    for (int idx = 0; idx < number; idx++) {
        p_in >> m_semantic_label[idx];
    }
}

void MapManager::export_to_parameters(const std::string& fn) {
    ofstream p_on(fn.c_str());

    p_on << m_semantic_label.size() << endl;

    for (auto& label: m_semantic_label) {
        p_on << label << endl;
    }
}

void MapManager::load_pcd_pcl(const std::string& fn) {
    PCDReader reader;
    reader.read<PointXYZRGB>(fn.c_str(), *m_pcd);
}

void MapManager::export_to_pcd(const std::string& fn) {
    pcl::PCDWriter writer;
    writer.writeBinaryCompressed<pcl::PointXYZRGB>(fn, *m_pcd);
}

void MapManager::load_from_dir(const std::string& dir) {
    const string mn = dir + "/" + model_name;
    load_pcd_pcl(mn);

    const string sn = dir + "/" + semantic_name;
    load_semantic_json(sn);

    const string pn = dir + "/" + parameters_name;
    load_parameters(pn);
}

void MapManager::export_to_dir(const std::string& dir) {
    const string mn = dir + "/" + model_name;
    export_to_pcd(mn);

    const string sn = dir + "/" + semantic_name;
    export_to_semantic_json(sn);

    const string pn = dir + "/" + parameters_name;
    export_to_parameters(pn);
}

void MapManager::dye_through_semantics() {
    int max_semantic = *max_element(m_semantic_label.begin(), m_semantic_label.end());
    LOG(INFO) << "max semantic label = " << max_semantic << endl;

    for (int idx = 0; idx < m_pcd->points.size(); idx++) {
        auto &pt = m_pcd->points[idx];
        auto &semantic = m_semantic_label[idx];

        if (semantic == -1) {
            pt.r = pt.g = pt.b = 10;
        } else {
            GroundColorMix(pt.r, pt.g, pt.b, normalize_value(semantic, 0, max_semantic));
        }
    }
}

void MapManager::dye_through_clusters() {
    int max_cluster_label = *max_element(m_cluster_label.begin(), m_cluster_label.end());

    for (int idx = 0; idx < m_pcd->points.size(); idx++) {
        auto &pt = m_pcd->points[idx];
        auto &cluster = m_cluster_label[idx];

        if (cluster == -1) {
            pt.r = pt.g = pt.b = 10;
        } else {
            GroundColorMix(pt.r, pt.g, pt.b, normalize_value(cluster, 0, max_cluster_label));
        }
    }
}

void MapManager::dye_through_landmarks() {
    for (int idx = 0; idx < m_pcd->points.size(); idx++) {
        auto &pt = m_pcd->points[idx];
        if (m_semantic_label[idx] == -1) {
            GroundColorMix(pt.r, pt.g, pt.b, normalize_value(LandmarkType::UNKNOWN, 0, LandmarkType::NUMBER));
        } else {
            auto label = m_landmark_label[m_semantic_label[idx]];
            GroundColorMix(pt.r, pt.g, pt.b, normalize_value(label, 0, LandmarkType::NUMBER));
        }
    }
}

void MapManager::figure_out_landmarks_annotation() {
    m_landmark_label.resize(m_semantic_names.size(), LandmarkType::UNKNOWN);
    m_index_of_landmark = PointIndices::Ptr{new PointIndices};
    m_index_of_background = PointIndices::Ptr{new PointIndices};
    m_index_of_removed = PointIndices::Ptr{new PointIndices};
    m_index_of_unknown = PointIndices::Ptr{new PointIndices};

    for (int idx = 0; idx < m_semantic_names.size(); idx++) {
        auto &name = m_semantic_names[idx];

        auto it1 = std::find(m_landmarks_semantic_names.begin(), m_landmarks_semantic_names.end(), name);
        if (it1 != m_landmarks_semantic_names.end()) {
            m_landmark_label[idx] = LandmarkType::LANDMARK;
            continue;
        }

        auto it2 = std::find(m_background_semantic_names.begin(), m_background_semantic_names.end(), name);
        if (it2 != m_background_semantic_names.end()) {
            m_landmark_label[idx] = LandmarkType::BACKGROUND;
            continue;
        }

        auto it3 = std::find(m_removed_semantic_names.begin(), m_removed_semantic_names.end(), name);
        if (it3 != m_removed_semantic_names.end()) {
            m_landmark_label[idx] = LandmarkType::REMOVED;
            continue;
        }
    }

    vector<int> counter;
    counter.resize(LandmarkType::NUMBER);
    for (int idx = 0; idx < m_pcd->points.size(); idx++) {
        if (m_semantic_label[idx] == -1) {
            counter[LandmarkType::UNKNOWN]++;
        } else {
            auto& label = m_landmark_label[m_semantic_label[idx]];
            counter[label]++;

            switch (label) {
            case LandmarkType::LANDMARK:
                m_index_of_landmark->indices.push_back(idx);
                break;

            case LandmarkType::BACKGROUND:
                m_index_of_background->indices.push_back(idx);
                break;

            case LandmarkType::REMOVED:
                m_index_of_removed->indices.push_back(idx);
                break; 

            default:
                m_index_of_unknown->indices.push_back(idx);
                break;
            }
        }
    }

    LOG(INFO) << "statistic information: " << endl;
    for (int idx = 0; idx < counter.size(); idx++) {
        LOG(INFO) << "\t" << idx << ":\t" << counter[idx] << endl;
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr MapManager::extract_landmarks() {
    PointCloud<PointXYZRGB>::Ptr    extracted_pcd{new PointCloud<PointXYZRGB>()};
    ExtractIndices<PointXYZRGB>     extractor;

    extractor.setInputCloud(m_pcd);
    extractor.setIndices(m_index_of_landmark);
    extractor.filter(*extracted_pcd);

    return extracted_pcd;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr MapManager::extract_background() {
    PointCloud<PointXYZRGB>::Ptr    extracted_pcd{new PointCloud<PointXYZRGB>()};
    ExtractIndices<PointXYZRGB>     extractor;

    extractor.setInputCloud(m_pcd);
    extractor.setIndices(m_index_of_background);
    extractor.filter(*extracted_pcd);

    return extracted_pcd;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr MapManager::extract_removed() {
    PointCloud<PointXYZRGB>::Ptr    extracted_pcd{new PointCloud<PointXYZRGB>()};
    ExtractIndices<PointXYZRGB>     extractor;

    extractor.setInputCloud(m_pcd);
    extractor.setIndices(m_index_of_removed);
    extractor.filter(*extracted_pcd);

    return extracted_pcd;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr MapManager::extract_unknown() {
    PointCloud<PointXYZRGB>::Ptr    extracted_pcd{new PointCloud<PointXYZRGB>()};
    ExtractIndices<PointXYZRGB>     extractor;

    extractor.setInputCloud(m_pcd);
    extractor.setIndices(m_index_of_unknown);
    extractor.filter(*extracted_pcd);

    return extracted_pcd;
}

void MapManager::show_point_cloud() {
    stop_view = false;
    while (!stop_view && !m_viewer->wasStopped()) {
        m_viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::microseconds(100000));
    }
    LOG(INFO) << "finished showing point cloud" << endl;
}

void MapManager::filter_useless_semantics_from_json(const std::string& fn) {
    LOG(INFO) << "read json" << endl;   
    json j_remained_semantic_names;
    ifstream ifn(fn.c_str());
    vector<string> remained_names;
    ifn >> j_remained_semantic_names;

    remained_names.resize(j_remained_semantic_names.size());

    for (int idx = 0; idx < remained_names.size(); idx++) {
        remained_names[idx] = j_remained_semantic_names[idx].get<string>();
    }

    filter_useless_semantics(m_semantic_label, m_semantic_names, remained_names);
    m_semantic_names = remained_names;
}

void MapManager::filter_outliers(float radius, int k) {
    figure_out_landmarks_annotation();
    // PointIndices::Ptr
    
}
}