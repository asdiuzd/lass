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
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "MapManager.h"

using namespace std;
using namespace pcl;
using namespace cv;
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
    } else if (event.getKeySym() == "n" && event.keyDown()) {
        stop_view = true;
    } else if (event.getKeySym() == "t" && event.keyDown()) {
        // LOG(INFO) << "Toggle use show flag" << endl;
        // maybe we need to use singleton
        // m_view_type = (++m_view_type) % 3;
        
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
    m_viewer->setBackgroundColor (0, 0, 0);
    switch (m_view_type) {
        case 0: {
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(m_pcd);
            m_viewer->addPointCloud<pcl::PointXYZRGB>(m_pcd, rgb, "pcd");
            m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "pcd");
            break;
        }
        case 1: {
            m_viewer->addPointCloud (m_target_pcd, "target_pcd");
            m_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "target_pcd");
            m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_pcd");
            break;
        }
        case 2: {
            m_viewer->addPointCloud (m_labeled_pcd, "labeled_pcd");
            m_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "labeled_pcd");
            m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "labeled_pcd");
            break;
        }
        default: {

        }
    }
    if (m_show_camera_extrinsics) {
        update_camera_trajectory_to_viewer();
    }
}

MapManager::MapManager():
    m_pcd(new PointCloud<PointXYZRGB>()),
    m_octree(1.0f),
    m_viewer(new visualization::PCLVisualizer()) {
    initialize_viewer();
}

MapManager::MapManager(const std::string& dir):
    m_pcd(new PointCloud<PointXYZRGB>()),
    m_octree(1.0f),
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

void MapManager::load_ply_pcl(const std::string& fn) {
    CHECK(io::loadPLYFile(fn.c_str(), *m_pcd) >= 0) << "can not load: " << fn << endl;
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
    LOG(INFO) << "max cluster label = " << max_cluster_label << endl;

    for (int idx = 0; idx < m_pcd->points.size(); idx++) {
        auto &pt = m_pcd->points[idx];
        auto &cluster = m_cluster_label[idx];

        if (cluster == -1) {
            pt.r = pt.g = pt.b = 40;
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

void MapManager::dye_through_render() {
    for (int idx = 0; idx < m_pcd->points.size(); idx++) {
        auto &pt = m_pcd->points[idx];

        auto label = m_render_label[idx];
        GroundColorMix(pt.r, pt.g, pt.b, normalize_value(label, 0, LandmarkType::NUMBER));
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

void MapManager::prepare_octree_for_target_pcd(float resolution) {
    cout << "Construct octree" << endl;
    m_octree.setResolution(resolution);
    m_octree.setInputCloud(m_target_pcd);
    m_octree.addPointsFromInputCloud();
}

// extrinsics: world to camera
void MapManager::raycasting_pcd(const Eigen::Matrix4f& extrinsics, const camera_intrinsics& intrinsics, pcl::PointCloud<pcl::PointXYZL>::Ptr& pcd, bool depthDE, int stride, float scale, const std::string &raycast_pcd_type) {
    // LOG(INFO) << "start raycasting" << endl;
    // const int scale = 4;
    // const int width = 1024 / scale, height = 1024 / scale;
    // const float fx = 400 / scale, fy = 400 / scale;
    // const float cx_left = 500.107605 / scale, cy_left = 511.461426 / scale;
    // const float cx_rear = 508.222931 / scale, cy_rear = 498.187378 / scale;
    // const float cx_right = 502.503754 / scale, cy_right = 490.259033 / scale;
    const int &width = intrinsics.width, &height = intrinsics.height;
    const float &fx = intrinsics.fx, &fy = intrinsics.fy, &cx = intrinsics.cx, &cy = intrinsics.cy;
    const int pixel_number = width * height;
    pcd->points.resize(width * height) ;

    Eigen::Vector3f origin = - extrinsics.block(0, 0, 3, 3).transpose() * extrinsics.block(0, 3, 3, 1);
    vector<vector<Eigen::Vector3f>> directions(width, vector<Eigen::Vector3f>(height));
    vector<double> depth(width * height, 999999);

    int hit_count = 0;
    
    pcl::PointCloud<pcl::PointXYZL>::Ptr raycast_pcd;

    if (raycast_pcd_type == "labeled") {
        raycast_pcd = m_labeled_pcd;
    } else if (raycast_pcd_type == "target") {
        raycast_pcd = m_target_pcd;
    }

    for (int u = 0; u < width; u++) {
        for (int v = 0; v < height; v++) {
            auto &d = directions[u][v];
            vector<int> k_indices;
            Eigen::Vector3f dc((u - cx) / fx, (v - cy) / fy, 1);
            d = extrinsics.block(0, 0, 3, 3).transpose() * dc + origin;

            auto &pt = pcd->points[v * width + u];
            pt.x = d(0);
            pt.y = d(1);
            pt.z = d(2);
            d = d - origin;
            d.normalize();

            m_octree.getIntersectedVoxelIndices(origin, d, k_indices, 1);
            if (k_indices.size() > 0) {
                auto &idx = k_indices[0];
                if (idx >= raycast_pcd->points.size()) {
                    LOG(INFO) << "Alert! " << idx << endl;
                }
                hit_count++;
                pt.label = raycast_pcd->points[idx].label;
                depth[v * width + u] = euclidean_distance(
                    raycast_pcd->points[idx].x - origin[0],
                    raycast_pcd->points[idx].y - origin[1],
                    raycast_pcd->points[idx].z - origin[2]
                );
            } else {
                pt.label = 0;
            }
        }
    }

    if (depthDE) {
        depth_based_DE(pcd, depth, intrinsics, stride, scale);
    }
}

void MapManager::grid_landmark_clustering() {
    LOG(INFO) << "grid landmark clustering" << endl;

    const double grid_resolution = 5.0;
}

void MapManager::euclidean_landmark_clustering() {
    LOG(INFO) << "euclidean landmark clustering" << endl;

    LOG(INFO) << "Build Kdtree" << endl;
    search::KdTree<PointXYZRGB>::Ptr tree(new search::KdTree<PointXYZRGB>);
    tree->setInputCloud(m_pcd);
    EuclideanClusterExtraction<PointXYZRGB> ec;
    cout << "Clustering extracting" << endl;
    ec.setClusterTolerance(0.4);
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(10000000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(m_pcd);
    ec.setIndices(m_index_of_landmark);
    ec.extract(m_clusters);
    cout << "Total: " << m_clusters.size() << " clusters" << endl;
    cout << "Labeling" << endl;
    
    int tot_clusters = 0;
    m_cluster_label.resize(0);

    m_cluster_label.resize(m_pcd->points.size(), -1);
    int outlier_clusters = 0;
    for (int label = 0; label < m_clusters.size(); label++) {
        auto &indices = m_clusters[label];
        for (auto &idx: indices.indices) {
            m_cluster_label[idx] = label;
        }
    }
    cout << "Total invalid clusters: " << outlier_clusters << endl;
}

void MapManager::supervoxel_landmark_clustering(float voxel_resolution, float seed_resolution, float spatial_importance, float color_importance, float normal_importance) {
    LOG(INFO) << "super voxel landmark clustering" << endl;

    // const float voxel_resolution = 1.0, seed_resolution = 21.0, spatial_importance = 1;
    SupervoxelClustering<PointXYZRGB> super(voxel_resolution, seed_resolution);
    map<uint32_t, Supervoxel<PointXYZRGB>::Ptr> supervoxel_clusters;

    super.setInputCloud(extract_landmarks());
    super.setColorImportance(color_importance);
    super.setNormalImportance(normal_importance);
    super.setSpatialImportance(spatial_importance);
    super.extract(supervoxel_clusters);
    LOG(INFO) << "super voxel cluster size = " << supervoxel_clusters.size() << endl;

    m_target_pcd = super.getLabeledVoxelCloud();
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr MapManager::extract_points(pcl::PointIndices::Ptr indices) {
    PointCloud<PointXYZRGB>::Ptr    extracted_pcd{new PointCloud<PointXYZRGB>()};
    ExtractIndices<PointXYZRGB>     extractor;

    extractor.setInputCloud(m_pcd);
    extractor.setIndices(indices);
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
    // dye_through_landmarks();
    pcl::PointCloud<PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<PointXYZRGB>);
    PointIndices::Ptr concerned_indices, filtered_indices{new PointIndices};
    PointIndices::Ptr filtered_landmark_indices, filtered_background_indices, remained_indices{new PointIndices};

    RadiusOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud(m_pcd);
    // landmark filtering
    concerned_indices = m_index_of_landmark;

    sor.setIndices(concerned_indices);
    sor.setMinNeighborsInRadius(k);
    sor.setRadiusSearch(radius);

    sor.filter(filtered_indices->indices);
    concerned_indices = filtered_indices;
    filtered_indices = PointIndices::Ptr{new PointIndices};
    filtered_landmark_indices = concerned_indices;

    concerned_indices = m_index_of_background;
    sor.setIndices(concerned_indices);
    sor.setMinNeighborsInRadius(k);
    sor.setRadiusSearch(radius);

    sor.filter(filtered_indices->indices);
    filtered_background_indices = filtered_indices;

    auto& indices = remained_indices->indices;
    indices.insert(indices.end(), filtered_landmark_indices->indices.begin(), filtered_landmark_indices->indices.end());
    indices.insert(indices.end(), filtered_background_indices->indices.begin(), filtered_background_indices->indices.end());
    m_render_label.resize(indices.size(), LandmarkType::LANDMARK);

    m_index_of_landmark->indices.resize(filtered_landmark_indices->indices.size());
    for (int idx = 0; idx < m_index_of_landmark->indices.size(); idx++) {
        m_index_of_landmark->indices[idx] = idx;
    }

    m_index_of_background->indices.resize(filtered_background_indices->indices.size());
    int base_idx = m_index_of_landmark->indices.size();
    for (int idx = 0; idx < m_index_of_background->indices.size(); idx++) {
        m_index_of_background->indices[idx] = base_idx + idx;
        m_render_label[m_index_of_background->indices[idx]] = LandmarkType::BACKGROUND;
    }

    PointCloud<PointXYZRGB>::Ptr    extracted_pcd{new PointCloud<PointXYZRGB>()};
    ExtractIndices<PointXYZRGB>     extractor;

    extractor.setInputCloud(m_pcd);
    extractor.setIndices(remained_indices);
    extractor.filter(*extracted_pcd);

    m_pcd = extracted_pcd;
}

void MapManager::filter_landmarks_through_background(int knn, float ratio) {
    PointXYZRGB min_pt, max_pt;
    getMinMax3D(*m_pcd, min_pt, max_pt);
    float resolution = 1.0;
    int width = (max_pt.x - min_pt.x) / resolution + 1, height = (max_pt.y - min_pt.y) / resolution + 1;
    vector<vector<unsigned char>> map(width, vector<unsigned char>(height, 0));
    /*
        0 - no road
        1 - road block on the boundary
        2 - road block in the center
    */
   LOG(INFO) << "width = " << width << endl;
   LOG(INFO) << "height = " << height << endl;

   for (int idx = 0; idx < m_pcd->points.size(); idx++) {
       auto& pt = m_pcd->points[idx];
       auto& label = m_render_label[idx];
       if (label == LandmarkType::LANDMARK) {
           continue;
       }

       int x = (pt.x - min_pt.x) / resolution, y = (pt.y - min_pt.y) / resolution;
       map[x][y] = 1;
   }

   for (int x = 1; x < width - 1; x++) {
       for (int y = 1; y < height - 1; y++) {
           if (map[x-1][y] != 0 && map[x+1][y] != 0 && map[x][y-1] != 0 && map[x][y+1] != 0) {
               map[x][y] = 2;
           }
       }
   }

   m_index_of_landmark->indices.resize(0);
   for (int idx = 0; idx < m_pcd->points.size(); idx++) {
       auto& pt = m_pcd->points[idx];
       auto& label = m_render_label[idx];
       if (label == LandmarkType::BACKGROUND) {
           continue;
       }

       int x = (pt.x - min_pt.x) / resolution, y = (pt.y - min_pt.y) / resolution;
    //    label = map[x][y];
       if (map[x][y] == 2) {
           label = LandmarkType::HIGHLIGHT;
       } else {
           m_index_of_landmark->indices.emplace_back(idx);
       }
   }
}

void MapManager::filter_supervoxels_through_background(const std::string &pcd_type) {
    LOG(INFO) << __func__ << endl;
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcd_ptr;
    if (pcd_type == "labeled") {
        pcd_ptr = m_labeled_pcd;
    } else if (pcd_type == "target") {
        pcd_ptr = m_target_pcd;
    }
    auto background_pcd = extract_background();
    PointXYZL min_pt, max_pt;
    getMinMax3D(*pcd_ptr, min_pt, max_pt);
    float resolution = 1.0;
    int width = (max_pt.x - min_pt.x) / resolution + 1, height = (max_pt.y - min_pt.y) / resolution + 1;
    vector<vector<unsigned char>> map(width, vector<unsigned char>(height, 0));
    /*
        0 - no road
        1 - road block on the boundary
        2 - road block in the center
    */
   LOG(INFO) << "width = " << width << endl;
   LOG(INFO) << "height = " << height << endl;

   for (int idx = 0; idx < background_pcd->points.size(); idx++) {
       auto& pt = background_pcd->points[idx];
       int x = (pt.x - min_pt.x) / resolution, y = (pt.y - min_pt.y) / resolution;
       map[x][y] = 1;
   }

   for (int x = 1; x < width - 1; x++) {
       for (int y = 1; y < height - 1; y++) {
           if (map[x-1][y] != 0 && map[x+1][y] != 0 && map[x][y-1] != 0 && map[x][y+1] != 0) {
               map[x][y] = 2;
           }
       }
   }

   PointIndices::Ptr valid_indices{new PointIndices};
   for (int idx = 0; idx < pcd_ptr->points.size(); idx++) {
       auto& pt = pcd_ptr->points[idx];

       int x = (pt.x - min_pt.x) / resolution, y = (pt.y - min_pt.y) / resolution;
       if (map[x][y] == 2) {
           pt.label = max_target_label;
       } else {
           valid_indices->indices.emplace_back(idx);
       }
   }

    PointCloud<PointXYZL>::Ptr    valid_pts{new PointCloud<PointXYZL>()};
    ExtractIndices<PointXYZL>     extractor;

    extractor.setInputCloud(pcd_ptr);
    extractor.setIndices(valid_indices);
    extractor.filter(*valid_pts);
    if (pcd_type == "labeled") {
        m_labeled_pcd = valid_pts;
    } else if (pcd_type == "target") {
        m_target_pcd = valid_pts;
    }
}

void MapManager::filter_minor_segmentations(int number_threshold, const std::string &pcd_type) {
    LOG(INFO) << __func__ << endl;
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcd_ptr;
    if (pcd_type == "labeled") {
        pcd_ptr = m_labeled_pcd;
    } else if (pcd_type == "target") {
        pcd_ptr = m_target_pcd;
    }

    int max_label = std::max_element(
                        pcd_ptr->points.begin(),
                        pcd_ptr->points.end(),
                        [&](const PointXYZL &p1, const PointXYZL &p2) {
                            return p1.label < p2.label;
                        })
                        ->label + 1;

    vector<int> label_counter(max_label, 0), label_mapping(max_label, 0);
    for (auto& p: pcd_ptr->points) {
        label_counter[p.label]++;
    }

    int zero_clusters = std::count(label_counter.begin(), label_counter.end(), 0);

    LOG(INFO) << "0 - " << label_counter[0] << endl;
    LOG(INFO) << "1 - " << label_counter[1] << endl;

    int counter = 1;
    for (int label = 0; label < label_counter.size(); label++) {
        if (label_counter[label] > number_threshold) {
            label_mapping[label] = counter++;
        }
    }
    max_target_label = counter;

    LOG(INFO) << "max label before fitering = " << max_label << endl;
    LOG(INFO) << "max label after fitering = " << counter << endl;
    LOG(INFO) << "number of clusters of zero size = " << zero_clusters << endl;
    LOG(INFO) << "size of label 0 = " << label_counter[0] << endl;

    PointCloud<PointXYZL>::Ptr remained_pts{new PointCloud<PointXYZL>};
    // remained_pts->points.reserve(pcd_ptr->points.size());

    for (auto& p:pcd_ptr->points) {
        auto& label = p.label;
        if (label_counter[label] > number_threshold) {
            label = label_mapping[label];
            remained_pts->points.push_back(p);
        }
    }

    if (pcd_type == "labeled") {
        m_labeled_pcd = remained_pts;
    } else if (pcd_type == "target") {
        m_target_pcd = remained_pts;
    }
}

void MapManager::filter_points_near_cameras(float radius) {
    LOG(INFO) << __func__ << endl;
    if (m_camera_extrinsics.empty()) {
        LOG(ERROR) << "m_camera_extrinsics is empty, filter_points_near_cameras would do nothing!";
        return;
    }
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(m_pcd);
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
    extract.setInputCloud(m_pcd);
    extract.setIndices(outliers);
    extract.setNegative(true);
    extract.filter(*m_pcd);
    LOG(INFO) << "outlier_indices size " << outlier_indices_set.size();
}

void MapManager::update_camera_trajectory_to_viewer() {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cam_centers(new PointCloud<PointXYZRGB>());
    assert(m_camera_types.size() == m_camera_extrinsics.size());
    for (size_t i = 0; i < m_camera_extrinsics.size(); ++i) {
        Eigen::Matrix4f e = m_camera_extrinsics[i];
        int camera_type = m_camera_types[i];
        // apply rotation fix to extrinsics
        e(1, 3) *= -1; e(2, 3) *= -1;
        e(0, 1) *= -1; e(0, 2) *= -1;
        e(1, 0) *= -1; e(2, 0) *= -1;
        Eigen::Vector3f pcw = e.block<3, 1>(0, 3);
        Eigen::Quaternionf qcw(e.block<3, 3>(0, 0));
        Eigen::Vector3f pwc = -(qcw.conjugate() * pcw);
        pcl::PointXYZRGB pt;
        pt.x = pwc.x();
        pt.y = pwc.y();
        pt.z = pwc.z();
        pt.r = pt.g = pt.b = 200;
        cam_centers->points.emplace_back(pt);
        for (int j = 1; j < 5; ++j) {
            Eigen::Vector3f pt_forward = qcw.conjugate() * Eigen::Vector3f(0, 0, 0.1 * j) + pwc;
            Eigen::Vector3i color;
            color.setZero();
            assert(camera_type < 3 && camera_type >= 0);
            color(camera_type) = 200;
            pcl::PointXYZRGB pt;
            pt.x = pt_forward.x();
            pt.y = pt_forward.y();
            pt.z = pt_forward.z();
            pt.r = color.x();
            pt.g = color.y();
            pt.b = color.z();
            cam_centers->points.emplace_back(pt);
        }
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> colors(cam_centers);
    m_viewer->addPointCloud(cam_centers, colors, "cam_centers");
    m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cam_centers");
}

void MapManager::assign_supervoxel_label_to_filtered_pcd() {
    LOG(INFO) << __func__ << endl;
    m_labeled_pcd.reset(new pcl::PointCloud<PointXYZL>());
    pcl::KdTreeFLANN<pcl::PointXYZL> kdtree;
    kdtree.setInputCloud(m_target_pcd);
    int K = 1;
    std::vector<int> point_indices(K);
    std::vector<float> distances(K);
    for (int i = 0; i < m_pcd->points.size(); ++i) {
        const auto &orig_pt = m_pcd->points[i];
        pcl::PointXYZL pt;
        pt.x = orig_pt.x;
        pt.y = orig_pt.y;
        pt.z = orig_pt.z;
        if (kdtree.nearestKSearch(pt, K, point_indices, distances) > 0) {    
            pt.label = m_target_pcd->points[point_indices[0]].label;
            m_labeled_pcd->points.emplace_back(pt);
        }
    }
}

}
