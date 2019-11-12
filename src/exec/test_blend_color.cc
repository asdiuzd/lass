#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>
#include <omp.h>
#include <Eigen/Eigen>
#include "utils.h"
#include "json.h"

using json = nlohmann::json;
using namespace lass;

struct CameraData {
    CameraData() {
        qcw.setIdentity();
        pcw.setZero();
        focal = 0;
    }
    std::string filename;
    Eigen::Quaterniond qcw;
    Eigen::Vector3d pcw;
    double focal;
};

inline void repaint_color(cv::Mat &img) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            auto &c = img.at<cv::Vec3b>(i, j);
            if (c[1] == 0 && c[2] == 0) continue;
            uint32_t unique_key = (uint32_t(c[0]) << 16) + (uint32_t(c[1]) << 8) + uint32_t(c[2]);
            hash_colormap(c[0], c[1], c[2], unique_key);
        }
    }
}

// args path_to_source_images path_to_raycast_images path_to_tran_test_list_json_base_dir src_img_format(jpg or png)
int main(int argc, char **argv) {
    const std::string dataset_base_dir = argv[1];
    const std::string segmentation_base_dir = argv[2];
    const std::string json_base_dir = argv[3];
    const std::string src_img_format = argv[4];
    // read cameras
    std::vector<CameraData> camera_list;
    {
        json j;
        std::ifstream ifs(json_base_dir + "/out_extrinsics.json");
        ifs >> j;
        for (auto &e : j.items()) {
            CameraData cd;
            cd.filename = e.key();
            Eigen::Matrix4d Tcw;
            auto &m = e.value();
            Tcw << m[0], m[1], m[2], m[3],
                m[4], m[5], m[6], m[7],
                m[8], m[9], m[10], m[11],
                0, 0, 0, 1;
            cd.qcw = Eigen::Quaterniond(Tcw.block<3, 3>(0, 0));
            cd.pcw = Eigen::Vector3d(Tcw.block<3, 1>(0, 3));
            if (e.value().size() == 13) {
                cd.focal = e.value()[12];
            }
            camera_list.emplace_back(std::move(cd));
        }
    }
    // read cluster centers
    std::vector<Eigen::Vector3d> centers;
    {
        json j;
        std::ifstream ifs(json_base_dir + "/id2centers.json");
        ifs >> j;
        for (auto &e : j.items()) {
            centers.emplace_back(e.value()[0], e.value()[1], e.value()[2]);
        }
        print_var(centers.size());
    }
#pragma omp parallel for num_threads(16)
    for (int i = 0; i < camera_list.size(); ++i) {
        const auto &cam = camera_list[i];
        std::string name = cam.filename;
        // std::cout << name << std::endl;
        cv::Mat img_src = cv::imread(dataset_base_dir + "/" + name.substr(0, name.length() - 3) + src_img_format);
        cv::Mat img_cast = cv::imread(segmentation_base_dir + "/" + name);
        cv::Mat img_orig_seg = img_cast.clone();
        // blend color
        repaint_color(img_cast);
        double cx = img_src.cols / 2, cy = img_src.rows / 2;
        int new_width = img_src.cols / 2, new_height = img_src.rows / 2;
        auto sz = cv::Size(new_width, new_height);
        cv::resize(img_src, img_src, sz);
        cv::resize(img_cast, img_cast, sz);
        cv::resize(img_orig_seg, img_orig_seg, sz);
        cv::Mat img_blend;
        cv::addWeighted(img_src, 0.5, img_cast, 0.5, 0.0, img_blend);
        // project to images
        for (int idx_center = 1; idx_center < centers.size(); idx_center++) {
            Eigen::Vector3d &center = centers[idx_center];
            Eigen::Vector3d pt_c = cam.qcw * center + cam.pcw;
            if (pt_c.z() < 0) continue;
            Eigen::Vector2d pt_2d = pt_c.hnormalized();
            pt_2d = {pt_2d.x() * cam.focal + cx, pt_2d.y() * cam.focal + cy};
            pt_2d /= 2.0;
            if (pt_2d.x() < 0 || pt_2d.x() > new_width - 1 || pt_2d.y() < 0 || pt_2d.y() > new_height - 1) continue;
            // check if label consist with segmentation (remove invisible projections)
            auto &color = img_orig_seg.at<cv::Vec3b>(std::round(pt_2d.y()), std::round(pt_2d.x()));
            int label = int(color[1]) * 256 + int(color[2]);
            if (label != idx_center) continue;
            cv::Point2f pt(float(pt_2d.x()), float(pt_2d.y()));
            // cv::circle(img_blend, pt, 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::drawMarker(img_blend, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 15, 1, cv::LINE_AA);
        }

        name = segmentation_base_dir + "/../blend_color/" + name;
        name = name.substr(0, name.length() - 3) + "jpg";
        cv::imshow("vis", img_blend);
        cv::waitKey(0);
        int ret = system(("mkdir -p " + name.substr(0, name.find_last_of("/"))).c_str());
        cv::imwrite(name, img_blend);
        static int count = 0;
        if (count % 10 == 0) {
            fprintf(stdout, "\r%d / %zu", count, camera_list.size());
            fflush(stdout);
        }
        count++;
    }

    return 0;
}
