#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>
#include <omp.h>
#include "utils.h"
#include "json.h"

using json = nlohmann::json;
using namespace lass;

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
    const std::string src_prefix = argv[1];
    const std::string seg_prefix = argv[2];
    const std::string json_base_dir = argv[3];
    const std::string src_img_format = argv[4];
    // read_json
    std::vector<std::string> img_list;
    json j;
    {
        std::ifstream ifs(json_base_dir + "/test_list.json");
        ifs >> j;
        for (auto& e : j) {
            img_list.push_back(e);
        }
    }
    {
        std::ifstream ifs(json_base_dir + "/train_list.json");
        ifs >> j;
        for (auto& e : j) {
            img_list.push_back(e);
        }
    }

    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < img_list.size(); ++i) {
        std::string name = img_list[i];
        // std::cout << name << std::endl;
        cv::Mat img_src = cv::imread(src_prefix + "/" + name.substr(0, name.length() - 3) + src_img_format);
        cv::Mat img_cast = cv::imread(seg_prefix + "/" + name);
        repaint_color(img_cast);
        auto sz = cv::Size(img_src.cols / 2, img_src.rows / 2);
        cv::resize(img_src, img_src, sz);
        cv::resize(img_cast, img_cast, sz);
        cv::Mat img_blend;
        cv::addWeighted(img_src, 0.5, img_cast, 0.5, 0.0, img_blend);
        
        name = seg_prefix + "/../blend_color/" + name;
        name = name.substr(0, name.length() - 3) + "jpg";
        cv::imwrite(name, img_blend);
        static int count = 0;
        if (count % 10 == 0) {
            fprintf(stdout, "\r%d / %zu", count, img_list.size());
            fflush(stdout);
        }
        count++;
    }

    return 0;
}
