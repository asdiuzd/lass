#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>
#include <omp.h>
#include "utils.h"
#include "json.h"

using json = nlohmann::json;
using namespace lass;

int main() {
    const std::string src_prefix = "/home/ybbbbt/Data/cambridge/ShopFacade/";
    const std::string seg_prefix = "/home/ybbbbt/Developer/lass/bin/cambridge_raycast/";
    int ret = system("rm -rf blend_color && mkdir -p blend_color/seq1 blend_color/seq2 blend_color/seq3");
    // read_json
    std::vector<std::string> img_list;
    json j;
    {
        std::ifstream ifs("test_list.json");
        ifs >> j;
        for (auto& e : j) {
            img_list.push_back(e);
        }
    }
    {
        std::ifstream ifs("train_list.json");
        ifs >> j;
        for (auto& e : j) {
            img_list.push_back(e);
        }
    }

    // file holes
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < img_list.size(); ++i) {
        std::string name = img_list[i];
        // std::cout << name << std::endl;
        cv::Mat img_src = cv::imread(src_prefix + "/" + name.substr(0, name.length() - 3) + "png");
        cv::Mat img_cast = cv::imread(seg_prefix + "/" + name);
        cv::resize(img_src, img_src, cv::Size(960, 540));
        cv::resize(img_cast, img_cast, cv::Size(960, 540));
        cv::Mat img_blend;
        cv::addWeighted(img_src, 0.5, img_cast, 0.5, 0.0, img_blend);
        
        name = "blend_color/" + name;
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
