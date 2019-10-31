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
    int ret = system("rm -rf fill_hole && mkdir -p fill_hole/left fill_hole/right fill_hole/rear");
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
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < img_list.size(); ++i) {
        std::string name = img_list[i];
        cv::Mat img = cv::imread(name);
        fillHoles_fast(img);
        // assign color for better visualization (because we don't need blue channel)
        for (int x = 0; x < img.rows; ++x) {
            for (int y = 0; y < img.cols; ++y) {
                auto &c = img.at<cv::Vec3b>(x, y);
                if (c[0] == 0 && c[1] == 0 && c[2] == 0) continue;
                c[0] = 120;
            }
        }
        name = "fill_hole/" + name;
        cv::imwrite(name, img);
        static int count = 0;
        if (count % 10 == 0) {
            fprintf(stdout, "\r%d / %zu", count, img_list.size());
            fflush(stdout);
        }
        count++;
    }

    return 0;
}
