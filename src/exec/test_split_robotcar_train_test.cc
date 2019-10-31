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
    // read_json
    std::vector<std::string> img_list;
    json j;
    {
        std::ifstream ifs("test_list.json");
        ifs >> j;
        for (auto& e : j) {
            std::string str = e;
            str = str.substr(str.find("/") + 1, 16);
            img_list.push_back(str);
        }
    }
    {
        std::ifstream ifs("train_list.json");
        ifs >> j;
        for (auto& e : j) {
            std::string str = e;
            str = str.substr(str.find("/") + 1, 16);
            img_list.push_back(str);
        }
    }
    std::sort(img_list.begin(), img_list.end());
    img_list.erase(std::unique(img_list.begin(), img_list.end()), img_list.end());

    std::vector<std::string> train_list, test_list;
    for (int idx = 0; idx <img_list.size(); idx++) {
        const std::string &s = img_list[idx];
        if (idx % 5 >= 4) {
            test_list.push_back(s);
        } else {
            train_list.push_back(s);
        }
    }

    ofstream o_train_list{"train_timestamps.json"};
    json j_train_list = train_list;
    o_train_list << std::setw(4) << j_train_list;
    ofstream o_test_list{"test_timestamps.json"};
    json j_test_list = test_list;
    o_test_list << std::setw(4) << j_test_list;
    // std::cout << img_list.size();


    return 0;
}
