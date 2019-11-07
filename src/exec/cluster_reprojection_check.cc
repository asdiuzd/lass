#include <iostream>
#include <string>
#include <fstream>
#include "json.h"

using json = nlohmann::json;

int main(int argc, char **argv) {
    json j_centers, j_extrinsics;
    std::string base_dir(argv[1]);
    std::ifstream ifs;
    ifs.open(base_dir + "/id2center.json");
    ifs >> j_centers;
    ifs.close();
    ifs.open(base_dir + "/out_extrinsics.json");
    ifs >> j_extrinsics;
    ifs.close();
    for (const auto &c : j_centers) {
        std::cout << c << "\n";
    }

    return 0;
}
