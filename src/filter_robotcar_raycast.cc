#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

void filter_few_colors(cv::Mat &img, int few_color_threshold = 25) {
    cv::imshow("before", img);
    // construct color count map
    std::map<int, int> color_count_map;
    for (int i = 0; i < img.cols; ++i) {
        for (int j = 0; j < img.rows; ++j) {
            cv::Vec3b &c = img.at<cv::Vec3b>(j, i);
            int color_key = c[0] * 255 * 255 + c[1] * 255 + c[2];
            if (color_count_map.count(color_key) == 0) {
                color_count_map[color_key] = 1;
            } else {
                color_count_map[color_key]++;
            }
        }
    }
    // find few color color_key
    std::set<int> few_color_keys;
    for (const auto &p : color_count_map) {
        if (p.second <= few_color_threshold) few_color_keys.insert(p.first);
    }

    // filter out few colors
    for (int i = 0; i < img.cols; ++i) {
        for (int j = 0; j < img.rows; ++j) {
            cv::Vec3b &c = img.at<cv::Vec3b>(j, i);
            int color_key = c[0] * 255 * 255 + c[1] * 255 + c[2];
            if (few_color_keys.count(color_key) > 0) {
                c[0] = c[1] = c[2] = 0;
            }
        }
    }
    cv::imshow("after", img);
    cv::waitKey(0);
}

int main(int argc, char **argv) {
    cv::Mat img = cv::imread(argv[1]);
    filter_few_colors(img, 16);
    return 0;
}
