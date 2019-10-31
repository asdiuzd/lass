#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <opencv2/opencv.hpp>

using Clock = std::chrono::high_resolution_clock;
#define print_var(x) std::cout << #x << " " << x << std::endl;

void fillHoles_fast(cv::Mat &img) {
    cv::Mat img_filled = img.clone();
    const int min_conscultive_angle_thresh = 180;
    const int distance_thresh = 25;
    const int distance_thresh_2 = distance_thresh * distance_thresh;
    const int bin_num = 72;
    const int bin_angle = 360 / bin_num;
    assert(std::abs((360.0 / bin_num) - bin_angle) < 1.0e-5);

    std::vector<int> bins(bin_num);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            if (img.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {
                const int i_lower = std::max(0, i - distance_thresh);
                const int i_upper = std::min(img.rows - 1, i + distance_thresh);
                const int j_lower = std::max(0, j - distance_thresh);
                const int j_upper = std::min(img.cols - 1, j + distance_thresh);

                bins.assign(bins.size(), std::numeric_limits<int>::max());
                int min_distance_2 = std::numeric_limits<int>::max();
                std::pair<int, int> closest_pos;
                for (int ii = i_lower; ii < i_upper; ++ii) {
                    for (int jj = j_lower; jj < j_upper; ++jj) {
                        if (img.at<cv::Vec3b>(ii, jj) == cv::Vec3b(0, 0, 0)) continue;
                        int distance_2 = (ii - i) * (ii - i) + (jj - j) * (jj - j);
                        if (distance_2 > distance_thresh_2) continue;
                        int y = ii - i, x = jj - j;
                        double theta = (atan2(y, x) + M_PI) * 180 / M_PI;
                        int bin_idx = theta / bin_angle;
                        bin_idx = std::min(bin_idx, int(bins.size() - 1));
                        bins[bin_idx] = std::min(bins[bin_idx], distance_2);
                        if (distance_2 < min_distance_2) {
                            min_distance_2 = distance_2;
                            closest_pos = {ii, jj};
                        }
                    }
                }
                int cnt = 0;
                int max_cnt = 0;
                for (int i = 0; i < bin_num * 2; ++i) {
                    if (bins[i % bin_num] != std::numeric_limits<int>::max()) {
                        ++cnt;
                        max_cnt = std::max(cnt, max_cnt);
                    } else {
                        cnt = 0;
                    }
                }
                int max_angle = std::max(max_angle, max_cnt * bin_angle);
                if (max_angle > min_conscultive_angle_thresh) {
                    img_filled.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(closest_pos.first, closest_pos.second);
                }
            }
        }
    }
    img = img_filled;
}

int main(int argc, char **argv) {
    cv::Mat img = cv::imread(argv[1]);
    cv::imshow("before", img);

    auto t_start = Clock::now();
    fillHoles_fast(img);
    auto t_end = Clock::now();
    double t_loop = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    printf("cpp time %.3fms\n", t_loop * 1000);
    cv::imshow("after", img);
    cv::waitKey(0);
    return 0;
}
