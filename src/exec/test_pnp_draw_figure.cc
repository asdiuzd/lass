#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include <experimental/filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "json.h"
#include "reprojection_error_cost.h"

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;
using namespace lass;

#define print_var(x) std::cout << #x << " " << x << std::endl;

struct FrameData {
    FrameData() {
        K.setIdentity();
        qcw_gt.setIdentity();
        qcw.setIdentity();
        pcw_gt.setZero();
        pcw.setZero();
    }
    Eigen::Quaterniond qcw_gt, qcw;
    Eigen::Vector3d pcw_gt, pcw;
    Eigen::Matrix3d K;
    std::vector<Eigen::Vector3d> landmarks;
    std::vector<Eigen::Vector2d> keypoints;
    std::vector<Eigen::Vector2d> projections;
    std::vector<Eigen::Vector2d> gt_projections;
    std::string filename;
    std::vector<int> inlier_indices;
};

inline std::vector<std::string> get_filenames(std::string base_dir) {
    std::vector<std::string> filenames;
    DIR *dir;
    struct dirent *entry;
    if ((dir = opendir(base_dir.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((entry = readdir(dir)) != NULL) {
            // ignore files and hidden folders
            if (entry->d_type == DT_REG && entry->d_name[0] != '.') {
                filenames.emplace_back(entry->d_name);
                // printf("Find files %s\n", entry->d_name);
            }
        }
        closedir(dir);
    }
    return filenames;
}

inline FrameData load_single_from_json(const std::string &filename, std::vector<Eigen::Vector3d> &id2center) {
    std::ifstream ifs(filename);
    json j_data;
    ifs >> j_data;
    ifs.close();
    // std::cout << j_data;
    CHECK(j_data["label"].size() == j_data["uv"].size()) << "size inconsistency";
    FrameData data;
    Eigen::Matrix3d R;
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 4; col++) {
            if (col >= 3) {
                data.pcw_gt(row) = j_data["pose"][row * 4 + col];
            } else {
                R(row, col) = j_data["pose"][row * 4 + col];
            }
        }
    }
    data.qcw_gt = R;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            data.K(row, col) = j_data["camera_k_matrix"][row * 3 + col];
        }
    }
    for (int idx = 0; idx < j_data["label"].size(); idx++) {
        int id = j_data["label"][idx];
        data.landmarks.push_back(id2center[id]);
        data.keypoints.emplace_back(j_data["uv"][idx][0], j_data["uv"][idx][1]);
    }
    data.filename = filename;
    // print_var(data.filename);
    // print_var(data.K);
    // print_var(data.pcw_gt);
    // print_var(data.qcw_gt.coeffs());
    // print_var(data.keypoints.size());
    // print_var(data.landmarks.size());
    return data;
}

inline std::vector<FrameData> load_data_from_json(const std::string &base_dir) {
    std::ifstream cfs(base_dir + "/id2centers.json");
    json j_centers;
    cfs >> j_centers;
    cfs.close();
    std::vector<Eigen::Vector3d> centers(j_centers.size());
    for (int idx = 0; idx < j_centers.size(); idx++) {
        json &j_center = j_centers[idx];
        centers[idx] = {j_center[0], j_center[1], j_center[2]};
    }

    auto filenames = get_filenames(base_dir + "/prediction_json");
    std::sort(filenames.begin(), filenames.end());
    std::vector<FrameData> frames;
    for (const auto &fn : filenames) {
        frames.push_back(load_single_from_json(base_dir + "/prediction_json/" + fn, centers));
    }
    return frames;
}

double compute_reprojection_error(FrameData &frame) {
    double rpe_sum = 0;
    double count = 0;
    frame.projections.clear();
    frame.gt_projections.clear();
    for (const auto &idx : frame.inlier_indices) {
        Eigen::Vector3d pt_c = frame.qcw * frame.landmarks[idx] + frame.pcw;
        Eigen::Vector2d pt_2d = pt_c.hnormalized();
        CHECK(std::abs(frame.K(0, 0) - frame.K(1, 1)) < 1e-5);
        pt_2d = {frame.K(0, 0) * pt_2d.x() + frame.K(0, 2), frame.K(0, 0) * pt_2d.y() + frame.K(1, 2)};
        frame.projections.push_back(pt_2d);
        double rpe = (pt_2d - frame.keypoints[idx]).norm();
        rpe_sum += rpe;
        count++;
        {
            Eigen::Vector3d pt_c = frame.qcw_gt * frame.landmarks[idx] + frame.pcw_gt;
            Eigen::Vector2d pt_2d = pt_c.hnormalized();
            CHECK(std::abs(frame.K(0, 0) - frame.K(1, 1)) < 1e-5);
            pt_2d = {frame.K(0, 0) * pt_2d.x() + frame.K(0, 2), frame.K(0, 0) * pt_2d.y() + frame.K(1, 2)};
            frame.gt_projections.push_back(pt_2d);
        }
    }
    return rpe_sum / count;
}

void solve_pose(FrameData &frame) {
    // get initial pose and inlier points via opencv solvePnPRansac
    std::vector<cv::Point2d> image_pts;
    std::vector<cv::Point3d> world_pts;
    for (int i = 0; i < frame.landmarks.size(); ++i) {
        image_pts.emplace_back(frame.keypoints[i].x(), frame.keypoints[i].y());
        world_pts.emplace_back(frame.landmarks[i].x(), frame.landmarks[i].y(), frame.landmarks[i].z());
    }
    cv::Mat K = cv::Mat::eye(cv::Size(3, 3), CV_64F);
    K.at<double>(0, 0) = frame.K(0, 0);
    K.at<double>(0, 2) = frame.K(0, 2);
    K.at<double>(1, 1) = frame.K(1, 1);
    K.at<double>(1, 2) = frame.K(1, 2);
    cv::Mat rotation, translation;
    std::vector<int> inlier_indices;
    cv::solvePnPRansac(world_pts, image_pts,
                       K, cv::Mat(),
                       rotation, translation, false,
                       200, 8.0, 0.99, inlier_indices,
                       cv::SOLVEPNP_P3P);
    cv::Rodrigues(rotation, rotation);
    Eigen::Matrix3d R;
    cv::cv2eigen(rotation, R);
    frame.qcw = R;
    cv::cv2eigen(translation, frame.pcw);
    // print_var(rotation);
    // print_var(translation);

    // check frustum
    for (auto it = inlier_indices.begin(); it != inlier_indices.end();) {
        Eigen::Vector3d pt_c = frame.qcw * frame.landmarks[*it] + frame.pcw;
        bool is_in_frustum = pt_c.z() > 0;
        Eigen::Vector2d pt_2d = pt_c.hnormalized();
        pt_2d = {frame.K(0, 0) * pt_2d.x() + frame.K(0, 2), frame.K(0, 0) * pt_2d.y() + frame.K(1, 2)};
        if (pt_2d.x() < 0 || pt_2d.x() > 480 || pt_2d.y() < 0 || pt_2d.y() > 270) {
            is_in_frustum = false;
        }
        if (!is_in_frustum) {
            std::cerr << "find pt not in frustum!\n";
            print_var(pt_c);
            print_var(pt_2d);
            it = inlier_indices.erase(it);
        } else {
            ++it;
        }
    }

    // prepare optimize data
    if (inlier_indices.size() < 5) {
        std::cerr << "inliers not enough! current inlier count " << inlier_indices.size() << std::endl;
    }

    frame.inlier_indices = inlier_indices;
    std::vector<Eigen::Vector3d> inlier_landmarks;
    std::vector<Eigen::Vector2d> inlier_keypoints;
    // print_var(inlier_indices.size());
    for (const auto &idx : inlier_indices) {
        inlier_keypoints.push_back(frame.keypoints[idx]);
        inlier_landmarks.push_back(frame.landmarks[idx]);
    }

    // print_var(frame.qcw_gt.coeffs());
    // print_var(frame.qcw.coeffs());
    // print_var(frame.pcw_gt);
    // print_var(frame.pcw);
    // print_var(compute_reprojection_error(frame));
    double rpe_before = compute_reprojection_error(frame);

    // optimize via ceres
    ceres::Problem problem;
    std::array<double, 7> camera = {frame.qcw.w(), frame.qcw.x(), frame.qcw.y(), frame.qcw.z(), frame.pcw.x(), frame.pcw.y(), frame.pcw.z()};
    auto loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *camera_parameterization =
        new ceres::ProductParameterization(
            new ceres::QuaternionParameterization(),
            new ceres::IdentityParameterization(3));
    problem.AddParameterBlock(camera.data(), 7, camera_parameterization);
    for (int i = 0; i < inlier_landmarks.size(); ++i) {
        problem.AddParameterBlock(inlier_landmarks[i].data(), 3);
        problem.SetParameterBlockConstant(inlier_landmarks[i].data());
        ceres::CostFunction *cost_function = ReprojectionErrorWithQuaternions::Create(
            inlier_keypoints[i], frame.K);
        problem.AddResidualBlock(cost_function, loss_function, camera.data(), inlier_landmarks[i].data());
    }
    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options.use_explicit_schur_complement = true;
    // solver_options.minimizer_progress_to_stdout = true;
    solver_options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary solver_summary;
    ceres::Solve(solver_options, &problem, &solver_summary);
    // print_var(camera[0]);
    // print_var(camera[1]);
    // print_var(camera[2]);
    // print_var(camera[3]);
    // print_var(camera[4]);
    // print_var(camera[5]);
    // print_var(camera[6]);
    // std::cout << solver_summary.BriefReport();
    // assign valie
    frame.pcw = {camera[4], camera[5], camera[6]};
    frame.qcw = Eigen::Quaterniond(camera[0], camera[1], camera[2], camera[3]);
    double rpe_after = compute_reprojection_error(frame);
    printf("reprojection error : %.4f -> %.4f\n", rpe_before, rpe_after);
}

void show_inlier_detection(FrameData &f, std::string filename) {
    cv::Mat img = cv::imread(filename);
    std::vector<Eigen::Vector2d> inlier_keypoints;
    for (const auto &idx : f.inlier_indices) {
        inlier_keypoints.push_back(f.keypoints[idx]);
    }
    CHECK(f.gt_projections.size() == inlier_keypoints.size());
    for (int i = 0; i < f.gt_projections.size(); ++i) {
        cv::Point2f pt2(float(inlier_keypoints[i].x()), float(inlier_keypoints[i].y()));
        cv::drawMarker(img, pt2, cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 13, 2, cv::LINE_AA);
        Eigen::Vector2d pt_2d = f.gt_projections[i];
        cv::Point2f pt(float(pt_2d.x()), float(pt_2d.y()));
        // cv::drawMarker(img, pt, cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 13, 2, cv::LINE_AA);
        cv::circle(img, pt, 5, cv::Scalar(0, 200, 0), 1, cv::LINE_AA);
    }
    cv::imshow("img", img);
    cv::waitKey(0);
    cv::imwrite("inlier_detection.png", img);
}

void show_keypoint_vertex(FrameData &f, std::string filename) {
    cv::Mat img = cv::imread(filename);
    std::vector<Eigen::Vector2d> inlier_keypoints;
    for (const auto &idx : f.inlier_indices) {
        inlier_keypoints.push_back(f.keypoints[idx]);
    }
    for (int i = 0; i < f.keypoints.size(); ++i) {
        Eigen::Vector2d pt_2d = f.keypoints[i];
        cv::Point2f pt(float(pt_2d.x()), float(pt_2d.y()));
        cv::circle(img, pt, 5, cv::Scalar(30, 0, 0), 1, cv::LINE_AA);
    }
    cv::imshow("img", img);
    cv::waitKey(0);
    cv::imwrite("keypoint_vertex.png", img);
}

int main(int argc, char **argv) {
    std::string base_dir = argv[1];
    auto frames = load_data_from_json(base_dir);
    std::vector<double> APEs, AREs;

    for (auto &f : frames) {
        if (f.filename.find("l2fgp000000012.json") == std::string::npos) continue;
        // solve and optimize
        solve_pose(f);
        // evaluate
        if (f.inlier_indices.empty()) {
            APEs.push_back(std::numeric_limits<double>::max());
            AREs.push_back(std::numeric_limits<double>::max());
            continue;
        }
        APEs.push_back((f.pcw - f.pcw_gt).norm() * 1e3);
        Eigen::Quaterniond q_diff = f.qcw * f.qcw_gt.conjugate();
        Eigen::AngleAxisd aa(q_diff);
        AREs.push_back(aa.angle() * 180 / M_PI);

        show_inlier_detection(f, "/home/ybbbbt/Data/prediction_dump/refined_voting/prediction/000000012.fig.image.jpg");
        show_keypoint_vertex(f, "/home/ybbbbt/Dropbox/LandmarkSegmentation/SyncSnippt/selected_figure/refined/000000012.fig.voting.pred.jpg");
    }
    std::sort(APEs.begin(), APEs.end());
    std::sort(AREs.begin(), AREs.end());
    printf("APE median: %.3f [mm]\nARE median %.3f[DEG]\n", APEs[APEs.size() / 2], AREs[AREs.size() / 2]);
    return 0;
}
