#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Eigen>

struct PoseData {
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
    std::string filename;
    int index;
};

inline Eigen::Vector3d logmap(const Eigen::Quaterniond &q) {
    Eigen::AngleAxisd aa(q);
    return aa.angle() * aa.axis();
}

inline std::vector<PoseData> read_7scenes_pose(const std::string &path, int pose_num, const std::string &pose_type) {
    std::vector<PoseData> poses;
    // printf("#idx px py pz qx qy qz qw\n");
    for (int idx = 0; idx < pose_num; idx++) {
        char filename[1024];
        if (pose_type == "gt") {
             sprintf(filename, "%s/frame-%06d.pose.txt", path.c_str(), idx);
        } else if (pose_type == "test") {
            sprintf(filename, "%s/frame-%06d.txt", path.c_str(), idx);
        } else {
            std::runtime_error("Unknown pose_type.");
        }
        std::ifstream in(filename);
        Eigen::Matrix4d e;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                in >> e(r, c);
            }
        }
        // origin: Twc
        Eigen::Quaterniond qwc(e.block<3, 3>(0, 0));
        Eigen::Vector3d pwc(e.block<3, 1>(0, 3));
        if (std::abs(qwc.norm() - 1.0) > 1.0e-3) {
           fprintf(stderr, "Warning, rotation matrix may not be valid, qwc.norm() %.4e", qwc.norm());
        }
        if (pose_type == "gt") {
            pwc.x() += 0.0245;
        } else if (pose_type == "test") {
            pwc = -(qwc.conjugate() * pwc);
            qwc = qwc.conjugate();
        } else {
            std::runtime_error("Unknown pose_type.");
        }
        PoseData pose;
        pose.q = qwc;
        pose.p = pwc;
        pose.filename = filename;
        pose.index = idx;
        poses.emplace_back(std::move(pose));
        // printf("%05d %.9e %.9e %.9e %.9e %.9e %.9e %.9e\n", idx, pwc.x(), pwc.y(), pwc.z(), qwc.x(), qwc.y(), qwc.z(), qwc.w());
    }
    return poses;
}

int main(int argc, char **argv) {
    std::string gt_dir = argv[1];
    std::string pred_dir = argv[2];
    const int pose_num = 1000;
    auto gt_poses = read_7scenes_pose(gt_dir, pose_num, "gt");
    auto pred_poses = read_7scenes_pose(pred_dir, pose_num, "test");
    assert(pred_poses.size() == gt_poses.size());
    double APE = 0, ARE = 0, Acount = 0;
    std::vector<double> APEs, AREs;
    for (int i = 0; i < pose_num; ++i) {
        const auto &gt_pose = gt_poses[i];
        const auto &pred_pose = pred_poses[i];
        Eigen::Vector3d p_error = gt_pose.p - pred_pose.p;
        Eigen::Vector3d q_error = logmap(gt_pose.q.conjugate() * pred_pose.q);
        APE += p_error.squaredNorm();
        ARE += q_error.squaredNorm();
        APEs.push_back(p_error.norm() * 1e3);
        AREs.push_back(q_error.norm() * 180 / M_PI);
        Acount++;
    }
    Acount = std::max(Acount, 1.0);
    APE = std::sqrt(APE / Acount) * 1e3;
    ARE = std::sqrt(ARE / Acount) * 180 / M_PI;
    printf("APE RMSE: %.3f [mm]\nARE RMSE %.3f[DEG]\n", APE, ARE);
    std::sort(APEs.begin(), APEs.end());
    std::sort(AREs.begin(), AREs.end());
    printf("APE median: %.3f [mm]\nARE median %.3f[DEG]\n", APEs[APEs.size() / 2], AREs[AREs.size() / 2]);
    return 0;
}
