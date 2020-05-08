#pragma once

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <Eigen/Eigen>

namespace lass {

struct ReprojectionErrorWithQuaternions {
    ReprojectionErrorWithQuaternions(Eigen::Vector2d pt, Eigen::Matrix3d K) :
        pt(pt), K(K) {
    }

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(camera, point, p);

        p[0] += camera[4];
        p[1] += camera[5];
        p[2] += camera[6];
        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];
        // Compute final projected point position.
        const T predicted_x = T(K(0, 0)) * xp + T(K(0, 2));
        const T predicted_y = T(K(1, 1)) * yp + T(K(1, 2));

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(pt.x());
        residuals[1] = predicted_y - T(pt.y());

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Vector2d pt,
                                       const Eigen::Matrix3d K) {
        return (new ceres::AutoDiffCostFunction<
                ReprojectionErrorWithQuaternions, 2, 7, 3>(
            new ReprojectionErrorWithQuaternions(pt, K)));
    }

    Eigen::Vector2d pt;
    Eigen::Matrix3d K;
};

} // namespace lass
