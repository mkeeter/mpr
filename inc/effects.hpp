#pragma once
#include <Eigen/Eigen>

#include "util.hpp"

namespace libfive {
namespace cuda {

struct Effects {
    Effects(int32_t image_size_px);

    Ptr<int32_t[]> tmp;
    Ptr<int32_t[]> image;

    void drawSSAO(const int32_t* depth, const uint32_t* norm);
    void drawShaded(const int32_t* depth, const uint32_t* norm);

protected:
    int32_t image_size_px;

    Eigen::Matrix<float, 64, 3> ssao_kernel;
    Eigen::Matrix<float, 16*16, 3> ssao_rvecs;
};

}   // namespace cuda
}   // namespace libfive
