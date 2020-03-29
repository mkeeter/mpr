#pragma once
#include <Eigen/Eigen>

#include "util.hpp"

namespace libfive {
namespace cuda {

struct Context;

struct Effects {
    Effects();

    Ptr<int32_t[]> tmp;
    Ptr<int32_t[]> image;

    void drawSSAO(const Context& ctx);
    void drawShaded(const Context& ctx);

protected:
    void resizeTo(const Context& ctx);

    int32_t image_size_px;

    Eigen::Matrix<float, 64, 3> ssao_kernel;
    Eigen::Matrix<float, 16*16, 3> ssao_rvecs;
};

}   // namespace cuda
}   // namespace libfive
