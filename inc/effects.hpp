/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
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
