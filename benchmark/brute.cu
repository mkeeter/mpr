/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "context.hpp"
#include "tape.hpp"

#include "stats.hpp"

////////////////////////////////////////////////////////////////////////////////

__global__ void evalRawTape(int image_size_px, int32_t* image)
{
#define NOT_CUSTOM_KERNEL
    uint32_t px = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t py = threadIdx.y + blockIdx.y * blockDim.y;

    if (px >= image_size_px && py >= image_size_px) {
        return;
    }

    const float x = 2.0f * ((px + 0.5f) / image_size_px - 0.5f);
    const float y = 2.0f * ((py + 0.5f) / image_size_px - 0.5f);
    const float z = 0.0f;

    const float v140496806626256 = x + 0.500000f;
    const float v140496806626432 = v140496806626256 * v140496806626256;
    const float v140496806626608 = y * y;
    const float v140496806626992 = z * z;
    const float v140496806627168 = v140496806626608 + v140496806626992;
    const float v140496806627488 = v140496806626432 + v140496806627168;
    const float v140496806627344 = sqrt(v140496806627488);
    const float v140496806627888 = v140496806627344 - 0.250000f;
    const float v140496806628064 = x - 0.500000f;
    const float v140496806628240 = v140496806628064 * v140496806628064;
    const float v140496806628880 = v140496806628240 + v140496806627168;
    const float v140496806628704 = sqrt(v140496806628880);
    const float v140496806628944 = v140496806628704 - 0.250000f;
    const float v140496806629120 = min(v140496806627888, v140496806628944);
    if (v140496806629120 < 0.0f) {
        image[px + py * image_size_px] = 255;
    } else {
        image[px + py * image_size_px] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    libfive::Tree t = libfive::Tree::X();
    if (argc == 2) {
        std::ifstream ifs;
        ifs.open(argv[1]);
        if (ifs.is_open()) {
            auto a = libfive::Archive::deserialize(ifs);
            t = a.shapes.front().tree;
        } else {
            fprintf(stderr, "Could not open file %s\n", argv[1]);
            exit(1);
        }

#ifdef NOT_CUSTOM_KERNEL
        fprintf(stderr, "brute.cu is not compiled with a kernel for this custom shape.\n"
                "Please run dump_tape and paste the results into evalRawTape in brute.cu;\n"
                "otherwise, results will not be meaningful.\n");
        exit(1);
#endif
    } else {
        auto X = libfive::Tree::X();
        auto Y = libfive::Tree::Y();
        auto Z = libfive::Tree::Z();
        t = min(sqrt((X + 0.5)*(X + 0.5) + Y*Y + Z*Z) - 0.25,
                sqrt((X - 0.5)*(X - 0.5) + Y*Y + Z*Z) - 0.25);
    }
    auto tape = libfive::cuda::Tape(t);

    // Note: we deliberately leak the heightmaps!
    //
    // libfive is compiled with -march=native, which means the heightmap images
    // are aligned for SSE; however, I can't convince CUDA to pass this flag
    // through, so the destructor will crash.  Leaking it "fixes" the issue,
    // which is fine, because this isn't user-facing code.
    std::cout << "Rendering brute-force with compiled kernel\n";
    for (int size=256; size <= 2048; size += 64) {
        auto image = CUDA_MALLOC(int32_t, size*size);

        std::cout << size << " ";
        get_stats([&]() {
            const unsigned bs = (size + 15) / 16;
            evalRawTape<<<dim3(bs, bs), dim3(16, 16)>>>(size, image);
            CUDA_CHECK(cudaDeviceSynchronize());
        });

        // Save the image using libfive::Heightmap
        libfive::Heightmap* out = new libfive::Heightmap(size, size);
        uint32_t i = 0;
        for (int x=0; x < size; ++x) {
            for (int y=0; y < size; ++y) {
                out->depth(x, y) = image[i++];
            }
        }
        CUDA_FREE(image);
        out->savePNG("out_kernel_" + std::to_string(size) + ".png");
    }
    std::cout << "Rendering brute-force with interpreter\n";
    for (int size=256; size <= 2048; size += 64) {
        auto ctx = libfive::cuda::Context(size);
        std::cout << size << " ";
        get_stats([&](){ ctx.render2D_brute(tape, Eigen::Matrix3f::Identity()); });

        // Save the image using libfive::Heightmap
        libfive::Heightmap* out = new libfive::Heightmap(size, size);
        uint32_t i = 0;
        for (int x=0; x < size; ++x) {
            for (int y=0; y < size; ++y) {
                out->depth(x, y) = ctx.stages[3].filled[i++];
            }
        }
        out->savePNG("out_brute_" + std::to_string(size) + ".png");
    }

    std::cout << "Rendering fancy algorithm with interpreter\n";
    for (int size=256; size <= 4096; size += 64) {
        auto ctx = libfive::cuda::Context(size);
        std::cout << size << " ";
        get_stats([&](){ ctx.render2D(tape, Eigen::Matrix3f::Identity()); });

        // Save the image using libfive::Heightmap
        libfive::Heightmap* out = new libfive::Heightmap(size, size);
        uint32_t i = 0;
        for (int x=0; x < size; ++x) {
            for (int y=0; y < size; ++y) {
                out->depth(x, y) = ctx.stages[3].filled[i++];
            }
        }
        out->savePNG("out_alg_" + std::to_string(size) + ".png");
    }
}
