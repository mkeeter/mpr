#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "renderable.hpp"
#include "v3.hpp"

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
    } else {
        auto X = libfive::Tree::X();
        auto Y = libfive::Tree::Y();
        auto Z = libfive::Tree::Z();
        t = min(sqrt((X + 0.5)*(X + 0.5) + Y*Y + Z*Z) - 0.25,
                sqrt((X - 0.5)*(X - 0.5) + Y*Y + Z*Z) - 0.25);
    }
    const std::vector<int> sizes = {256, 512, 1024, 1536, 2048};
    std::cout << "Rendering with v3 architecture:" << std::endl;
    for (auto size: sizes) {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(3,2) = 0.3f;
        auto r = build_v3_blob(t, size);
        // Warm-up
        for (unsigned i=0; i < 20; ++i) {
            render_v3_blob(r, Eigen::Matrix4f::Identity());
        }
        auto start_gpu = std::chrono::steady_clock::now();
        const auto count = 100;
        for (unsigned i=0; i < count; ++i) {
            render_v3_blob(r, Eigen::Matrix4f::Identity());
        }
        auto end_gpu = std::chrono::steady_clock::now();
        std::cout << size << " " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() / (float)count <<
            " ms\n";

        libfive::Heightmap out(size, size);
        uint32_t i = 0;
        for (int x=0; x < size; ++x) {
            for (int y=0; y < size; ++y) {
                out.depth(y, x) = r.image[i++];
            }
        }
        out.savePNG("out_gpu_depth_v3_" + std::to_string(size) + ".png");
        free_v3_blob(r);
    }
    std::cout << "Rendering with original architecture:" << std::endl;
    for (auto size: sizes) {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(3,2) = 0.3f;

        auto r = Renderable::build(t, size, 3);
        // Warm-up
        for (unsigned i=0; i < 20; ++i) {
            r->run({Eigen::Matrix4f::Identity()}, Renderable::MODE_HEIGHTMAP);
        }
        auto start_gpu = std::chrono::steady_clock::now();
        const auto count = 100;
        for (unsigned i=0; i < count; ++i) {
            r->run({Eigen::Matrix4f::Identity()}, Renderable::MODE_HEIGHTMAP);
        }
        auto end_gpu = std::chrono::steady_clock::now();
        std::cout << size << " " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() / (float)count <<
            " ms\n";

        libfive::Heightmap out(size, size);
        for (int x=0; x < size; ++x) {
            for (int y=0; y < size; ++y) {
                out.depth(y, x) = r->heightAt(y, x);
            }
        }
        out.savePNG("out_gpu_depth_orig_" + std::to_string(size) + ".png");
    }
    return 0;
}
