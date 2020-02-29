#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "v2.hpp"

int main(int argc, char** argv)
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
        auto a = sqrt((X - 0.5)*(X - 0.5) + Y*Y + Z*Z) - 0.2;
        auto b = sqrt((X + 0.5)*(X + 0.5) + Y*Y + Z*Z) - 0.2;
        t = min(a, b);
    }

    auto blob = build_v2_blob(t, 64);
    render_v2_blob(blob, Eigen::Matrix4f::Identity());

    // Save the image using libfive::Heightmap
    libfive::Heightmap out(blob.image_size_px, blob.image_size_px);
    uint32_t i = 0;
    for (unsigned x=0; x < blob.image_size_px; ++x) {
        for (unsigned y=0; y < blob.image_size_px; ++y) {
            out.depth(y, x) = blob.image[i++];
        }
        printf("\n");
    }
    out.savePNG("v2.png");

    free_v2_blob(blob);
}
