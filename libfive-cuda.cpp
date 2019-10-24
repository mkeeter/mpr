#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>

// CUDA runtime
#include <cuda_runtime.h>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "renderable.hpp"

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
        t= min(sqrt((X + 0.5)*(X + 0.5)+ Y*Y) - 0.25,
               sqrt((X - 0.5)*(X - 0.5) + Y*Y) - 0.25);
    }
    auto r = Renderable::build(t, 2048);
    r->run({{0, 0}, 1});
    cudaDeviceSynchronize();

    // Save the image using libfive::Heightmap
    libfive::Heightmap out(r->IMAGE_SIZE_PX, r->IMAGE_SIZE_PX);
    for (unsigned x=0; x < r->IMAGE_SIZE_PX; ++x) {
        for (unsigned y=0; y < r->IMAGE_SIZE_PX; ++y) {
            out.depth(y, x) = r->image[x + y * r->IMAGE_SIZE_PX] << 16;
        }
    }
    out.savePNG("out_gpu.png");

    if (r->IMAGE_SIZE_PX == 256) {
        for (unsigned i=0; i < r->IMAGE_SIZE_PX; ++i) {
            for (unsigned j=0; j < r->IMAGE_SIZE_PX; ++j) {
                switch (r->image[i * r->IMAGE_SIZE_PX + j]) {
                    case 0:     printf(" "); break;
                    case 0xF0:  printf("."); break;
                    default:    printf("X"); break;
                }
            }
            printf("\n");
        }
    }

    std::atomic_bool abort(false);
    auto h = libfive::Heightmap::render(t,
            libfive::Voxels({-1, -1, 0}, {1, 1, 0}, r->IMAGE_SIZE_PX / 2),
            abort);
    h->savePNG("out_cpu.png");

    return 0;
}
