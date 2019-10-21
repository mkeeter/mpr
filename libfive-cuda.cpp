#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>

// CUDA runtime
#include <cuda_runtime.h>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>

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
        t= min(sqrt((X + 1.5)*(X + 1.5)+ Y*Y) - 1.0,
               sqrt((X - 1.5)*(X - 1.5) + Y*Y) - 1.0);
    }
    auto r = Renderable::build(t, 256);
    for (unsigned i=0; i < 10; ++i) {
        r->run({{0, 0}, 2});
        cudaDeviceSynchronize();
    }

    if (r->IMAGE_SIZE_PX == 256) {
        for (unsigned i=0; i < r->IMAGE_SIZE_PX; ++i) {
            for (unsigned j=0; j < r->IMAGE_SIZE_PX; ++j) {
                switch (r->image[i * r->IMAGE_SIZE_PX + j]) {
                    case 0:     printf(" "); break;
                    case 1:     printf("."); break;
                    default:    printf("X"); break;
                }
            }
            printf("\n");
        }
    }

    return 0;
}
