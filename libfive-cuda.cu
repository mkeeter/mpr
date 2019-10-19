// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <math_constants.h>

// libfive
#include <libfive/tree/opcode.hpp>
#include <libfive/tree/tree.hpp>

#include "renderable.hpp"

int main(int argc, char **argv)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto circle = sqrt(X*X + Y*Y) - 1.0;
    auto r = Renderable::build(circle, 4096, 16);
    r->run();
    cudaDeviceSynchronize();

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
