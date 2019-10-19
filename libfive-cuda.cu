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
    auto r = Renderable::build(circle, 256, 16);
    r->run();

    //auto d_out = callProcessTiles(tape);
    cudaDeviceSynchronize();
    //printf("%u %u\n", d_out->num_active, d_out->num_filled);

    return 0;
}
