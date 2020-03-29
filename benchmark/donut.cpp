#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "clause.hpp"
#include "tape.hpp"
#include "gpu_opcode.hpp"

// Not actually a benchmark, used to generate data for the paper
int main(int argc, char** argv)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto t = max(sqrt(X*X + Y*Y) - 1, 0.5 - sqrt(X*X + Y*Y));

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
    }
    auto r = libfive::cuda::Tape(t);

    for (int i=1; i < r.length - 1; ++i) {
        const auto c = r.data[i];
        std::cout << gpu_op_str(OP(&c)) << " & "
                  << (int)I_LHS(&c) << " & "
                  << (int)I_RHS(&c) << " & "
                  << IMM(&c) << " & "
                  << (int)I_OUT(&c) << "\\\\\n";
    }
}
