#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "renderable.hpp"

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
    auto r = Renderable::build(t, 128, 2);

    for (unsigned i=0; i < r->tape.num_clauses; ++i) {
        const auto c = r->tape[i];
        std::cout << libfive::Opcode::toString((libfive::Opcode::Opcode)c.opcode) << " & "
                  << ((c.banks & 1) ? "constant " : "slot ") << c.lhs << " & "
                  << ((c.banks & 2) ? "constant " : "slot ") << c.rhs << " & "
                  << c.out << "\\\\\n";
    }
    std::cout << "\n\n";
    for (unsigned i=0; i < r->tape.num_constants; ++i) {
        const float f = r->tape.constant(i);
        std::cout << i << " & " << f << "\\\\\n";
    }
}
