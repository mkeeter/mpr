#include "tape.hpp"
#include "check.hpp"

Tape Tape::build(libfive::Tree tree) {
    auto ordered = tree.ordered();

    std::map<libfive::Tree::Id, libfive::Tree::Id> last_used;
    std::vector<float> constant_data = {0};
    std::map<libfive::Tree::Id, uint16_t> constants;
    uint16_t num_csg_choices = 0;
    for (auto& c : ordered) {
        if (c->op == libfive::Opcode::CONSTANT) {
            // Store constants in a separate list
            if (constant_data.size() == UINT16_MAX) {
                fprintf(stderr, "Ran out of constants!\n");
            }
            constants.insert({c.id(), constant_data.size()});
            constant_data.push_back(c->value);
        } else {
            // Very simple tracking of active spans, without clause reordering
            // or any other cleverness.
            last_used[c.lhs().id()] = c.id();
            last_used[c.rhs().id()] = c.id();

            num_csg_choices += (c->op == libfive::Opcode::OP_MIN ||
                                c->op == libfive::Opcode::OP_MAX);
        }
    }

    std::list<uint16_t> free_registers;
    std::map<libfive::Tree::Id, uint16_t> bound_registers;
    uint16_t num_registers = 1; // Use register 0 as the empty register
    std::vector<Clause> flat;
    for (auto& c : ordered) {
        // Constants are not inserted into the tape, because they
        // live in a separate data array addressed with flags in
        // the 'banks' argument of a Clause.
        if (constants.find(c.id()) != constants.end()) {
            continue;
        }

        uint8_t banks = 0;
        auto f = [&](libfive::Tree::Id id, uint8_t mask) {
            if (id == nullptr) {
                return static_cast<uint16_t>(0);
            }
            {   // Check whether this is a constant
                auto itr = constants.find(id);
                if (itr != constants.end()) {
                    banks |= mask;
                    return itr->second;
                }
            }
            {   // Otherwise, it must be a bound register
                auto itr = bound_registers.find(id);
                if (itr != bound_registers.end()) {
                    return itr->second;
                } else {
                    fprintf(stderr, "Could not find bound register");
                    return static_cast<uint16_t>(0);
                }
            }
        };

        const uint16_t lhs = f(c.lhs().id(), 1);
        const uint16_t rhs = f(c.rhs().id(), 2);

        // Release registers if this was their last use.  We do this now so
        // that one of them can be reused for the output register below.
        for (auto& h : {c.lhs().id(), c.rhs().id()}) {
            if (h != nullptr && h->op != libfive::Opcode::CONSTANT &&
                last_used[h] == c.id())
            {
                auto itr = bound_registers.find(h);
                free_registers.push_back(itr->second);
                bound_registers.erase(itr);
            }
        }

        // Pick a registers for the output of this opcode
        uint16_t out;
        if (free_registers.size()) {
            out = free_registers.back();
            free_registers.pop_back();
        } else {
            out = num_registers++;
            if (num_registers == UINT16_MAX) {
                fprintf(stderr, "Ran out of registers!\n");
            }
        }
        bound_registers[c.id()] = out;

        flat.push_back({static_cast<uint8_t>(c->op), banks, out, lhs, rhs});

        std::cout << libfive::Opcode::toString(c->op) << " ";
        if (banks & 1) {
            std::cout << constant_data[lhs] << "f ";
        } else {
            std::cout << lhs << " ";
        }
        if (banks & 2) {
            std::cout << constant_data[rhs] << "f ";
        } else {
            std::cout << rhs << " ";
        }
        std::cout << " -> " << out << "\n";
    }

    auto d_tape = CUDA_MALLOC(Clause, flat.size());
    auto d_constants = CUDA_MALLOC(float, constant_data.size());

    CHECK(cudaDeviceSynchronize());
    memcpy(d_tape, flat.data(), sizeof(Clause) * flat.size());
    memcpy(d_constants, constant_data.data(),
           sizeof(float) * constant_data.size());

    return Tape(d_tape, static_cast<uint32_t>(flat.size()),
                num_registers, num_csg_choices,
                d_constants);
}

Tape::Tape(const Clause* data, uint32_t tape_length,
           uint16_t num_regs, uint16_t num_csg_choices,
           const float* constants)
    : tape_length(tape_length),
      num_regs(num_regs), num_csg_choices(num_csg_choices),
      data(data), constants(constants)
{
    // Nothing to do here
}

Tape::Tape(Tape&& other)
    : tape_length(other.tape_length),
      num_regs(other.num_regs), num_csg_choices(other.num_csg_choices),
      data(other.data), constants(other.constants)
{
    other.data = nullptr;
    other.constants = nullptr;
}

Tape::~Tape()
{
    CHECK(cudaFree((void*)data));
    CHECK(cudaFree((void*)constants));
}
