#include "tape.hpp"
#include "check.hpp"
#include "parameters.hpp"

Tape Tape::build(libfive::Tree tree) {
    auto ordered = tree.orderedDfs();

    std::map<libfive::Tree::Id, libfive::Tree::Id> last_used;
    std::vector<float> constant_data;
    std::map<libfive::Tree::Id, uint16_t> constants;
    uint16_t num_csg_choices = 0;
    bool has_axis[3] = {false, false, false};
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

        has_axis[0] |= (c->op == libfive::Opcode::VAR_X);
        has_axis[1] |= (c->op == libfive::Opcode::VAR_Y);
        has_axis[2] |= (c->op == libfive::Opcode::VAR_Z);
    }

    std::list<uint16_t> free_registers;
    std::list<uint16_t> free_fast_registers;
    std::map<libfive::Tree::Id, uint16_t> bound_registers;

    uint16_t num_registers = LIBFIVE_CUDA_FAST_REG_COUNT;
    for (unsigned i=0; i < LIBFIVE_CUDA_FAST_REG_COUNT; ++i) {
        free_fast_registers.push_back(i);
    }
    // Bind the axes to known registers, so that we can store their values
    // before beginning an evaluation.
    const libfive::Tree axis_trees[3] = {
        libfive::Tree::X(), libfive::Tree::Y(), libfive::Tree::Z()};
    Axes axes;
    for (unsigned i=0; i < 3; ++i) {
        if (has_axis[i]) {
            axes.reg[i] = free_fast_registers.back();
            free_fast_registers.pop_back();
            bound_registers[axis_trees[i].id()] = axes.reg[i];
        } else {
            axes.reg[i] = UINT16_MAX;
        }
    }

    std::vector<Clause> flat;
    for (auto& c : ordered) {
        // Constants are not inserted into the tape, because they
        // live in a separate data array addressed with flags in
        // the 'banks' argument of a Clause.
        if (constants.find(c.id()) != constants.end()) {
            continue;
        } else if (c->op == libfive::Opcode::VAR_X ||
                   c->op == libfive::Opcode::VAR_Y ||
                   c->op == libfive::Opcode::VAR_Z)
        {
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
                if (itr->second < LIBFIVE_CUDA_FAST_REG_COUNT) {
                    free_fast_registers.push_back(itr->second);
                } else {
                    free_registers.push_back(itr->second);
                }
                bound_registers.erase(itr);
            }
        }

        // Pick a registers for the output of this opcode
        uint16_t out;
        if (free_fast_registers.size()) {
            out = free_fast_registers.back();
            free_fast_registers.pop_back();
        } else if (free_registers.size()) {
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

    auto data = CUDA_MALLOC(char, sizeof(Clause) * flat.size() +
                                  sizeof(float) * constant_data.size());
    CHECK(cudaDeviceSynchronize());

    // Copy everything onto the GPU
    CHECK(cudaMemcpy(data, flat.data(), sizeof(Clause) * flat.size(),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(data + sizeof(Clause) * flat.size(), constant_data.data(),
                     sizeof(float) * constant_data.size(),
                     cudaMemcpyHostToDevice));

    return Tape(data,
                static_cast<uint16_t>(flat.size()),
                static_cast<uint16_t>(constant_data.size()),
                num_registers, num_csg_choices,
                axes);
}

Tape::Tape(const char* data,
           uint16_t num_clauses, uint16_t num_constants,
           uint16_t num_regs, uint16_t num_csg_choices,
           Axes axes)
    : num_clauses(num_clauses), num_constants(num_constants),
      num_regs(num_regs), num_csg_choices(num_csg_choices),
      axes(axes), data(data)
{
    // Nothing to do here
}

Tape::Tape(Tape&& other)
    : num_clauses(other.num_clauses), num_constants(other.num_constants),
      num_regs(other.num_regs), num_csg_choices(other.num_csg_choices),
      axes(other.axes), data(other.data)
{
    other.data = nullptr;
}

Tape::~Tape()
{
    CHECK(cudaFree((void*)data));
}

void Tape::sendToConstantMemory(const char* ptr)
{
    const char* dev_ptr;
    CHECK(cudaGetSymbolAddress((void**)&dev_ptr, ptr));

    tape = reinterpret_cast<const Clause*>(dev_ptr);
    constants = reinterpret_cast<const float*>(
            dev_ptr + sizeof(Clause) * num_clauses);

    CHECK(cudaMemcpyToSymbol(ptr, data,
            sizeof(Clause) * num_clauses +
            sizeof(float)  * num_constants,
            0, cudaMemcpyDeviceToDevice));
}
