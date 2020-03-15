#include <algorithm>

#include "libfive/tree/cache.hpp"

#include "renderable_tape.hpp"
#include "util.hpp"
#include "renderable_parameters.hpp"

Tape Tape::build(libfive::Tree tree) {
    // Hold a single cache lock to avoid needing mutex locks everywhere
    auto lock = libfive::Cache::instance();

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

    std::vector<uint16_t> free_registers;
    std::map<libfive::Tree::Id, uint16_t> bound_registers;
    uint16_t num_registers = 0;

    auto getRegister = [&](libfive::Tree::Id id) {
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
        bound_registers[id] = out;
        return out;
    };

    // Bind the axes to known registers, so that we can store their values
    // before beginning an evaluation.
    const libfive::Tree axis_trees[3] = {
        libfive::Tree::X(), libfive::Tree::Y(), libfive::Tree::Z()};
    Axes axes;
    for (unsigned i=0; i < 3; ++i) {
        axes.reg[i] = has_axis[i] ? getRegister(axis_trees[i].id())
                                  : UINT16_MAX;
    }

    std::vector<Clause> flat;
    flat.reserve(ordered.size());

    uint8_t banks = 0;
    auto f = [&](const std::shared_ptr<libfive::Tree::Tree_>& tree,
                 uint8_t mask)
    {
        if (tree == nullptr) {
            return static_cast<uint16_t>(0);
        }
        if (tree->op == libfive::Opcode::CONSTANT) {   // Check whether this is a constant
            auto itr = constants.find(tree.get());
            if (itr != constants.end()) {
                banks |= mask;
                return itr->second;
            } else {
                fprintf(stderr, "Could not find constant");
                return static_cast<uint16_t>(0);
            }
        }
        {   // Otherwise, it must be a bound register
            auto itr = bound_registers.find(tree.get());
            if (itr != bound_registers.end()) {
                return itr->second;
            } else {
                fprintf(stderr, "Could not find bound register");
                return static_cast<uint16_t>(0);
            }
        }
    };
    for (auto& c : ordered) {
        // Constants are not inserted into the tape, because they
        // live in a separate data array addressed with flags in
        // the 'banks' argument of a Clause.
        if (c->op == libfive::Opcode::CONSTANT ||
            c->op == libfive::Opcode::VAR_X ||
            c->op == libfive::Opcode::VAR_Y ||
            c->op == libfive::Opcode::VAR_Z)
        {
            continue;
        }

        banks = 0;
        const uint16_t lhs = f(c->lhs, 1);
        const uint16_t rhs = f(c->rhs, 2);

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

        const uint16_t out = getRegister(c.id());
        flat.push_back({static_cast<uint8_t>(c->op), banks, out, lhs, rhs});
    }

    // Count up how many times each slot is used
    std::vector<uint32_t> reg_use_count;
    reg_use_count.resize(num_registers, 0);
    for (auto& c : flat) {
        if (!(c.banks & 1)) {
            reg_use_count[c.lhs]++;
        }
        if (c.opcode >= libfive::Opcode::OP_ADD && !(c.banks & 2)) {
            reg_use_count[c.rhs]++;
        }
        reg_use_count[c.out]++;
    }

    // Prepare to sort by register use
    std::vector<uint16_t> sorted_regs;
    sorted_regs.reserve(num_registers);
    for (unsigned i=0; i < reg_use_count.size(); ++i) {
        sorted_regs.push_back(i);
    }
    std::sort(sorted_regs.begin(), sorted_regs.end(),
            [&](const uint16_t& a, const uint16_t& b){
                return reg_use_count[a] > reg_use_count[b];
            });

    // Plan to remap registers based on frequency
    std::vector<uint16_t> remapped_regs;
    remapped_regs.resize(sorted_regs.size());
    for (uint16_t i=0; i < sorted_regs.size(); ++i) {
        remapped_regs[sorted_regs[i]] = i;
    }

    // Apply remap to clauses in the tape
    for (auto& c : flat) {
        if (!(c.banks & 1)) {
            c.lhs = remapped_regs[c.lhs];
        }
        if (c.opcode >= libfive::Opcode::OP_ADD && !(c.banks & 2)) {
            c.rhs = remapped_regs[c.rhs];
        }
        c.out = remapped_regs[c.out];
    }

    // Also apply remap to the original axes
    for (unsigned i=0; i < 3; ++i) {
        axes.reg[i] = has_axis[i] ? remapped_regs[axes.reg[i]]
                                  : UINT16_MAX;
    }

    auto data = CUDA_MALLOC(char, sizeof(Clause) * flat.size() +
                                  sizeof(float) * std::max(1UL, constant_data.size()));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy everything onto the GPU
    CUDA_CHECK(cudaMemcpy(data, flat.data(), sizeof(Clause) * flat.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data + sizeof(Clause) * flat.size(),
                          constant_data.data(),
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
      axes(axes), data(data),
      tape(reinterpret_cast<const Clause*>(data)),
      constants(reinterpret_cast<const float*>(tape + num_clauses))
{
    // Nothing to do here
}

Tape::Tape(Tape&& other)
    : num_clauses(other.num_clauses), num_constants(other.num_constants),
      num_regs(other.num_regs), num_csg_choices(other.num_csg_choices),
      axes(other.axes), data(other.data), tape(other.tape),
      constants(other.constants)
{
    other.data = nullptr;
    other.tape = nullptr;
    other.constants = nullptr;
}

Tape::~Tape()
{
    CUDA_FREE((void*)data);
}

void Tape::print(std::ostream& o)
{
    for (unsigned i=0; i < num_clauses; ++i) {
        const Clause c = tape[i];
        const auto op = static_cast<libfive::Opcode::Opcode>(c.opcode);
        o << libfive::Opcode::toString(op) << " ";
        if (c.banks & 1) {
            o << constants[c.lhs] << "f ";
        } else {
            o << c.lhs << " ";
        }
        if (c.banks & 2) {
            o << constants[c.rhs] << "f ";
        } else {
            o << c.rhs << " ";
        }
        o << " -> " << c.out << "\n";
    }
}
