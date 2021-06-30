/*
Reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include "libfive/tree/tree.hpp"
#include "libfive/tree/data.hpp"

#include "clause.hpp"
#include "tape.hpp"
#include "gpu_opcode.hpp"

namespace mpr {

Tape::Tape(const libfive::Tree& tree) {
    const auto tree_ = tree.optimized();
    auto ordered = tree_.walk();
    std::vector<const libfive::Tree::Data*> ordered_fast;
    ordered_fast.reserve(ordered.size());

    std::map<libfive::Tree::Id, libfive::Tree::Id> last_used;
    const libfive::Tree::Data* axes_used[3] = {nullptr};
    for (auto& c : ordered) {
        using namespace libfive::Opcode;
        // Very simple tracking of active spans, without clause reordering
        // or any other cleverness.
        switch (c->op()) {
            case CONSTANT: continue;
            case VAR_X: axes_used[0] = c; break;
            case VAR_Y: axes_used[1] = c; break;
            case VAR_Z: axes_used[2] = c; break;

            // Opcodes which take two arguments store their RHS
            case OP_ADD:
            case OP_MUL:
            case OP_MIN:
            case OP_MAX:
            case OP_SUB:
            case OP_DIV:    last_used[c->rhs().id()] = c;
                            // FALLTHROUGH

            // Unary opcodes (and fallthrough) store their LHS)
            case OP_SQUARE:
            case OP_SQRT:
            case OP_NEG:
            case OP_SIN:
            case OP_COS:
            case OP_ASIN:
            case OP_ACOS:
            case OP_ATAN:
            case OP_EXP:
            case OP_ABS:
            case OP_LOG:    last_used[c->lhs().get()] = c;
                            ordered_fast.push_back(c);
                            break;
            default:    break;
        }
    }

    std::vector<uint8_t> free_slots;
    std::map<const libfive::Tree::Data*, uint8_t> bound_slots;
    uint8_t num_slots = 1;

    auto getSlot = [&](const libfive::Tree::Data* id) {
        // Pick a slot for the output of this opcode
        uint8_t out = 0;
        if (free_slots.size()) {
            out = free_slots.back();
            free_slots.pop_back();
        } else {
            if (num_slots == UINT8_MAX) {
                fprintf(stderr, "Ran out of slots!\n");
            } else {
                out = num_slots++;
            }
        }
        bound_slots[id] = out;
        return out;
    };

    // Bind the axes to known slots, so that we can store their values
    // before beginning an evaluation.
    uint64_t start = 0;
    for (unsigned i=0; i < 3; ++i) {
        if (axes_used[i] != nullptr) {
            ((uint8_t*)&start)[i + 1] = getSlot(axes_used[i]);
        }
    }
    std::vector<uint64_t> flat;
    flat.reserve(ordered.size());
    flat.push_back(start);

    auto get_reg = [&](const libfive::TreeData* tree) {
        auto itr = bound_slots.find(tree);
        if (itr != bound_slots.end()) {
            return itr->second;
        } else {
            fprintf(stderr, "Could not find bound slots %i\n", tree->op());
            return static_cast<uint8_t>(0);
        }
    };

    for (auto& c : ordered_fast) {
        uint64_t clause = 0;
        switch (c->op()) {
            using namespace libfive::Opcode;

            case CONSTANT:
            case VAR_X:
            case VAR_Y:
            case VAR_Z:
                continue;

#define OP_UNARY(p) \
            case OP_##p: { \
                OP(&clause) = GPU_OP_##p##_LHS;             \
                I_LHS(&clause) = get_reg(c->lhs().get());   \
                break;                                      \
            }
            OP_UNARY(SQUARE)
            OP_UNARY(SQRT);
            OP_UNARY(NEG);
            OP_UNARY(SIN);
            OP_UNARY(COS);
            OP_UNARY(ASIN);
            OP_UNARY(ACOS);
            OP_UNARY(ATAN);
            OP_UNARY(EXP);
            OP_UNARY(ABS);
            OP_UNARY(LOG);

#define OP_COMMUTATIVE(p) \
            case OP_##p: { \
                if (c->lhs()->op() == CONSTANT) {               \
                    OP(&clause) = GPU_OP_##p##_LHS_IMM;         \
                    I_LHS(&clause) = get_reg(c->rhs().get());   \
                    IMM(&clause) = c->lhs()->value();           \
                } else if (c->rhs()->op() == CONSTANT) {        \
                    OP(&clause) = GPU_OP_##p##_LHS_IMM;         \
                    I_LHS(&clause) = get_reg(c->lhs().get());   \
                    IMM(&clause) = c->rhs()->value();           \
                } else {                                        \
                    OP(&clause) = GPU_OP_##p##_LHS_RHS;         \
                    I_LHS(&clause) = get_reg(c->lhs().get());   \
                    I_RHS(&clause) = get_reg(c->rhs().get());   \
                }                                               \
                break;                                          \
            }
            OP_COMMUTATIVE(ADD)
            OP_COMMUTATIVE(MUL)
            OP_COMMUTATIVE(MIN)
            OP_COMMUTATIVE(MAX)

#define OP_NONCOMMUTATIVE(p) \
            case OP_##p: { \
                if (c->lhs()->op() == CONSTANT) {               \
                    OP(&clause) = GPU_OP_##p##_IMM_RHS;         \
                    I_RHS(&clause) = get_reg(c->rhs().get());   \
                    IMM(&clause) = c->lhs()->value();           \
                } else if (c->rhs()->op() == CONSTANT) {        \
                    OP(&clause) = GPU_OP_##p##_LHS_IMM;         \
                    I_LHS(&clause) = get_reg(c->lhs().get());   \
                    IMM(&clause) = c->rhs()->value();           \
                } else {                                        \
                    OP(&clause) = GPU_OP_##p##_LHS_RHS;         \
                    I_LHS(&clause) = get_reg(c->lhs().get());   \
                    I_RHS(&clause) = get_reg(c->rhs().get());   \
                }                                               \
                break;                                          \
            }
            OP_NONCOMMUTATIVE(SUB)
            OP_NONCOMMUTATIVE(DIV)

            case INVALID:
            case OP_TAN:
            case OP_RECIP:
            case OP_ATAN2:
            case OP_POW:
            case OP_NTH_ROOT:
            case OP_MOD:
            case OP_NANFILL:
            case OP_COMPARE:
            case VAR_FREE:
            case CONST_VAR:
            case ORACLE:
            case LAST_OP:
                fprintf(stderr, "Unimplemented opcode");
                break;
        }

        // Release slots if this was their last use.  We do this now so
        // that one of them can be reused for the output slots below.
        if (libfive::Opcode::args(c->op()) == 2) {
            for (auto& h : {c->lhs().get(), c->rhs().get()}) {
                if (h->op() != libfive::Opcode::CONSTANT &&
                    last_used[h] == c)
                {
                    auto itr = bound_slots.find(h);
                    free_slots.push_back(itr->second);
                    bound_slots.erase(itr);
                }
            }
        } else if (libfive::Opcode::args(c->op()) == 1) {
            auto h = c->lhs().get();
            if (h->op() != libfive::Opcode::CONSTANT &&
                last_used[h] == c)
            {
                auto itr = bound_slots.find(h);
                free_slots.push_back(itr->second);
                bound_slots.erase(itr);
            }
        }

        I_OUT(&clause) = getSlot(c);
        flat.push_back(clause);
    }

    {   // Push the end of the tape, which points to the final clauses's
        // output slot so that we know where to read the result.
        uint64_t end = 0;
        I_OUT(&end) = get_reg(ordered.back());
        flat.push_back(end);
    }

    data.reset(CUDA_MALLOC(uint64_t, flat.size()));
    CUDA_CHECK(cudaMemcpy(data.get(), flat.data(),
                          sizeof(uint64_t) * flat.size(),
                          cudaMemcpyHostToDevice));
    length = flat.size();
}

} // namespace mpr

