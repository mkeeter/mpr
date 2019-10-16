// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <math_constants.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// libfive
#include <libfive/tree/opcode.hpp>
#include <libfive/tree/tree.hpp>

// Our Interval arithmetic class
#include "gpu_interval.hpp"

struct Clause {
    const uint8_t opcode;
    const uint8_t banks;
    const uint16_t out;
    const uint16_t lhs;
    const uint16_t rhs;
};

// The Tape is an on-device representation, so the pointers
// are returned from cudaMalloc.
struct Tape {
    const Clause* const __restrict__ tape;
    const uint32_t tape_length;

    const uint16_t num_regs;
    const uint16_t num_csg_choices;

    const float* const __restrict__ constants;
};

__device__ void walk(const Tape tape,
                     const Interval X, const Interval Y,
                     Interval* const __restrict__ regs,
                     uint8_t* const __restrict__ choices)
{
    uint32_t choice_index = 0;
    for (uint32_t i=0; i < tape.tape_length; ++i) {
        const Clause c = tape.tape[i];
#define LHS ((!(c.banks & 1) ? regs[c.lhs] : Interval{tape.constants[c.lhs], \
                                                     tape.constants[c.lhs]}))
#define RHS ((!(c.banks & 2) ? regs[c.rhs] : Interval{tape.constants[c.rhs], \
                                                     tape.constants[c.rhs]}))
        using namespace libfive::Opcode;
        switch (c.opcode) {
            case VAR_X: regs[c.out] = X; break;
            case VAR_Y: regs[c.out] = Y; break;

            case OP_SQUARE: regs[c.out] = LHS.square(); break;
            case OP_SQRT: regs[c.out] = LHS.sqrt(); break;
            case OP_NEG: regs[c.out] = -LHS; break;
            // Skipping transcendental functions for now

            case OP_ADD: regs[c.out] = LHS + RHS; break;
            case OP_MUL: regs[c.out] = LHS * RHS; break;
            case OP_MIN: if (LHS.upper < RHS.lower) {
                             choices[choice_index] = 1;
                             regs[c.out] = LHS;
                         } else if (RHS.upper < LHS.lower) {
                             choices[choice_index] = 2;
                             regs[c.out] = RHS;
                         } else {
                             choices[choice_index] = 0;
                             regs[c.out] = LHS.min(RHS);
                         }
                         choice_index++;
                         break;
            case OP_MAX: if (LHS.lower > RHS.upper) {
                             choices[choice_index] = 1;
                             regs[c.out] = LHS;
                         } else if (RHS.lower > LHS.upper) {
                             choices[choice_index] = 2;
                             regs[c.out] = RHS;
                         } else {
                             choices[choice_index] = 0;
                             regs[c.out] = LHS.max(RHS);
                         }
                         choice_index++;
                         break;
            case OP_SUB: regs[c.out] = LHS - RHS; break;

            // Skipping various hard functions here
            default: break;
        }
    }
#undef LHS
#undef RHS
}

struct Output {
    uint32_t* const __restrict__ tiles;
    const uint32_t tiles_length;

    uint32_t num_active;
    uint32_t num_filled;
};

const static uint32_t IMAGE_SIZE_PX = 65535;
const static uint32_t TILE_SIZE_PX = 16;
const static uint32_t TILE_COUNT = IMAGE_SIZE_PX / TILE_SIZE_PX;
const static uint32_t TOTAL_TILES = TILE_COUNT * TILE_COUNT;

const static uint32_t NUM_BLOCKS = 128;
const static uint32_t THREADS_PER_BLOCK = TILE_COUNT / NUM_BLOCKS;

__global__ void processTiles(const Tape tape,
        // Flat array for all pseudoregisters
        Interval* const __restrict__ regs_,

        // Flat array for all CSG choices
        uint8_t* const __restrict__ csg_choices_,

        // Output data
        Output* const __restrict__ out)
{
    const float x = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    const float y = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;

    const Interval X = {x / TILE_COUNT, (x + 1) / TILE_COUNT};
    const Interval Y = {y / TILE_COUNT, (y + 1) / TILE_COUNT};

    // Unpack a 1D offset into the data arrays
    const uint32_t index = x * TILE_COUNT + y;
    auto regs = regs_ + index * tape.num_regs;
    auto csg_choices = csg_choices_ + index * tape.num_csg_choices;
    walk(tape, X, Y, regs, csg_choices);

    const Interval result = regs[tape.tape[tape.tape_length - 1].out];
    if (result.lower <= 0.0f && result.upper >= 0.0f) {
        printf("[%f %f][%f %f]: [%f %f]\n",
                X.lower, X.upper,
                Y.lower, Y.upper,
                result.lower, result.upper);
    }
}

Tape prepareTape(libfive::Tree tree) {
    auto ordered = tree.ordered();

    std::map<libfive::Tree::Id, libfive::Tree::Id> last_used;
    std::vector<float> constant_data;
    std::map<libfive::Tree::Id, uint16_t> constants;
    uint16_t num_csg_choices = 0;
    for (auto& c : ordered) {
        if (c->op == libfive::Opcode::CONSTANT) {
            // Store constants in a separate list
            constants.insert({c.id(), constant_data.size()});
            constant_data.push_back(c->value);
        } else {
            // Very simple tracking of active spans, without clause reordering
            // or any other cleverness.
            last_used.insert({c.lhs().id(), c.id()});
            last_used.insert({c.rhs().id(), c.id()});

            num_csg_choices += (c->op == libfive::Opcode::OP_MIN ||
                                c->op == libfive::Opcode::OP_MAX);
        }
    }

    std::list<uint16_t> free_registers;
    std::map<libfive::Tree::Id, uint16_t> bound_registers;
    uint16_t num_registers = 0;
    std::vector<Clause> flat;
    for (auto& c : ordered) {
        // Constants are not inserted into the tape, because they
        // live in a separate data array addressed with flags in
        // the 'banks' argument of a Clause.
        if (constants.find(c.id()) != constants.end()) {
            continue;
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
        bound_registers.insert({c.id(), out});

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
                    fprintf(stderr, "Could not LHS?");
                    return static_cast<uint16_t>(0);
                }
            }
        };

        const uint16_t lhs = f(c.lhs().id(), 1);
        const uint16_t rhs = f(c.rhs().id(), 2);

        flat.push_back({static_cast<uint8_t>(c->op), banks, out, lhs, rhs});

        std::cout << libfive::Opcode::toString(c->op) << " "
                  << ((banks & 1) ? constant_data[lhs] : lhs) << " "
                  << ((banks & 2) ? constant_data[rhs] : rhs) << " -> " << out << "\n";

        // Release registers if this was their last use
        for (auto& h : {c.lhs().id(), c.rhs().id()}) {
            if (h != nullptr && h->op != libfive::Opcode::CONSTANT &&
                last_used[h] == c.id())
            {
                auto itr = bound_registers.find(h);
                free_registers.push_back(itr->second);
                bound_registers.erase(itr);
            }
        }
    }

    Clause* d_flat_tape;
    checkCudaErrors(cudaMalloc(
                reinterpret_cast<void **>(&d_flat_tape),
                sizeof(Clause) * flat.size()));
    checkCudaErrors(cudaMemcpy(d_flat_tape, flat.data(),
                sizeof(Clause) * flat.size(),
                cudaMemcpyHostToDevice));

    float* d_flat_constants;
    checkCudaErrors(cudaMalloc(
                reinterpret_cast<void **>(&d_flat_constants),
                sizeof(float) * constant_data.size()));
    checkCudaErrors(cudaMemcpy(
                d_flat_constants, constant_data.data(),
                sizeof(float) * constant_data.size(),
                cudaMemcpyHostToDevice));

    return Tape {
        d_flat_tape,
        static_cast<uint32_t>(flat.size()),
        num_registers,
        num_csg_choices,
        d_flat_constants
    };
}

void callProcessTiles(Tape tape) {
    Interval* d_regs;
    checkCudaErrors(cudaMalloc(
                reinterpret_cast<void **>(&d_regs),
                sizeof(Interval) * tape.num_regs * TOTAL_TILES));

    uint8_t* d_csg_choices;
    checkCudaErrors(cudaMalloc(
                reinterpret_cast<void **>(&d_csg_choices),
                sizeof(uint8_t) * tape.num_csg_choices * TOTAL_TILES));

    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid(NUM_BLOCKS, NUM_BLOCKS);
    processTiles <<< grid, threads >>>(tape,
        d_regs, d_csg_choices,
        nullptr  /* out */);
    const auto code = cudaGetLastError();
    if (code != cudaSuccess) {
        fprintf(stderr, "Failed to launch: %s\n",
                cudaGetErrorString(code));
    }
    cudaDeviceSynchronize();
}

/**
 * Program main
 */
int main(int argc, char **argv)
{
    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    {   // CUDA, help me pick magic numbers:
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
            processTiles);
        printf("Min grid size: %i\tBlock size: %i\n", min_grid_size, block_size);
    }

    {
        auto X = libfive::Tree::X();
        auto Y = libfive::Tree::Y();
        auto circle = sqrt(X*X + Y*Y) - 1.0;
        auto tape = prepareTape(circle);
        callProcessTiles(tape);
    }
    return 0;
}
