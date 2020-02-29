#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

#include "libfive/tree/cache.hpp"

#include "v2.hpp"
#include "check.hpp"
#include "gpu_interval.hpp"
#include "gpu_opcode.hpp"
#include "parameters.hpp"

////////////////////////////////////////////////////////////////////////////////

struct in_tile_t {
    uint32_t position;
    uint32_t tape;
    Interval X, Y, Z;
};

struct out_tile_t {
    uint32_t position;
    uint32_t tape;
};

////////////////////////////////////////////////////////////////////////////////

#define OP(d) (((uint8_t*)d)[0])
#define I_OUT(d) (((uint8_t*)d)[1])
#define I_LHS(d) (((uint8_t*)d)[2])
#define I_RHS(d) (((uint8_t*)d)[3])
#define IMM(d) (((float*)d)[1])
#define JUMP_TARGET(d) (((int32_t*)d)[1])

__device__ void copy_imm_i(const uint64_t data,
                           Interval* const __restrict__ slots)
{
    const float lhs = IMM(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = {lhs, lhs};
}

__device__ void copy_imm_f(const uint64_t data,
                           float* const __restrict__ slots)
{
    const float lhs = IMM(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = lhs;
}

__device__ void copy_lhs_i(const uint64_t data,
                           Interval* const __restrict__ slots)
{
    const uint8_t i_lhs = I_LHS(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = slots[i_lhs];
}

__device__ void copy_lhs_f(const uint64_t data,
                           float* const __restrict__ slots)
{
    const uint8_t i_lhs = I_LHS(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = slots[i_lhs];
}

__device__ void copy_rhs_i(const uint64_t data,
                           Interval* const __restrict__ slots)
{
    const uint8_t i_rhs = I_RHS(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = slots[i_rhs];
}

__device__ void copy_rhs_f(const uint64_t data,
                           float* const __restrict__ slots)
{
    const uint8_t i_rhs = I_RHS(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = slots[i_rhs];
}

#define FUNCTION_PREAMBLE_LHS(name, T, suffix)              \
__device__                                                  \
void name##_lhs_##suffix(const uint64_t data,               \
                    T* const __restrict__ slots)            \
{                                                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs];                             \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_IMM_RHS(name, T, suffix)          \
__device__                                                  \
void name##_imm_rhs_##suffix(const uint64_t data,           \
                    T* const __restrict__ slots)            \
{                                                           \
    const float lhs = IMM(&data);                           \
    const uint8_t i_rhs = I_RHS(&data);                     \
    const T rhs = slots[i_rhs];                             \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_LHS_IMM(name, T, suffix)          \
__device__                                                  \
void name##_lhs_imm_##suffix(const uint64_t data,           \
                    T* const __restrict__ slots)            \
{                                                           \
    const float rhs = IMM(&data);                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs];                             \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_LHS_RHS(name, T, suffix)          \
__device__                                                  \
void name##_lhs_rhs_##suffix(const uint64_t data,           \
                    T* const __restrict__ slots)            \
{                                                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs];                             \
    const uint8_t i_rhs = I_RHS(&data);                     \
    const T rhs = slots[i_rhs];                             \
    const uint8_t i_out = I_OUT(&data);                     \

// Special implementations of min and max, which manipulate the choices array
FUNCTION_PREAMBLE_LHS_IMM(min, float, f)
    slots[i_out] = fminf(lhs, rhs);
}
FUNCTION_PREAMBLE_LHS_IMM(min, Interval, i)
    uint8_t choice = 0;
    slots[i_out] = min(lhs, rhs, choice);
    slots[0].v.x = choice;
}
FUNCTION_PREAMBLE_LHS_RHS(min, float, f)
    slots[i_out] = fminf(lhs, rhs);
}
FUNCTION_PREAMBLE_LHS_RHS(min, Interval, i)
    uint8_t choice = 0;
    slots[i_out] = min(lhs, rhs, choice);
    slots[0].v.x = choice;
}

FUNCTION_PREAMBLE_LHS_IMM(max, float, f)
    slots[i_out] = fmaxf(lhs, rhs);
}
FUNCTION_PREAMBLE_LHS_IMM(max, Interval, i)
    uint8_t choice = 0;
    slots[i_out] = max(lhs, rhs, choice);
    slots[0].v.x = choice;
}
FUNCTION_PREAMBLE_LHS_RHS(max, float, f)
    slots[i_out] = fmaxf(lhs, rhs);
}
FUNCTION_PREAMBLE_LHS_RHS(max, Interval, i)
    uint8_t choice = 0;
    slots[i_out] = max(lhs, rhs, choice);
    slots[0].v.x = choice;
}

#define COMMUTATIVE_OP(name, form)                                  \
FUNCTION_PREAMBLE_LHS_IMM(name, Interval, i)                        \
    slots[i_out] = (form);                                          \
}                                                                   \
FUNCTION_PREAMBLE_LHS_RHS(name, Interval, i)                        \
    slots[i_out] = (form);                                          \
}                                                                   \
FUNCTION_PREAMBLE_LHS_IMM(name, float, f)                           \
    slots[i_out] = (form);                                          \
}                                                                   \
FUNCTION_PREAMBLE_LHS_RHS(name, float, f)                           \
    slots[i_out] = (form);                                          \
}

COMMUTATIVE_OP(add, lhs + rhs);
COMMUTATIVE_OP(mul, lhs * rhs);

#define NONCOMMUTATIVE_OP(name, form)                               \
FUNCTION_PREAMBLE_IMM_RHS(name, Interval, i)                        \
    slots[i_out] = (form);                                          \
}                                                                   \
FUNCTION_PREAMBLE_IMM_RHS(name, float, f)                           \
    slots[i_out] = (form);                                          \
}                                                                   \
COMMUTATIVE_OP(name, form)

NONCOMMUTATIVE_OP(sub, lhs - rhs);
NONCOMMUTATIVE_OP(div, lhs / rhs);

#define UNARY_OP(name, form_f, form_i)                              \
FUNCTION_PREAMBLE_LHS(name, Interval, i)                            \
    slots[i_out] = (form_i);                                        \
}                                                                   \
FUNCTION_PREAMBLE_LHS(name, float, f)                               \
    slots[i_out] = (form_f);                                        \
}
#define UNARY_OP_F(func) UNARY_OP(func, func##f(lhs), func(lhs))

// Completely different shapes
UNARY_OP(abs, fabsf(lhs), abs(lhs))
UNARY_OP(square, lhs * lhs, square(lhs))

// Same form for float and interval
UNARY_OP(neg, -lhs, -lhs)

// Standardized names based on function
UNARY_OP_F(sqrt)
UNARY_OP_F(asin)
UNARY_OP_F(acos)
UNARY_OP_F(atan)
UNARY_OP_F(exp)
UNARY_OP_F(sin)
UNARY_OP_F(cos)
UNARY_OP_F(log)

////////////////////////////////////////////////////////////////////////////////

__global__
void v2_load_i(uint32_t tile_offset,
               uint32_t tile_count,
               uint32_t tile_size,
               uint32_t image_size,
               const Eigen::Matrix4f mat,
               in_tile_t* __restrict__ out)
{
    // Load the axis values
    const uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tile = thread_index + tile_offset;
    const uint32_t tiles_per_side = image_size / tile_size;

    if (tile >= tile_count) {
        return;
    }

    const float size_recip = 1.0f / image_size;
    const uint32_t x = tile % tiles_per_side;
    const Interval ix = {(x * tile_size * size_recip - 0.5f) * 2.0f,
                   ((x + 1) * tile_size * size_recip - 0.5f) * 2.0f};
    const uint32_t y = (tile / tiles_per_side) % tiles_per_side;
    const Interval iy = {(y * tile_size * size_recip - 0.5f) * 2.0f,
                   ((y + 1) * tile_size * size_recip - 0.5f) * 2.0f};
    const uint32_t z = (tile / tiles_per_side) / tiles_per_side;
    const Interval iz = {(z * tile_size * size_recip - 0.5f) * 2.0f,
                   ((z + 1) * tile_size * size_recip - 0.5f) * 2.0f};

    Interval ix_, iy_, iz_, iw_;
    ix_ = mat(0, 0) * ix +
          mat(0, 1) * iy +
          mat(0, 2) * iz + mat(0, 3);
    iy_ = mat(1, 0) * ix +
          mat(1, 1) * iy +
          mat(1, 2) * iz + mat(1, 3);
    iz_ = mat(2, 0) * ix +
          mat(2, 1) * iy +
          mat(2, 2) * iz + mat(2, 3);
    iw_ = mat(3, 0) * ix +
          mat(3, 1) * iy +
          mat(3, 2) * iz + mat(3, 3);

    // Projection!
    ix_ = ix_ / iw_;
    iy_ = iy_ / iw_;
    iz_ = iz_ / iw_;

    out[thread_index].X = ix_;
    out[thread_index].Y = iy_;
    out[thread_index].Z = iz_;
    out[thread_index].position = tile;
    out[thread_index].tape = 0;
}

////////////////////////////////////////////////////////////////////////////////

__global__
void v2_load_s(const out_tile_t* __restrict__ in_tiles,
               const uint32_t num_in_tiles,
               const uint32_t in_thread_offset,
               const uint32_t tile_size,
               const uint32_t image_size,
               const Eigen::Matrix4f mat,
               in_tile_t* __restrict__ out_tiles)
{
    const uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tile_index = (thread_index + in_thread_offset) / 64;
    if (tile_index >= num_in_tiles) {
        return;
    }

    const uint32_t tiles_per_side = image_size / tile_size;

    const uint32_t in_parent_tile = in_tiles[tile_index].position;
    const uint32_t tx = (in_parent_tile % tiles_per_side);
    const uint32_t ty = ((in_parent_tile / tiles_per_side) % tiles_per_side);
    const uint32_t tz = ((in_parent_tile / tiles_per_side) / tiles_per_side);

    // We subdivide at a constant rate of 4x
    const uint32_t subtile_size = tile_size / 4;
    const uint32_t subtiles_per_side = tiles_per_side * 4;
    const uint32_t subtile_offset = thread_index % 64;
    const uint32_t sx = tx * 4 + subtile_offset % 4;
    const uint32_t sy = ty * 4 + (subtile_offset / 4) % 4;
    const uint32_t sz = tz * 4 + (subtile_offset / 4) / 4;

    const float size_recip = 1.0f / image_size;
    const Interval ix = {(sx * subtile_size * size_recip - 0.5f) * 2.0f,
                   ((sx + 1) * subtile_size * size_recip - 0.5f) * 2.0f};
    const Interval iy = {(sy * subtile_size * size_recip - 0.5f) * 2.0f,
                   ((sy + 1) * subtile_size * size_recip - 0.5f) * 2.0f};
    const Interval iz = {(sz * subtile_size * size_recip - 0.5f) * 2.0f,
                   ((sz + 1) * subtile_size * size_recip - 0.5f) * 2.0f};

    Interval ix_, iy_, iz_, iw_;
    ix_ = mat(0, 0) * ix +
          mat(0, 1) * iy +
          mat(0, 2) * iz + mat(0, 3);
    iy_ = mat(1, 0) * ix +
          mat(1, 1) * iy +
          mat(1, 2) * iz + mat(1, 3);
    iz_ = mat(2, 0) * ix +
          mat(2, 1) * iy +
          mat(2, 2) * iz + mat(2, 3);
    iw_ = mat(3, 0) * ix +
          mat(3, 1) * iy +
          mat(3, 2) * iz + mat(3, 3);

    // Projection!
    ix_ = ix_ / iw_;
    iy_ = iy_ / iw_;
    iz_ = iz_ / iw_;

    const uint32_t subtile_index = tile_index * 64 + subtile_offset;
    out_tiles[subtile_index].X = ix_;
    out_tiles[subtile_index].Y = iy_;
    out_tiles[subtile_index].Z = iz_;
    out_tiles[subtile_index].position =
        sx +
        sy * subtiles_per_side +
        sz * subtiles_per_side * subtiles_per_side;
    out_tiles[subtile_index].tape = in_tiles[tile_index].tape;
}

__global__
void v2_exec_universal(uint64_t* const __restrict__ tape_data,
                       uint32_t* const __restrict__ tape_index,

                       uint32_t* const __restrict__ image,
                       const uint32_t tiles_per_side,

                       in_tile_t* const __restrict__ in_tiles,
                       const uint32_t in_tile_count,
                       const uint32_t in_thread_offset,

                       out_tile_t* __restrict__ out_tile,
                       uint32_t* const __restrict__ out_tile_index)
{
    const uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tile_index = thread_index + in_thread_offset;
    if (tile_index >= in_tile_count) {
        return;
    }

    Interval slots[128];

    // Pick out the tape based on the pointer stored in the tiles list
    const uint64_t* __restrict__ data = &tape_data[in_tiles[tile_index].tape];

    // Load the axis values (precomputed by v2_load_*)
    // If the axis isn't assigned to a slot, then it writes to slot 0,
    // which is otherwise unused.
    slots[((const uint8_t*)data)[1]] = in_tiles[tile_index].X;
    slots[((const uint8_t*)data)[2]] = in_tiles[tile_index].Y;
    slots[((const uint8_t*)data)[3]] = in_tiles[tile_index].Z;

    uint32_t choices[256] = {0};
    unsigned choice_index = 0;
    bool has_any_choice = false;

    while (OP(++data)) {
        switch (OP(data)) {
            case GPU_OP_JUMP: data += JUMP_TARGET(data); continue;

            case GPU_OP_SQUARE_LHS: square_lhs_i(*data, slots); break;
            case GPU_OP_SQRT_LHS: sqrt_lhs_i(*data, slots); break;
            case GPU_OP_NEG_LHS: neg_lhs_i(*data, slots); break;
            case GPU_OP_SIN_LHS: sin_lhs_i(*data, slots); break;
            case GPU_OP_COS_LHS: cos_lhs_i(*data, slots); break;
            case GPU_OP_ASIN_LHS: asin_lhs_i(*data, slots); break;
            case GPU_OP_ACOS_LHS: acos_lhs_i(*data, slots); break;
            case GPU_OP_ATAN_LHS: atan_lhs_i(*data, slots); break;
            case GPU_OP_EXP_LHS: exp_lhs_i(*data, slots); break;
            case GPU_OP_ABS_LHS: abs_lhs_i(*data, slots); break;
            case GPU_OP_LOG_LHS: log_lhs_i(*data, slots); break;

            // Commutative opcodes
            case GPU_OP_ADD_LHS_IMM: add_lhs_imm_i(*data, slots); break;
            case GPU_OP_ADD_LHS_RHS: add_lhs_rhs_i(*data, slots); break;
            case GPU_OP_MUL_LHS_IMM: mul_lhs_imm_i(*data, slots); break;
            case GPU_OP_MUL_LHS_RHS: mul_lhs_rhs_i(*data, slots); break;
            case GPU_OP_MIN_LHS_IMM: min_lhs_imm_i(*data, slots); break;
            case GPU_OP_MIN_LHS_RHS: min_lhs_rhs_i(*data, slots); break;
            case GPU_OP_MAX_LHS_IMM: max_lhs_imm_i(*data, slots); break;
            case GPU_OP_MAX_LHS_RHS: max_lhs_rhs_i(*data, slots); break;

            // Non-commutative opcodes
            case GPU_OP_SUB_LHS_IMM: sub_lhs_imm_i(*data, slots); break;
            case GPU_OP_SUB_IMM_RHS: sub_imm_rhs_i(*data, slots); break;
            case GPU_OP_SUB_LHS_RHS: sub_lhs_rhs_i(*data, slots); break;
            case GPU_OP_DIV_LHS_IMM: div_lhs_imm_i(*data, slots); break;
            case GPU_OP_DIV_IMM_RHS: div_imm_rhs_i(*data, slots); break;
            case GPU_OP_DIV_LHS_RHS: div_lhs_rhs_i(*data, slots); break;

            case GPU_OP_COPY_IMM: copy_imm_i(*data, slots); break;
            case GPU_OP_COPY_LHS: copy_lhs_i(*data, slots); break;
            case GPU_OP_COPY_RHS: copy_rhs_i(*data, slots); break;
        }
        // If this opcode makes a choice, then append that choice to the list
        if (OP(data) >= GPU_OP_MIN_LHS_IMM && OP(data) <= GPU_OP_MAX_LHS_RHS) {
            const uint8_t choice = slots[0].lower();
            choices[choice_index / 16] |= (choice << ((choice_index % 16) * 2));
            choice_index++;
            has_any_choice |= (choice != 0);
        }
    }

    // Check the result
    const uint8_t i_out = I_OUT(data);
#if 0
    printf("%u:%u => [%f %f] [%f %f] [%f %f] => [%f %f]\n",
            threadIdx.x, blockIdx.x,
            in_tiles[tile_index].X.lower(),
            in_tiles[tile_index].X.upper(),
            in_tiles[tile_index].Y.lower(),
            in_tiles[tile_index].Y.upper(),
            in_tiles[tile_index].Z.lower(),
            in_tiles[tile_index].Z.upper(),
            slots[i_out].lower(),
            slots[i_out].upper());
#endif
    if (slots[i_out].lower() > 0.0f) {
        return;
    } else if (slots[i_out].upper() < 0.0f) {
        const uint32_t tile = in_tiles[tile_index].position;
        const uint32_t tx = tile % tiles_per_side;
        const uint32_t ty = (tile / tiles_per_side) % tiles_per_side;
        const uint32_t tz = (tile / tiles_per_side) / tiles_per_side;
        atomicMax(&image[tx + ty * tiles_per_side], tz);
        return;
    }

    // Claim the tile
    out_tile += atomicAdd(out_tile_index, 1);
    out_tile->position = in_tiles[tile_index].position;

    // If the tape won't change at all, then skip pushing it
    // (to save on GPU RAM usage)
    if (!has_any_choice) {
        out_tile->tape = in_tiles[tile_index].tape;
        return;
    }

    // Re-use the slots array for tracking which slots are active
    bool* const __restrict__ active = (bool*)slots;
    memset(active, 0, 128);
    active[i_out] = true;

    // Claim a chunk of tape
    uint64_t out_index = atomicAdd(tape_index, LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE);
    uint64_t out_offset = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
    assert(out_index + out_offset < LIBFIVE_CUDA_NUM_SUBTAPES *
                                    LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE);

    // Write out the end of the tape, which is the same as the ending
    // of the previous tape (0 opcode, with i_out as the last slot)
    out_offset--;
    tape_data[out_index + out_offset] = *data;

    while (OP(--data)) {
        const uint8_t op = OP(data);
        if (op == GPU_OP_JUMP) {
            data += JUMP_TARGET(data);
            continue;
        }

        uint8_t choice = 0;
        if (op >= GPU_OP_MIN_LHS_IMM && op <= GPU_OP_MAX_LHS_RHS) {
            --choice_index;
            choice = (choices[choice_index / 16] >>
                                  ((choice_index % 16) * 2)) & 3;
        }

        const uint8_t i_out = I_OUT(data);
        if (!active[i_out]) {
            continue;
        }

        // If we're about to write a new piece of data to the tape,
        // (and are done with the current chunk), then we need to
        // add another link to the linked list.
        --out_offset;
        if (out_offset == 0) {
            const int32_t prev_index = out_index;
            out_index = atomicAdd(tape_index, LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE);
            out_offset = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
            assert(out_index + out_offset < LIBFIVE_CUDA_NUM_SUBTAPES *
                                            LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE);

            // Forward-pointing link
            OP(&tape_data[out_index + out_offset]) = GPU_OP_JUMP;
            const int32_t delta = (int32_t)prev_index -
                                  (int32_t)out_index;
            JUMP_TARGET(&tape_data[out_index + out_offset]) = delta;

            // Backward-pointing link
            OP(&tape_data[prev_index]) = GPU_OP_JUMP;
            JUMP_TARGET(&tape_data[prev_index]) = -delta;
        }

        active[i_out] = false;
        tape_data[out_index + out_offset] = *data;
        if (choice == 0) {
            const uint8_t i_lhs = I_LHS(data);
            active[i_lhs] = true;
            const uint8_t i_rhs = I_RHS(data);
            active[i_rhs] = true;
        } else if (choice == 1 /* LHS */) {
            // The non-immediate is always the LHS in commutative ops, and
            // min/max (the only clauses that produce a choice) are commutative
            OP(&tape_data[out_index + out_offset]) = GPU_OP_COPY_LHS;
            const uint8_t i_lhs = I_LHS(data);
            active[i_lhs] = true;
        } else if (choice == 2 /* RHS */) {
            const uint8_t i_rhs = I_RHS(data);
            if (i_rhs) {
                OP(&tape_data[out_index + out_offset]) = GPU_OP_COPY_RHS;
                active[i_rhs] = true;
            } else {
                OP(&tape_data[out_index + out_offset]) = GPU_OP_COPY_IMM;
            }
        }
    }

    // Write the beginning of the tape
    out_offset--;
    tape_data[out_index + out_offset] = *data;

    // Record the beginning of the tape in the output tile
    out_tile->tape = out_index + out_offset;
}

////////////////////////////////////////////////////////////////////////////////

__global__
void v2_load_f(const out_tile_t* __restrict__ in_tiles,
               const uint32_t num_in_tiles,
               const uint32_t in_thread_offset,
               const uint32_t tile_size,
               const uint32_t image_size,
               const Eigen::Matrix4f mat,
               float* __restrict__ out)
{
    const uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tile_index = (thread_index + in_thread_offset) / 64;
    if (tile_index >= num_in_tiles) {
        return;
    }

    const uint32_t tiles_per_side = image_size / tile_size;

    const uint32_t in_parent_tile = in_tiles[tile_index].position;
    const uint32_t tx = (in_parent_tile % tiles_per_side);
    const uint32_t ty = ((in_parent_tile / tiles_per_side) % tiles_per_side);
    const uint32_t tz = ((in_parent_tile / tiles_per_side) / tiles_per_side);

    // We subdivide at a constant rate of 4x
    const uint32_t subtile_offset = thread_index % 64;
    const uint32_t px = tx * 4 + subtile_offset % 4;
    const uint32_t py = ty * 4 + (subtile_offset / 4) % 4;
    const uint32_t pz = tz * 4 + (subtile_offset / 4) / 4;

    const float size_recip = 1.0f / image_size;
    const float fx = ((px + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fy = ((py + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fz = ((pz + 0.5f) * size_recip - 0.5f) * 2.0f;

    float fx_, fy_, fz_, fw_;
    fx_ = mat(0, 0) * fx +
          mat(0, 1) * fy +
          mat(0, 2) * fz + mat(0, 3);
    fy_ = mat(1, 0) * fx +
          mat(1, 1) * fy +
          mat(1, 2) * fz + mat(1, 3);
    fz_ = mat(2, 0) * fx +
          mat(2, 1) * fy +
          mat(2, 2) * fz + mat(2, 3);
    fw_ = mat(3, 0) * fx +
          mat(3, 1) * fy +
          mat(3, 2) * fz + mat(3, 3);

    // Projection!
    fx_ = fx_ / fw_;
    fy_ = fy_ / fw_;
    fz_ = fz_ / fw_;

    out[thread_index * 3] = fx_;
    out[thread_index * 3 + 1] = fy_;
    out[thread_index * 3 + 2] = fz_;
}

__global__
void v2_exec_f(const uint64_t* const __restrict__ tapes,
               uint32_t* const __restrict__ image,
               const uint32_t tiles_per_side,

               const out_tile_t* const __restrict__ in_tiles,
               const uint32_t num_in_tiles,
               const uint32_t in_thread_offset,

               const float* __restrict__ values,
               const uint32_t image_size)
{
    float slots[128];

    const uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tile_index = (thread_index + in_thread_offset) / 64;
    if (tile_index >= num_in_tiles) {
        return;
    }

    // Pick out the tape based on the pointer stored in the tiles list
    const uint64_t* __restrict__ data = &tapes[in_tiles[tile_index].tape];

    // Load the axis values (precomputed by v2_load_f)
    slots[((const uint8_t*)data)[1]] = values[thread_index * 3];
    slots[((const uint8_t*)data)[2]] = values[thread_index * 3 + 1];
    slots[((const uint8_t*)data)[3]] = values[thread_index * 3 + 2];

    while (OP(++data)) {
        switch (OP(data)) {
            case GPU_OP_JUMP: data += JUMP_TARGET(data); continue;

            case GPU_OP_SQUARE_LHS: square_lhs_f(*data, slots); break;
            case GPU_OP_SQRT_LHS: sqrt_lhs_f(*data, slots); break;
            case GPU_OP_NEG_LHS: neg_lhs_f(*data, slots); break;
            case GPU_OP_SIN_LHS: sin_lhs_f(*data, slots); break;
            case GPU_OP_COS_LHS: cos_lhs_f(*data, slots); break;
            case GPU_OP_ASIN_LHS: asin_lhs_f(*data, slots); break;
            case GPU_OP_ACOS_LHS: acos_lhs_f(*data, slots); break;
            case GPU_OP_ATAN_LHS: atan_lhs_f(*data, slots); break;
            case GPU_OP_EXP_LHS: exp_lhs_f(*data, slots); break;
            case GPU_OP_ABS_LHS: abs_lhs_f(*data, slots); break;
            case GPU_OP_LOG_LHS: log_lhs_f(*data, slots); break;

            // Commutative opcodes
            case GPU_OP_ADD_LHS_IMM: add_lhs_imm_f(*data, slots); break;
            case GPU_OP_ADD_LHS_RHS: add_lhs_rhs_f(*data, slots); break;
            case GPU_OP_MUL_LHS_IMM: mul_lhs_imm_f(*data, slots); break;
            case GPU_OP_MUL_LHS_RHS: mul_lhs_rhs_f(*data, slots); break;
            case GPU_OP_MIN_LHS_IMM: min_lhs_imm_f(*data, slots); break;
            case GPU_OP_MIN_LHS_RHS: min_lhs_rhs_f(*data, slots); break;
            case GPU_OP_MAX_LHS_IMM: max_lhs_imm_f(*data, slots); break;
            case GPU_OP_MAX_LHS_RHS: max_lhs_rhs_f(*data, slots); break;

            // Non-commutative opcodes
            case GPU_OP_SUB_LHS_IMM: sub_lhs_imm_f(*data, slots); break;
            case GPU_OP_SUB_IMM_RHS: sub_imm_rhs_f(*data, slots); break;
            case GPU_OP_SUB_LHS_RHS: sub_lhs_rhs_f(*data, slots); break;
            case GPU_OP_DIV_LHS_IMM: div_lhs_imm_f(*data, slots); break;
            case GPU_OP_DIV_IMM_RHS: div_imm_rhs_f(*data, slots); break;
            case GPU_OP_DIV_LHS_RHS: div_lhs_rhs_f(*data, slots); break;

            case GPU_OP_COPY_IMM: copy_imm_f(*data, slots); break;
            case GPU_OP_COPY_LHS: copy_lhs_f(*data, slots); break;
            case GPU_OP_COPY_RHS: copy_rhs_f(*data, slots); break;
        }
    }

    // Check the result
    const uint8_t i_out = I_OUT(data);
    if (slots[i_out] < 0.0f) {
        const uint32_t tile = in_tiles[tile_index].position;
        const uint32_t tx = tile % tiles_per_side;
        const uint32_t ty = (tile / tiles_per_side) % tiles_per_side;
        const uint32_t tz = (tile / tiles_per_side) / tiles_per_side;

        // We subdivide at a constant rate of 4x
        const uint32_t subtile_offset = thread_index % 64;
        const uint32_t px = tx * 4 + subtile_offset % 4;
        const uint32_t py = ty * 4 + (subtile_offset / 4) % 4;
        const uint32_t pz = tz * 4 + (subtile_offset / 4) / 4;

        atomicMax(&image[px + py * image_size], pz);
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__
void v2_build_image(uint32_t* __restrict__ image,
                    const uint32_t image_size_px,
                    const uint32_t* __restrict__ tiles,
                    const uint32_t* __restrict__ subtiles,
                    const uint32_t* __restrict__ microtiles)
{
    const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < image_size_px && y < image_size_px) {
        uint32_t t = tiles[x / 64 + y / 64 * (image_size_px / 64)];
        if (t) { t = t * 64 + 63; }

        uint32_t s = subtiles[x / 16 + y / 16 * (image_size_px / 16)];
        if (s) { s = s * 16 + 15; }

        uint32_t u = microtiles[x / 4 + y / 4 * (image_size_px / 4)];
        if (u) { u = u * 4 + 3; }

        const uint32_t p = image[x + y * image_size_px];

        const uint32_t a = (t > s ? t : s);
        const uint32_t b = (u > p ? u : p);
        const uint32_t r = (a > b ? a : b);
        image[x + y * image_size_px] = r;
    }
}

////////////////////////////////////////////////////////////////////////////////

v2_blob_t build_v2_blob(libfive::Tree tree, const uint32_t image_size_px) {
    v2_blob_t out = {0};

    out.tiles.filled      = CUDA_MALLOC(uint32_t, pow(image_size_px / 64, 2));
    out.subtiles.filled   = CUDA_MALLOC(uint32_t, pow(image_size_px / 16, 2));
    out.microtiles.filled = CUDA_MALLOC(uint32_t, pow(image_size_px / 4,  2));

    out.image_size_px = image_size_px;
    out.image = CUDA_MALLOC(uint32_t, pow(image_size_px, 2));

    out.tape_data = CUDA_MALLOC(uint64_t, LIBFIVE_CUDA_NUM_SUBTAPES *
                                          LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE);
    out.tape_index = CUDA_MALLOC(uint32_t, 1);
    *out.tape_index = 0;

    for (stage_t* t : {&out.tiles, &out.subtiles, &out.microtiles}) {
        t->output_index = CUDA_MALLOC(uint32_t, 1);
        *(t->output_index) = 0;
    }

    // The first array of tiles must have enough space to hold all of the
    // 64^3 tiles in the volume, which shouldn't be too much.
    out.tiles.resize_to_fit(pow(out.image_size_px / 64, 3));

    // We leave the other stage_t's input/output arrays unallocated for now,
    // since they're initialized to all zeros and will be resized to fit later.

    // Allocate room for the floating-point values, used in load/exec_f
    out.values = CUDA_MALLOC(float,
                    LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS *
                    LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK * 64 * 3);

    // TAPE PLANNING TIME!
    // Hold a single cache lock to avoid needing mutex locks everywhere
    auto lock = libfive::Cache::instance();

    auto ordered = tree.orderedDfs();

    std::map<libfive::Tree::Id, libfive::Tree::Id> last_used;
    for (auto& c : ordered) {
        if (c->op != libfive::Opcode::CONSTANT) {
            // Very simple tracking of active spans, without clause reordering
            // or any other cleverness.
            last_used[c.lhs().id()] = c.id();
            last_used[c.rhs().id()] = c.id();
        }
    }

    std::vector<uint8_t> free_slots;
    std::map<libfive::Tree::Id, uint8_t> bound_slots;
    uint8_t num_slots = 1;

    auto getSlot = [&](libfive::Tree::Id id) {
        // Pick a slot for the output of this opcode
        uint8_t out;
        if (free_slots.size()) {
            out = free_slots.back();
            free_slots.pop_back();
        } else {
            out = num_slots++;
            if (num_slots == UINT8_MAX) {
                fprintf(stderr, "Ran out of slots!\n");
            }
        }
        bound_slots[id] = out;
        return out;
    };

    // Bind the axes to known slots, so that we can store their values
    // before beginning an evaluation.
    const libfive::Tree axis_trees[3] = {
        libfive::Tree::X(),
        libfive::Tree::Y(),
        libfive::Tree::Z()};
    uint64_t start = 0;
    for (unsigned i=0; i < 3; ++i) {
        if (last_used.find(axis_trees[i].id()) != last_used.end()) {
            ((uint8_t*)&start)[i + 1] = getSlot(axis_trees[i].id());
        }
    }
    std::vector<uint64_t> flat;
    flat.reserve(ordered.size());
    flat.push_back(start);

    auto get_reg = [&](const std::shared_ptr<libfive::Tree::Tree_>& tree) {
        auto itr = bound_slots.find(tree.get());
        if (itr != bound_slots.end()) {
            return itr->second;
        } else {
            fprintf(stderr, "Could not find bound slots");
            return static_cast<uint8_t>(0);
        }
    };

    for (auto& c : ordered) {
        uint64_t clause = 0;
        switch (c->op) {
            using namespace libfive::Opcode;

            case CONSTANT:
            case VAR_X:
            case VAR_Y:
            case VAR_Z:
                continue;

#define OP_UNARY(p) \
            case OP_##p: { \
                OP(&clause) = GPU_OP_##p##_LHS;      \
                I_LHS(&clause) = get_reg(c->lhs);    \
                break;                              \
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
                if (c->lhs->op == CONSTANT) {                   \
                    OP(&clause) = GPU_OP_##p##_LHS_IMM;         \
                    I_LHS(&clause) = get_reg(c->rhs);           \
                    IMM(&clause) = c->lhs->value;               \
                } else if (c->rhs->op == CONSTANT) {            \
                    OP(&clause) = GPU_OP_##p##_LHS_IMM;         \
                    I_LHS(&clause) = get_reg(c->lhs);           \
                    IMM(&clause) = c->rhs->value;               \
                } else {                                        \
                    OP(&clause) = GPU_OP_##p##_LHS_RHS;         \
                    I_LHS(&clause) = get_reg(c->lhs);           \
                    I_RHS(&clause) = get_reg(c->rhs);           \
                }                                               \
                break;                                          \
            }
            OP_COMMUTATIVE(ADD)
            OP_COMMUTATIVE(MUL)
            OP_COMMUTATIVE(MIN)
            OP_COMMUTATIVE(MAX)

#define OP_NONCOMMUTATIVE(p) \
            case OP_##p: { \
                if (c->lhs->op == CONSTANT) {                   \
                    OP(&clause) = GPU_OP_##p##_IMM_RHS;         \
                    I_RHS(&clause) = get_reg(c->rhs);           \
                    IMM(&clause) = c->lhs->value;               \
                } else if (c->rhs->op == CONSTANT) {            \
                    OP(&clause) = GPU_OP_##p##_LHS_IMM;         \
                    I_LHS(&clause) = get_reg(c->lhs);           \
                    IMM(&clause) = c->rhs->value;               \
                } else {                                        \
                    OP(&clause) = GPU_OP_##p##_LHS_RHS;         \
                    I_LHS(&clause) = get_reg(c->lhs);           \
                    I_RHS(&clause) = get_reg(c->rhs);           \
                }                                               \
                break;                                          \
            }
            OP_COMMUTATIVE(SUB)
            OP_COMMUTATIVE(DIV)

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
        for (auto& h : {c.lhs().id(), c.rhs().id()}) {
            if (h != nullptr &&
                h->op != libfive::Opcode::CONSTANT &&
                last_used[h] == c.id())
            {
                auto itr = bound_slots.find(h);
                free_slots.push_back(itr->second);
                bound_slots.erase(itr);
            }
        }

        I_OUT(&clause) = getSlot(c.id());
        flat.push_back(clause);
    }
    {   // Push the end of the tape, which points to the final clauses's
        // output slot so that we know where to read the result.
        uint64_t end = 0;
        I_OUT(&end) = get_reg(ordered.back().operator->());
        flat.push_back(end);
    }

    CUDA_CHECK(cudaMemcpy(out.tape_data, flat.data(),
                          sizeof(uint64_t) * flat.size(),
                          cudaMemcpyHostToDevice));
    out.tape_length = flat.size();

    return out;
}

void free_v2_blob(v2_blob_t blob) {
    cudaFree(blob.image);
    cudaFree(blob.tape_data);
    cudaFree(blob.tape_index);
    cudaFree(blob.values);

    for (auto& t: {blob.tiles, blob.subtiles, blob.microtiles}) {
        cudaFree(t.filled);
        cudaFree(t.input);
        cudaFree(t.output);
        cudaFree(t.output_index);
    }
}


void render_v2_blob(v2_blob_t blob, Eigen::Matrix4f mat) {
    // Reset the tape index
    *blob.tape_index = blob.tape_length;

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 64x64x64 tiles
    ////////////////////////////////////////////////////////////////////////////

    // Reset all of the data arrays
    CUDA_CHECK(cudaMemset(blob.tiles.filled, 0, sizeof(uint32_t) *
                          pow(blob.image_size_px / 64, 2)));
    CUDA_CHECK(cudaMemset(blob.subtiles.filled, 0, sizeof(uint32_t) *
                          pow(blob.image_size_px / 16, 2)));
    CUDA_CHECK(cudaMemset(blob.microtiles.filled, 0, sizeof(uint32_t) *
                          pow(blob.image_size_px / 4, 2)));
    CUDA_CHECK(cudaMemset(blob.image, 0, sizeof(uint32_t) *
                          pow(blob.image_size_px, 2)));
    *blob.tiles.output_index = 0;
    *blob.subtiles.output_index = 0;
    *blob.microtiles.output_index = 0;

    uint32_t stride, count;
    stride = LIBFIVE_CUDA_TILE_BLOCKS * LIBFIVE_CUDA_TILE_THREADS;
    count = pow(blob.image_size_px / 64, 3);
    for (unsigned offset=0; offset < count; offset += stride) {
        // First stage of interval evaluation on every interval
        v2_load_i<<<LIBFIVE_CUDA_TILE_BLOCKS,
                    LIBFIVE_CUDA_TILE_THREADS>>>(
                offset,
                count,
                64,
                blob.image_size_px,
                mat,
                blob.tiles.input);
        v2_exec_universal<<<LIBFIVE_CUDA_TILE_BLOCKS,
                            LIBFIVE_CUDA_TILE_THREADS>>>(
                blob.tape_data,
                blob.tape_index,

                blob.tiles.filled,
                blob.image_size_px / 64,

                blob.tiles.input,
                count,
                offset,

                blob.tiles.output,
                blob.tiles.output_index);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 16x16x16 subtiles
    ////////////////////////////////////////////////////////////////////////////
    count = *blob.tiles.output_index;
    stride = LIBFIVE_CUDA_REFINE_BLOCKS *
             LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK;
    blob.subtiles.resize_to_fit(count * 64);
    for (unsigned offset=0; offset < count; offset += stride) {
        v2_load_s<<<LIBFIVE_CUDA_REFINE_BLOCKS,
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64>>>(
                blob.tiles.output,
                count,
                offset * 64,
                64,
                blob.image_size_px,
                mat,
                blob.subtiles.input);
        v2_exec_universal<<<LIBFIVE_CUDA_REFINE_BLOCKS,
                            LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64>>>(
                blob.tape_data,
                blob.tape_index,

                blob.subtiles.filled,
                blob.image_size_px / 16,

                blob.subtiles.input,
                count * 64,
                offset * 64,

                blob.subtiles.output,
                blob.subtiles.output_index);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 4x4x4 subtiles
    ////////////////////////////////////////////////////////////////////////////
    count = *blob.subtiles.output_index;
    // stride is unchanged
    blob.microtiles.resize_to_fit(count * 64);
    for (unsigned offset=0; offset < count; offset += stride) {
        v2_load_s<<<LIBFIVE_CUDA_REFINE_BLOCKS,
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64>>>(
                blob.subtiles.output,
                count,
                offset * 64,
                16,
                blob.image_size_px,
                mat,
                blob.microtiles.input);
        v2_exec_universal<<<LIBFIVE_CUDA_REFINE_BLOCKS,
                            LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64>>>(
                blob.tape_data,
                blob.tape_index,

                blob.microtiles.filled,
                blob.image_size_px / 4,

                blob.microtiles.input,
                count * 64,
                offset * 64,

                blob.microtiles.output,
                blob.microtiles.output_index);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of individual voxels
    ////////////////////////////////////////////////////////////////////////////
    count = *blob.microtiles.output_index;
    stride = LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS *
             LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK;
    for (unsigned offset=0; offset < count; offset += stride) {
        v2_load_f<<<LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS,
                    LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK * 64>>>(
            blob.microtiles.output,
            count,
            offset * 64,
            4,
            blob.image_size_px,
            mat,
            blob.values);
        v2_exec_f<<<LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS,
                    LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK * 64>>>(
            blob.tape_data,
            blob.image,
            blob.image_size_px / 4,

            blob.microtiles.output,
            count,
            offset,
            blob.values,
            blob.image_size_px);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    {   // Copy all of the filled stuff into the per-pixel image
        const uint32_t u = (blob.image_size_px + 31) / 32;
        v2_build_image<<<dim3(u, u), dim3(32, 32)>>>(
                blob.image,
                blob.image_size_px,
                blob.tiles.filled,
                blob.subtiles.filled,
                blob.microtiles.filled);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////

void stage_t::resize_to_fit(size_t count) {
    if (input_array_size < count) {
        cudaFree(input);
        cudaFree(output);

        input = CUDA_MALLOC(in_tile_t, count);
        input_array_size = count;

        output = CUDA_MALLOC(out_tile_t, count);
        *output_index = 0;
    }
}
