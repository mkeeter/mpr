#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

#include "v2.hpp"
#include "check.hpp"
#include "gpu_interval.hpp"
#include "gpu_opcode.hpp"
#include "parameters.hpp"

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
    uint8_t choice;
    slots[i_out] = min(lhs, rhs, choice);
    slots[0].v.x = choice;
}
FUNCTION_PREAMBLE_LHS_RHS(min, float, f)
    slots[i_out] = fminf(lhs, rhs);
}
FUNCTION_PREAMBLE_LHS_RHS(min, Interval, i)
    uint8_t choice;
    slots[i_out] = min(lhs, rhs, choice);
    slots[0].v.x = choice;
}

FUNCTION_PREAMBLE_LHS_IMM(max, float, f)
    slots[i_out] = fmaxf(lhs, rhs);
}
FUNCTION_PREAMBLE_LHS_IMM(max, Interval, i)
    uint8_t choice;
    slots[i_out] = max(lhs, rhs, choice);
    slots[0].v.x = choice;
}
FUNCTION_PREAMBLE_LHS_RHS(max, float, f)
    slots[i_out] = fmaxf(lhs, rhs);
}
FUNCTION_PREAMBLE_LHS_RHS(max, Interval, i)
    uint8_t choice;
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

struct in_tile_t {
    uint32_t position;
    uint32_t tape;
    Interval X, Y, Z;
};

struct out_tile_t {
    uint32_t position;
    uint32_t tape;
};

__global__
void v2_load_i(uint32_t tile_offset,
               uint32_t tile_size, uint32_t image_size,
               const Eigen::Matrix4f mat,
               in_tile_t* __restrict__ out)
{
    // Load the axis values
    const uint32_t thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tile = thread_index + tile_offset;
    const uint32_t tiles_per_side = image_size / tile_size;

    if (tile >= tiles_per_side * tiles_per_side * tiles_per_side) {
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
               const uint32_t tile_size, const uint32_t image_size,
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
    uint32_t tx = (in_parent_tile % tiles_per_side);
    uint32_t ty = ((in_parent_tile / tiles_per_side) % tiles_per_side);
    uint32_t tz = ((in_parent_tile / tiles_per_side) / tiles_per_side);

    // We subdivide at a constant rate of 4x
    const uint32_t subtile_size = tile_size / 4;
    const uint32_t subtiles_per_side = tiles_per_side * 4;
    const uint32_t subtile_offset = thread_index % 64;
    tx = tx * 4 + subtile_offset % 4;
    ty = ty * 4 + (subtile_offset / 4) % 4;
    tz = tz * 4 + (subtile_offset / 4) / 4;

    const float size_recip = 1.0f / image_size;
    const Interval ix = {(tx * subtile_size * size_recip - 0.5f) * 2.0f,
                   ((tx + 1) * subtile_size * size_recip - 0.5f) * 2.0f};
    const Interval iy = {(ty * subtile_size * size_recip - 0.5f) * 2.0f,
                   ((ty + 1) * subtile_size * size_recip - 0.5f) * 2.0f};
    const Interval iz = {(tz * subtile_size * size_recip - 0.5f) * 2.0f,
                   ((tz + 1) * subtile_size * size_recip - 0.5f) * 2.0f};

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

    out_tiles[thread_index].X = ix_;
    out_tiles[thread_index].Y = iy_;
    out_tiles[thread_index].Z = iz_;
    out_tiles[thread_index].position =
        tx +
        ty * subtiles_per_side +
        tz * subtiles_per_side * subtiles_per_side;
    out_tiles[thread_index].tape = in_tiles[tile_index].tape;
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
    data++;

    uint32_t choices[256] = {0};
    unsigned choice_index = 0;
    bool has_any_choice = false;

    while (OP(data)) {
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
        data++;
    }

    // Check the result
    const uint8_t i_out = I_OUT(data);
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
    uint64_t out_index = atomicAdd(tape_index, 64);
    uint64_t out_offset = 64;
    assert(out_index + out_offset < LIBFIVE_CUDA_NUM_SUBTAPES * 64);

    // Write out the end of the tape, which is a 0 opcode and the i_out
    out_offset--;
    OP(&tape_data[out_index + out_offset]) = 0;
    I_OUT(&tape_data[out_index + out_offset]) = i_out;

    data--;
    while (OP(data)) {
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
        if (active[i_out]) {
            // If we're about to write a new piece of data to the tape,
            // (and are done with the current chunk), then we need to
            // add another link to the linked list.
            if (out_offset == 1) {
                const uint64_t prev_index = out_index;
                out_index = atomicAdd(tape_index, 64);
                out_offset = 64;
                assert(out_index + out_offset < LIBFIVE_CUDA_NUM_SUBTAPES * 64);

                // Forward-pointing link
                out_offset--;
                OP(&tape_data[out_index + out_offset]) = GPU_OP_JUMP;
                const int32_t delta = (int32_t)prev_index -
                                      (int32_t)out_index + 1;
                JUMP_TARGET(&tape_data[out_index + out_offset]) = delta;

                // Backward-pointing link
                OP(&tape_data[prev_index]) = GPU_OP_JUMP;
                JUMP_TARGET(&tape_data[prev_index]) = -delta;
            }
            out_offset--;
            active[i_out] = false;
            tape_data[out_index + out_offset] = *data;
            if (choice == 0) {
                const uint8_t i_lhs = I_LHS(data);
                active[i_lhs] = true;
                const uint8_t i_rhs = I_RHS(data);
                active[i_rhs] = true;
            } else if (choice == 1 /* LHS */) {
                // The non-immediate is always the LHS in commutative ops,
                // and min/max are commutative
                OP(&tape_data[out_index + out_offset]) = GPU_OP_COPY_LHS;
                const uint8_t i_lhs = I_LHS(data);
                active[i_lhs] = true;
            } else if (choice == 2 /* RHS */) {
                const uint8_t i_rhs = I_RHS(data);
                if (i_rhs) {
                    OP(tape_data[out_index + out_offset]) = GPU_OP_COPY_RHS;
                } else {
                    OP(tape_data[out_index + out_offset]) = GPU_OP_COPY_IMM;
                }
            }
        }
        data--;
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
    uint32_t tx = (in_parent_tile % tiles_per_side);
    uint32_t ty = ((in_parent_tile / tiles_per_side) % tiles_per_side);
    uint32_t tz = ((in_parent_tile / tiles_per_side) / tiles_per_side);

    // We subdivide at a constant rate of 4x
    const uint32_t subtile_offset = thread_index % 64;
    tx = tx * 4 + subtile_offset % 4;
    ty = ty * 4 + (subtile_offset / 4) % 4;
    tz = tz * 4 + (subtile_offset / 4) / 4;

    const float size_recip = 1.0f / image_size;
    const float fx = ((tx + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fy = ((ty + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fz = ((tz + 0.5f) * size_recip - 0.5f) * 2.0f;

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

               const out_tile_t* const __restrict__ in_tiles,
               const uint32_t num_in_tiles,
               const uint32_t in_thread_offset,

               const float* __restrict__ values,
               const uint32_t tiles_per_side)
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
    data++;

    while (OP(data)) {
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
    const uint8_t i_out = data[1];
    if (slots[i_out] < 0.0f) {
        const uint32_t in_parent_tile = in_tiles[tile_index].position;
        uint32_t px = (in_parent_tile % tiles_per_side);
        uint32_t py = ((in_parent_tile / tiles_per_side) % tiles_per_side);
        uint32_t pz = ((in_parent_tile / tiles_per_side) / tiles_per_side);

        // We subdivide at a constant rate of 4x
        const uint32_t pixels_per_side = tiles_per_side * 4;
        const uint32_t subtile_offset = thread_index % 64;
        px = px * 4 + subtile_offset % 4;
        py = py * 4 + (subtile_offset / 4) % 4;
        pz = pz * 4 + (subtile_offset / 4) / 4;

        atomicMax(&image[px + py * pixels_per_side], pz);
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
        const uint32_t t = tiles[x / 64 + y / 64 * (image_size_px / 64)];
        const uint32_t s = subtiles[x / 16 + y / 16 * (image_size_px / 16)];
        const uint32_t u = microtiles[x / 4 + y / 4 * (image_size_px / 4)];
        const uint32_t p = image[x + y * image_size_px];

        const uint32_t a = (t > s ? t : s);
        const uint32_t b = (u > p ? u : p);
        const uint32_t r = (a > b ? a : b);
        image[x + y * image_size_px] = r;
    }
}

////////////////////////////////////////////////////////////////////////////////
#if 0
// Brute-force evaluator for the 2D case
__global__
void v2_exec_2d(const uint64_t* __restrict__ data,
                uint32_t* __restrict__ image,
                uint32_t size, float size_recip)
{
    float slots[128];

    const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Load the axis values
    uint16_t i_out = *((const uint16_t*)&data[1]);
    if (i_out != UINT16_MAX) {
        slots[i_out & 0x3FFF] = ((x * size_recip) - 0.5f) * 2.0f;
    }
    i_out = *((const uint16_t*)&data[1] + 1);
    if (i_out != UINT16_MAX) {
        slots[i_out & 0x3FFF] = ((y * size_recip) - 0.5f) * 2.0f;
    }
    i_out = *((const uint16_t*)&data[1] + 2);
    if (i_out != UINT16_MAX) {
        slots[i_out & 0x3FFF] = 0.0f;
    }
    data += 2;

    while (data[0]) {
        // Check for jumps (not yet implemented in the tape)
        if (data[0] & FLAG_JUMPS) {
            data = (const uint64_t*)(data[1]);
        } else {
            (OperationFunc(data[0] & ~7))(data[1], slots);
            data += 2;
        }
    }

    // Check the result
    i_out = data[1];
    if (x < size && y < size && slots[i_out] < 0.0f) {
        image[size * y + x] = 1;
    } else {
        image[size * y + x] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////

__global__
void v2_translate(const Tape* __restrict__ tape,
                  uint64_t* const __restrict__ out)
{
    // Only write clauses within the tape
    const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t* const __restrict__ data = out + index * 2 + 1;

    // We reserve the 0 slot for choices, so add 1 to all registers here
    if (index == 0) {
        // Null termination of the tape beginning, plus axis registers
        out[index * 2] = 0;
        *((uint16_t*)data) = (*tape).axes.reg[0] + 1;
        *((uint16_t*)data + 1) = (*tape).axes.reg[1] + 1;
        *((uint16_t*)data + 2) = (*tape).axes.reg[2] + 1;
    } else if (index == tape->num_clauses + 1) {
        // Null termination of the tape, plus the output position
        out[index * 2] = 0;
        *data = (*tape)[tape->num_clauses - 1].out + 1;
    } else if (index < tape->num_clauses + 1) {
        const Clause c = (*tape)[index - 1];
        using namespace libfive::Opcode;
#define CASE_COMMUTATIVE_(opcode, name, f) \
        case opcode: {                                                      \
            if (c.banks == 0) {                                             \
                out[index * 2] = (uint64_t)&name##_arg_arg | f;             \
                *(uint16_t*)data = c.lhs + 1;                               \
                *((uint16_t*)data + 2) = c.rhs + 1;                         \
                *((uint16_t*)data + 3) = c.out + 1;                         \
            } else if (c.banks == 1) {                                      \
                out[index * 2] = (uint64_t)&name##_imm_arg | f | FLAG_IMM;  \
                *(float*)data = (*tape).constant(c.lhs);                    \
                *((uint16_t*)data + 2) = c.rhs + 1;                         \
                *((uint16_t*)data + 3) = c.out + 1;                         \
            } else if (c.banks == 2) {                                      \
                out[index * 2] = (uint64_t)&name##_imm_arg | f | FLAG_IMM; \
                *(float*)data = (*tape).constant(c.rhs);                    \
                *((uint16_t*)data + 2) = c.lhs + 1;                         \
                *((uint16_t*)data + 3) = c.out + 1;                         \
            } else {                                                        \
                assert(false);                                              \
            }                                                               \
            break;                                                          \
        }
#define CASE_COMMUTATIVE(opcode, name) CASE_COMMUTATIVE_(opcode, name, 0)
#define CASE_COMMUTATIVE_CHOICE(opcode, name) CASE_COMMUTATIVE_(opcode, name, FLAG_CHOICE)

#define CASE_NONCOMMUTATIVE(opcode, name) \
        case opcode: {                                                      \
            if (c.banks == 0) {                                             \
                out[index * 2] = (uint64_t)&name##_arg_arg;                 \
                *(uint16_t*)data = c.lhs + 1;                               \
                *((uint16_t*)data + 2) = c.rhs + 1;                         \
                *((uint16_t*)data + 3) = c.out + 1;                         \
            } else if (c.banks == 1) {                                      \
                out[index * 2] = (uint64_t)&name##_imm_arg | FLAG_IMM;      \
                *(float*)data = (*tape).constant(c.lhs);                    \
                *((uint16_t*)data + 2) = c.rhs + 1;                         \
                *((uint16_t*)data + 3) = c.out + 1;                         \
            } else if (c.banks == 2) {                                      \
                out[index * 2] = (uint64_t)&name##_arg_imm | FLAG_IMM;      \
                *(float*)data = (*tape).constant(c.rhs);                    \
                *((uint16_t*)data + 2) = c.lhs + 1;                         \
                *((uint16_t*)data + 3) = c.out + 1;                         \
            } else {                                                        \
                assert(false);                                              \
            }                                                               \
            break;                                                          \
        }

#define CASE_UNARY(opcode, name) \
        case opcode: {                                      \
            if (c.banks == 0) {                             \
                out[index * 2] = (uint64_t)&name##_arg;     \
                *(uint16_t*)data = c.lhs + 1;               \
                *((uint16_t*)data + 2) = c.lhs + 1;         \
                *((uint16_t*)data + 3) = c.out + 1;         \
            } else {                                        \
                assert(false);                              \
            }                                               \
            break;                                          \
        }

        switch (c.opcode) {
            CASE_COMMUTATIVE(OP_ADD, add);
            CASE_COMMUTATIVE(OP_MUL, mul);
            CASE_COMMUTATIVE_CHOICE(OP_MIN, min);
            CASE_COMMUTATIVE_CHOICE(OP_MAX, max);
            CASE_NONCOMMUTATIVE(OP_SUB, sub);
            CASE_NONCOMMUTATIVE(OP_DIV, div);


            CASE_UNARY(OP_SQUARE, square);
            CASE_UNARY(OP_SQRT, sqrt);
            CASE_UNARY(OP_NEG, neg);
            CASE_UNARY(OP_ABS, abs);

            CASE_UNARY(OP_ASIN, asin);
            CASE_UNARY(OP_ACOS, acos);
            CASE_UNARY(OP_ATAN, atan);
            CASE_UNARY(OP_EXP, exp);
            CASE_UNARY(OP_SIN, sin);
            CASE_UNARY(OP_COS, cos);
            CASE_UNARY(OP_LOG, log);
            default: assert(false);
        }
    }
}

uint64_t* build_v2_tape(const Tape& tape, const uint32_t size) {
    const uint32_t u = (tape.num_clauses + 31) / 32;
    uint64_t* data = CUDA_MALLOC(uint64_t, (tape.num_clauses + 2) * 2);
    v2_translate<<<u, 32>>>(&tape, data);
    cudaDeviceSynchronize();
    return data;
}

void eval_v2_tape(const uint64_t* data, uint32_t* image, uint32_t size) {
    const uint32_t u = (size + 15) / 16;
    v2_exec_2d<<<dim3(u, u), dim3(16, 16)>>>(data, image, size, 1.0f / size);
    cudaDeviceSynchronize();
}


v2_blob_t build_v2_blob(const Tape& tape, const uint32_t image_size_px) {
    v2_blob_t out = {0};

    out.filled_tiles      = CUDA_MALLOC(uint32_t, pow(image_size_px / 64, 2));
    out.filled_subtiles   = CUDA_MALLOC(uint32_t, pow(image_size_px / 16, 2));
    out.filled_microtiles = CUDA_MALLOC(uint32_t, pow(image_size_px / 4,  2));

    out.image_size_px = image_size_px;
    out.image = CUDA_MALLOC(uint32_t, pow(image_size_px,  2));

    {   // Convert the tape to a tape with pointers, etc
        const uint32_t u = (tape.num_clauses + 31) / 32;
        out.tape = CUDA_MALLOC(uint64_t, (tape.num_clauses + 2) * 2);
        v2_translate<<<u, 32>>>(&tape, out.tape);
    }

    // Allocate these three indexes in a single chunk
    out.subtape_index = CUDA_MALLOC(uint32_t, 3);
    out.ping_index = out.subtape_index + 1;
    out.pong_index = out.subtape_index + 2;

    out.subtapes = CUDA_MALLOC(uint64_t, LIBFIVE_CUDA_NUM_SUBTAPES * 64);

    cudaDeviceSynchronize();
    *out.subtape_index = 0;

    *out.ping_index = 0;
    out.ping_queue = NULL;
    out.ping_queue_len = 0;

    *out.pong_index = 0;
    out.pong_queue = NULL;
    out.pong_queue_len = 0;

    out.values = CUDA_MALLOC(float, 3 * 2 * LIBFIVE_CUDA_REFINE_BLOCKS * LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64);
    return out;
}

void free_v2_blob(v2_blob_t blob) {
    cudaFree(blob.filled_tiles);
    cudaFree(blob.filled_subtiles);
    cudaFree(blob.filled_microtiles);

    cudaFree(blob.tape);
    cudaFree(blob.subtape_index);
}

void render_v2_blob(v2_blob_t blob, Eigen::Matrix4f mat) {
    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 64x64x64 tiles
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t max_tiles = pow(blob.image_size_px / 64, 3);
    if (blob.ping_queue_len < max_tiles * 2) {
        cudaFree(blob.ping_queue);
        blob.ping_queue = CUDA_MALLOC(uint32_t, max_tiles * 2);
        blob.ping_queue_len = max_tiles * 2;
        *blob.ping_index = 0;
    }
    CUDA_CHECK(cudaMemset(blob.filled_tiles, 0,
                          pow(blob.image_size_px / 64, 2) * sizeof(uint32_t)));

    *blob.subtape_index = 0;

    uint32_t stride, count;

    stride = LIBFIVE_CUDA_TILE_BLOCKS * LIBFIVE_CUDA_TILE_THREADS;
    count = pow(blob.image_size_px / 64, 3);
    for (unsigned offset=0; offset < count; offset += stride) {
        // First stage of interval evaluation on every interval
        v2_load_i<<<LIBFIVE_CUDA_TILE_BLOCKS, LIBFIVE_CUDA_TILE_THREADS>>>(
                offset, 64, blob.image_size_px,
                mat, (Interval*)blob.values);
        v2_exec_i<<<LIBFIVE_CUDA_TILE_BLOCKS, LIBFIVE_CUDA_TILE_THREADS>>>(
                blob.tape,
                blob.filled_tiles,
                blob.subtapes, blob.subtape_index,
                blob.ping_queue, blob.ping_index,
                offset,
                (const Interval*)blob.values,
                blob.image_size_px / 64);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 16x16x16 subtiles
    ////////////////////////////////////////////////////////////////////////////

    // Resize pong queue to fit subtiles
    const uint32_t ambiguous_tile_count = *blob.ping_index / 2;
    if (blob.pong_queue_len < ambiguous_tile_count * 2 * 64) {
        cudaFree(blob.pong_queue);
        blob.pong_queue = CUDA_MALLOC(uint32_t, ambiguous_tile_count * 2 * 64);
        blob.pong_queue_len = ambiguous_tile_count * 2 * 64;
        *blob.pong_index = 0;
    }
    CUDA_CHECK(cudaMemset(blob.filled_subtiles, 0,
                          pow(blob.image_size_px / 16, 2) * sizeof(uint32_t)));
    stride = LIBFIVE_CUDA_REFINE_BLOCKS * LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK;
    count = ambiguous_tile_count;
    for (unsigned offset=0; offset < count; offset += stride) {
        v2_load_s<<<LIBFIVE_CUDA_REFINE_BLOCKS,
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64>>>(
                blob.ping_queue, ambiguous_tile_count,
                offset * 64, 64, blob.image_size_px,
                mat, (Interval*)blob.values);
        v2_exec_s<<<LIBFIVE_CUDA_REFINE_BLOCKS,
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64>>>(
                blob.subtapes, blob.subtape_index,
                blob.filled_subtiles,
                blob.pong_queue, blob.pong_index,
                blob.ping_queue, ambiguous_tile_count, offset * 64,
                (Interval*)blob.values,
                blob.image_size_px / 64);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 4x4x4 microtiles
    ////////////////////////////////////////////////////////////////////////////

    // Resize ping queue to fit microtiles
    const uint32_t ambiguous_subtile_count = *blob.pong_index / 2;
    if (blob.ping_queue_len < ambiguous_subtile_count * 2 * 64) {
        cudaFree(blob.ping_queue);
        blob.ping_queue = CUDA_MALLOC(uint32_t, ambiguous_subtile_count * 2 * 64);
        blob.ping_queue_len = ambiguous_subtile_count * 2 * 64;
        *blob.ping_index = 0;
    }
    CUDA_CHECK(cudaMemset(blob.filled_microtiles, 0,
                          pow(blob.image_size_px / 4, 2) * sizeof(uint32_t)));
    stride = LIBFIVE_CUDA_REFINE_BLOCKS * LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK;
    count = ambiguous_subtile_count;
    for (unsigned offset=0; offset < count; offset += stride) {
        v2_load_s<<<LIBFIVE_CUDA_REFINE_BLOCKS,
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64>>>(
                blob.pong_queue, ambiguous_subtile_count,
                offset * 64, 16, blob.image_size_px,
                mat, (Interval*)blob.values);
        v2_exec_s<<<LIBFIVE_CUDA_REFINE_BLOCKS,
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK * 64>>>(
                blob.subtapes, blob.subtape_index,
                blob.filled_microtiles,
                blob.ping_queue, blob.ping_index,
                blob.pong_queue, ambiguous_subtile_count, offset * 64,
                (Interval*)blob.values,
                blob.image_size_px / 16);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of individual voxels
    ////////////////////////////////////////////////////////////////////////////
    CUDA_CHECK(cudaMemset(blob.image, 0,
                          pow(blob.image_size_px, 2) * sizeof(uint32_t)));

    const uint32_t ambiguous_microtile_count = *blob.ping_index / 2;
    stride = LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS * LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK;
    count = ambiguous_microtile_count;
    for (unsigned offset=0; offset < count; offset += stride) {
        v2_load_f<<<LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS,
                    LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK * 64>>>(
                blob.ping_queue, ambiguous_microtile_count,
                offset * 64, 4, blob.image_size_px,
                mat, (float*)blob.values);
        v2_exec_f<<<LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS,
                    LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK * 64>>>(
                blob.subtapes,
                blob.image,
                blob.ping_queue, ambiguous_microtile_count, offset * 64,
                (float*)blob.values,
                blob.image_size_px / 4);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    {   // Copy all of the filled stuff into the per-pixel image
        const uint32_t u = (blob.image_size_px + 31) / 32;
        v2_build_image<<<dim3(u, u), dim3(32, 32)>>>(
                blob.image,
                blob.image_size_px,
                blob.filled_tiles,
                blob.filled_subtiles,
                blob.filled_microtiles);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif
