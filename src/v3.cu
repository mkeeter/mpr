#include <cassert>

#include "libfive/tree/cache.hpp"

#include "v3.hpp"
#include "check.hpp"
#include "gpu_interval.hpp"
#include "gpu_opcode.hpp"

// No need for parameters.hpp, we want to compile faster
// (without rebuilding everything else)
#define NUM_THREADS (64 * 2)
#define NUM_BLOCKS (512)
#define SUBTAPE_CHUNK_SIZE 64
#define NUM_SUBTAPES 3200000

////////////////////////////////////////////////////////////////////////////////
// COPYPASTA
#define OP(d) (((uint8_t*)d)[0])
#define I_OUT(d) (((uint8_t*)d)[1])
#define I_LHS(d) (((uint8_t*)d)[2])
#define I_RHS(d) (((uint8_t*)d)[3])
#define IMM(d) (((float*)d)[1])
#define JUMP_TARGET(d) (((int32_t*)d)[1])

static __device__ void copy_imm_i(const uint64_t data,
                                  Interval* const __restrict__ slots)
{
    const float lhs = IMM(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = {lhs, lhs};
}

static __device__ void copy_imm_f(const uint64_t data,
                                  float* const __restrict__ slots)
{
    const float lhs = IMM(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = lhs;
}

static __device__ void copy_lhs_i(const uint64_t data,
                                  Interval* const __restrict__ slots)
{
    const uint8_t i_lhs = I_LHS(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = slots[i_lhs];
}

static __device__ void copy_lhs_f(const uint64_t data,
                                  float* const __restrict__ slots)
{
    const uint8_t i_lhs = I_LHS(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = slots[i_lhs];
}

static __device__ void copy_rhs_i(const uint64_t data,
                                  Interval* const __restrict__ slots)
{
    const uint8_t i_rhs = I_RHS(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = slots[i_rhs];
}

static __device__ void copy_rhs_f(const uint64_t data,
                                  float* const __restrict__ slots)
{
    const uint8_t i_rhs = I_RHS(&data);
    const uint8_t i_out = I_OUT(&data);
    slots[i_out] = slots[i_rhs];
}

#define FUNCTION_PREAMBLE_LHS(name, T, suffix)              \
static __device__                                           \
void name##_lhs_##suffix(const uint64_t data,               \
                    T* const __restrict__ slots)            \
{                                                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs];                             \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_LHS_ARRAY(name, T, suffix)        \
static __device__                                           \
void name##_lhs_##suffix(const uint64_t data,               \
                   T (*__restrict__ const slots)[64])       \
{                                                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs][threadIdx.x % 64];           \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_IMM_RHS(name, T, suffix)          \
static __device__                                           \
void name##_imm_rhs_##suffix(const uint64_t data,           \
                    T* const __restrict__ slots)            \
{                                                           \
    const float lhs = IMM(&data);                           \
    const uint8_t i_rhs = I_RHS(&data);                     \
    const T rhs = slots[i_rhs];                             \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_IMM_RHS_ARRAY(name, T, suffix)    \
static __device__                                           \
void name##_imm_rhs_##suffix(const uint64_t data,           \
                      T (*__restrict__ const slots)[64])    \
{                                                           \
    const float lhs = IMM(&data);                           \
    const uint8_t i_rhs = I_RHS(&data);                     \
    const T rhs = slots[i_rhs][threadIdx.x % 64];           \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_LHS_IMM(name, T, suffix)          \
static __device__                                           \
void name##_lhs_imm_##suffix(const uint64_t data,           \
                    T* const __restrict__ slots)            \
{                                                           \
    const float rhs = IMM(&data);                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs];                             \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_LHS_IMM_ARRAY(name, T, suffix)    \
static __device__                                           \
void name##_lhs_imm_##suffix(const uint64_t data,           \
                      T (*__restrict__ const slots)[64])    \
{                                                           \
    const float rhs = IMM(&data);                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs][threadIdx.x % 64];           \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_LHS_RHS(name, T, suffix)          \
static __device__                                           \
void name##_lhs_rhs_##suffix(const uint64_t data,           \
                    T* const __restrict__ slots)            \
{                                                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs];                             \
    const uint8_t i_rhs = I_RHS(&data);                     \
    const T rhs = slots[i_rhs];                             \
    const uint8_t i_out = I_OUT(&data);                     \

#define FUNCTION_PREAMBLE_LHS_RHS_ARRAY(name, T, suffix)    \
static __device__                                           \
void name##_lhs_rhs_##suffix(const uint64_t data,           \
                    T (*const __restrict__ slots)[64])      \
{                                                           \
    const uint8_t i_lhs = I_LHS(&data);                     \
    const T lhs = slots[i_lhs][threadIdx.x % 64];           \
    const uint8_t i_rhs = I_RHS(&data);                     \
    const T rhs = slots[i_rhs][threadIdx.x % 64];           \
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
void v3_preload_tiles(v3_tile_node_t* const __restrict__ in_tiles,
                      const int32_t in_tile_count,
                      const int32_t offset)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count) {
        return;
    }

    in_tiles[tile_index].position = tile_index + offset;
    in_tiles[tile_index].tape = 0;
    in_tiles[tile_index].next = -1;
}

__global__
void v3_calculate_intervals(const v3_tile_node_t* const __restrict__ in_tiles,
                            const uint32_t num_in_tiles,
                            const uint32_t tiles_per_side,
                            const Eigen::Matrix4f mat,
                            Interval* const __restrict__ values)
{
    const uint32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= num_in_tiles) {
        return;
    }

    const uint32_t tile = in_tiles[tile_index].position;

    const uint32_t x = tile % tiles_per_side;
    const Interval ix = {(x / (float)tiles_per_side - 0.5f) * 2.0f,
                   ((x + 1) / (float)tiles_per_side - 0.5f) * 2.0f};
    const uint32_t y = (tile / tiles_per_side) % tiles_per_side;
    const Interval iy = {(y / (float)tiles_per_side - 0.5f) * 2.0f,
                   ((y + 1) / (float)tiles_per_side - 0.5f) * 2.0f};
    const uint32_t z = (tile / tiles_per_side) / tiles_per_side;
    const Interval iz = {(z / (float)tiles_per_side - 0.5f) * 2.0f,
                   ((z + 1) / (float)tiles_per_side - 0.5f) * 2.0f};

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

    values[tile_index * 3] = ix_;
    values[tile_index * 3 + 1] = iy_;
    values[tile_index * 3 + 2] = iz_;
}

__global__
void v3_eval_tiles_i(const uint64_t* const __restrict__ tape_data,
                     int32_t* const __restrict__ image,
                     const uint32_t tiles_per_side,

                     v3_tile_node_t* const __restrict__ in_tiles,
                     const int32_t in_tile_count,

                     const Interval* __restrict__ values,

                     v3_tape_push_data_t* const __restrict__ push_data)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count) {
        return;
    }

    Interval slots[128];
    slots[((const uint8_t*)tape_data)[1]] = values[tile_index * 3];
    slots[((const uint8_t*)tape_data)[2]] = values[tile_index * 3 + 1];
    slots[((const uint8_t*)tape_data)[3]] = values[tile_index * 3 + 2];

    // Pick out the tape based on the pointer stored in the tiles list
    const uint64_t* __restrict__ data = &tape_data[in_tiles[tile_index].tape];

    uint32_t choices[256] = {0};
    int choice_index = 0;
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

            default: assert(false);
        }
        // If this opcode makes a choice, then append that choice to the list
        if (OP(data) >= GPU_OP_MIN_LHS_IMM && OP(data) <= GPU_OP_MAX_LHS_RHS) {
            const uint8_t c = slots[0].v.x;
            choices[choice_index / 16] |= (c << ((choice_index % 16) * 2));
            choice_index++;
            has_any_choice |= (c != 0);
        }
    }

    // Check the result
    const uint8_t i_out = I_OUT(data);

    if (slots[i_out].lower() > 0.0f) {
        in_tiles[tile_index].position = -1;
        return;
    }

    // Filled
    if (slots[i_out].upper() < 0.0f) {
        const int32_t tile = in_tiles[tile_index].position;
        const int32_t txy = tile % (tiles_per_side * tiles_per_side);
        const int32_t tz  = tile / (tiles_per_side * tiles_per_side);
        in_tiles[tile_index].position = -1;
        atomicMax(&image[txy], tz);
        return;
    }

    if (has_any_choice) {
        // Copy the choice data to global memory so that we can push the tape
        // in a separate kernel, after compacting the list of active tiles.
        for (int i=0; i < (choice_index + 15) / 16; ++i) {
            push_data[tile_index].choices[i] = choices[i];
        }
        push_data[tile_index].choice_index = choice_index;
        push_data[tile_index].tape_end = data - tape_data;
    } else {
        push_data[tile_index].choice_index = -1;
    }
}

__global__
void v3_mask_filled_tiles(int32_t* const __restrict__ image,
                          const uint32_t tiles_per_side,

                          v3_tile_node_t* const __restrict__ in_tiles,
                          const int32_t in_tile_count)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count) {
        return;
    }

    const int32_t tile = in_tiles[tile_index].position;
    // Already marked as filled or empty
    if (tile == -1) {
        return;
    }

    const int32_t txy = tile % (tiles_per_side * tiles_per_side);
    const int32_t tz  = tile / (tiles_per_side * tiles_per_side);

    // If this tile is completely masked by the image, then skip it
    if (image[txy] > tz) {
        in_tiles[tile_index].position = -1;
    }
}

// Accumulates a list of active tiles/tapes which need pushing,
// storing tile thread indexes in tapes_to_push and the total
// count in num_tapes_to_push.
__global__
void v3_plan_tape_push(const v3_tile_node_t* const __restrict__ in_tiles,
                       const int32_t in_tile_count,

                       // Compute in the previous step
                       const v3_tape_push_data_t* __restrict__ const push_data,

                       int32_t* __restrict__ const tapes_to_push,
                       int32_t* __restrict__ const num_tapes_to_push)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;

    const bool needs_pushing = tile_index < in_tile_count &&
                               in_tiles[tile_index].position != -1 &&
                               push_data[tile_index].choice_index != -1;

    // Do two levels of accumulation, to reduce atomic pressure on a single
    // global variable.  Does this help?  Who knows!
    __shared__ int local_offset;
    if (threadIdx.x == 0) {
        local_offset = 0;
    }
    __syncthreads();

    int my_offset;
    if (needs_pushing) {
        my_offset = atomicAdd(&local_offset, 1);
    }
    __syncthreads();

    // Only one thread gets to contribute to the global offset
    if (threadIdx.x == 0) {
        local_offset = atomicAdd(num_tapes_to_push, local_offset);
    }
    __syncthreads();

    if (needs_pushing) {
        tapes_to_push[local_offset + my_offset] = tile_index;
    }
}

__global__
void v3_execute_tape_push(v3_tile_node_t* const __restrict__ in_tiles,
                          uint64_t* const __restrict__ tape_data,
                          int32_t* const __restrict__ tape_index,

                          const int32_t* __restrict__ const tapes_to_push,
                          const int32_t* __restrict__ const num_tapes_to_push,
                          const v3_tape_push_data_t* const __restrict__ push_data)
{
    const int32_t target_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (target_index >= *num_tapes_to_push) {
        return;
    }
    const int32_t tile_index = tapes_to_push[target_index];

    // Copy choices from global to a thread-local array
    uint32_t choices[256];
    int choice_index = push_data[tile_index].choice_index;
    for (int i=0; i < (choice_index + 15) / 16; ++i) {
        choices[i] = push_data[tile_index].choices[i];
    }

    // We start with the cursor positioned at the end of the tape
    const uint64_t* __restrict__ data = tape_data + push_data[tile_index].tape_end;
    const uint8_t i_out = I_OUT(data);

    // Use this array to track which slots are active
    bool active[128] = {0};
    active[i_out] = true;

    // Claim a chunk of tape
    int32_t out_index = atomicAdd(tape_index, SUBTAPE_CHUNK_SIZE);
    int32_t out_offset = SUBTAPE_CHUNK_SIZE;
    assert(out_index + out_offset < NUM_SUBTAPES *
                                    SUBTAPE_CHUNK_SIZE);

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

        const bool has_choice = op >= GPU_OP_MIN_LHS_IMM &&
                                op <= GPU_OP_MAX_LHS_RHS;
        choice_index -= has_choice;

        const uint8_t i_out = I_OUT(data);
        if (!active[i_out]) {
            continue;
        }

        assert(!has_choice || choice_index >= 0);

        const uint8_t choice = has_choice
            ? ((choices[choice_index / 16] >>
              ((choice_index % 16) * 2)) & 3)
            : 0;

        // If we're about to write a new piece of data to the tape,
        // (and are done with the current chunk), then we need to
        // add another link to the linked list.
        --out_offset;
        if (out_offset == 0) {
            const int32_t prev_index = out_index;
            out_index = atomicAdd(tape_index, SUBTAPE_CHUNK_SIZE);
            out_offset = SUBTAPE_CHUNK_SIZE;
            assert(out_index + out_offset < NUM_SUBTAPES *
                                            SUBTAPE_CHUNK_SIZE);
            --out_offset;

            // Forward-pointing link
            OP(&tape_data[out_index + out_offset]) = GPU_OP_JUMP;
            const int32_t delta = (int32_t)prev_index -
                                  (int32_t)(out_index + out_offset);
            JUMP_TARGET(&tape_data[out_index + out_offset]) = delta;

            // Backward-pointing link
            OP(&tape_data[prev_index]) = GPU_OP_JUMP;
            JUMP_TARGET(&tape_data[prev_index]) = -delta;

            // We've written the jump, so adjust the offset again
            --out_offset;
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
    in_tiles[tile_index].tape = out_index + out_offset;
}

// Sets the tile.next to an index in the upcoming tile list, without
// actually doing any work (since that list may not be allocated yet)
__global__
void v3_assign_next_nodes(v3_tile_node_t* const __restrict__ in_tiles,
                          const int32_t in_tile_count,

                          int32_t* __restrict__ const num_active_tiles)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count) {
        return;
    }

    const bool is_active = tile_index < in_tile_count &&
                           in_tiles[tile_index].position != -1;

    // Do two levels of accumulation, to reduce atomic pressure on a single
    // global variable.  Does this help?  Who knows!
    __shared__ int local_offset;
    if (threadIdx.x == 0) {
        local_offset = 0;
    }
    __syncthreads();

    int my_offset;
    if (is_active) {
        my_offset = atomicAdd(&local_offset, 1);
    }
    __syncthreads();

    // Only one thread gets to contribute to the global offset
    if (threadIdx.x == 0) {
        local_offset = atomicAdd(num_active_tiles, local_offset);
    }
    __syncthreads();

    if (is_active) {
        in_tiles[tile_index].next = local_offset + my_offset;
    } else {
        in_tiles[tile_index].next = -1;
    }
}

// Copies each active tile into 64 subtiles
__global__
void v3_subdivide_tiles(const v3_tile_node_t* const __restrict__ in_tiles,
                        const int32_t in_tile_count,
                        v3_tile_node_t* const __restrict__ out_tiles)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count || in_tiles[tile_index].next == -1) {
        return;
    }

    int t = in_tiles[tile_index].next * 64;
    for (int i=0; i < 64; ++i) {
        out_tiles[t + i].position = 0; // TODO
        out_tiles[t + i].tape = in_tiles[tile_index].tape;
        out_tiles[t + i].next = -1;
    }
}
////////////////////////////////////////////////////////////////////////////////

v3_blob_t build_v3_blob(libfive::Tree tree, const int32_t image_size_px) {
    v3_blob_t out = {0};

    out.tiles.filled      = CUDA_MALLOC(int32_t, pow(image_size_px / 64, 2));
    out.subtiles.filled   = CUDA_MALLOC(int32_t, pow(image_size_px / 16, 2));
    out.microtiles.filled = CUDA_MALLOC(int32_t, pow(image_size_px / 4,  2));

    out.image_size_px = image_size_px;
    out.image = CUDA_MALLOC(int32_t, pow(image_size_px, 2));

    out.tape_data = CUDA_MALLOC(uint64_t, NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE);
    out.tape_index = CUDA_MALLOC(int32_t, 1);
    *out.tape_index = 0;

    // Allocate an index to keep track of active tiles
    out.num_active_tiles = CUDA_MALLOC(int32_t, 1);

    // Allocate temporary storage in global memory to calculate pushed tapes
    out.push_data = CUDA_MALLOC(v3_tape_push_data_t, NUM_THREADS * NUM_BLOCKS);
    out.push_target_buffer = CUDA_MALLOC(int32_t, NUM_THREADS * NUM_BLOCKS);
    out.push_target_count = CUDA_MALLOC(int32_t, 1);

    // Allocate a bunch of scratch space for passing intervals around
    out.values = CUDA_MALLOC(Interval, NUM_THREADS * NUM_BLOCKS * 3);

    // The first array of tiles must have enough space to hold all of the
    // 64^3 tiles in the volume, which shouldn't be too much.
    out.tiles.tiles = CUDA_MALLOC(v3_tile_node_t, pow(out.image_size_px / 64, 3));

    // We leave the other stage_t's input/output arrays unallocated for now,
    // since they're initialized to all zeros and will be resized to fit later.

    // TAPE PLANNING TIME!
    // Hold a single cache lock to avoid needing mutex locks everywhere
    auto lock = libfive::Cache::instance();

    auto ordered = tree.orderedDfs();

    std::map<libfive::Tree::Id, libfive::Tree::Id> last_used;
    bool axes_used[3] = {false, false, false};
    for (auto& c : ordered) {
        if (c->op != libfive::Opcode::CONSTANT) {
            // Very simple tracking of active spans, without clause reordering
            // or any other cleverness.
            last_used[c.lhs().id()] = c.id();
            last_used[c.rhs().id()] = c.id();
        }
        axes_used[0] |= c == libfive::Tree::X();
        axes_used[1] |= c == libfive::Tree::Y();
        axes_used[2] |= c == libfive::Tree::Z();
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
        if (axes_used[i]) {
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

void free_v3_blob(v3_blob_t& blob) {
    CUDA_CHECK(cudaFree(blob.tiles.filled));
    CUDA_CHECK(cudaFree(blob.subtiles.filled));
    CUDA_CHECK(cudaFree(blob.microtiles.filled));

    CUDA_CHECK(cudaFree(blob.image));

    CUDA_CHECK(cudaFree(blob.tape_data));
    CUDA_CHECK(cudaFree(blob.tape_index));

    CUDA_CHECK(cudaFree(blob.num_active_tiles));

    CUDA_CHECK(cudaFree(blob.push_data));
    CUDA_CHECK(cudaFree(blob.push_target_buffer));
    CUDA_CHECK(cudaFree(blob.push_target_count));

    CUDA_CHECK(cudaFree(blob.values));

    CUDA_CHECK(cudaFree(blob.tiles.tiles));
    CUDA_CHECK(cudaFree(blob.subtiles.tiles));
    CUDA_CHECK(cudaFree(blob.microtiles.tiles));
}

////////////////////////////////////////////////////////////////////////////////

void render_v3_blob(v3_blob_t& blob, Eigen::Matrix4f mat) {
    // Reset the tape index
    *blob.tape_index = blob.tape_length;

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 64x64x64 tiles
    ////////////////////////////////////////////////////////////////////////////

    // Reset all of the data arrays
    CUDA_CHECK(cudaMemset(blob.tiles.filled, 0, sizeof(int32_t) *
                          pow(blob.image_size_px / 64, 2)));
    CUDA_CHECK(cudaMemset(blob.subtiles.filled, 0, sizeof(int32_t) *
                          pow(blob.image_size_px / 16, 2)));
    CUDA_CHECK(cudaMemset(blob.microtiles.filled, 0, sizeof(int32_t) *
                          pow(blob.image_size_px / 4, 2)));
    CUDA_CHECK(cudaMemset(blob.image, 0, sizeof(int32_t) *
                          pow(blob.image_size_px, 2)));

    uint32_t stride, count;

    // Go the whole list of first-stage tiles, assigning each to
    // be [position, tape = 0, next = -1]
    stride = NUM_BLOCKS * NUM_THREADS;
    count = pow(blob.image_size_px / 64, 3);
    for (unsigned offset=0; offset < count; offset += stride) {
        v3_preload_tiles<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.tiles.tiles + offset,
            std::min(stride, count - offset),
            offset);
    }

    // Now loop through doing evaluation, one batch at a time
    for (unsigned offset=0; offset < count; offset += stride) {
        // Unpack position values into interval X/Y/Z in the values array
        v3_calculate_intervals<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.tiles.tiles + offset,
            std::min(stride, count - offset),
            blob.image_size_px / 64,
            mat,
            (Interval*)blob.values);

        // Do the actual tape evaluation, which is the expensive step
        v3_eval_tiles_i<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.tape_data,
            blob.tiles.filled,
            blob.image_size_px / 64,

            blob.tiles.tiles + offset,
            std::min(stride, count - offset),

            (Interval*)blob.values,
            blob.push_data);

        // Mark every tile which is covered in the image as masked,
        // which means it will be skipped later on.
        v3_mask_filled_tiles<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.tiles.filled, blob.image_size_px / 64,
            blob.tiles.tiles + offset, std::min(stride, count - offset));

        // Reset the push target count to 0, without requiring
        // a sync with the host.
        cudaMemsetAsync(blob.push_target_count, 0, sizeof(int32_t));

        // Figure out which tapes need pushing, storing them in the
        // push buffer and recording the count (with atomic increments)
        // in push_count.
        v3_plan_tape_push<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.tiles.tiles + offset,
            std::min(stride, count - offset),

            blob.push_data,

            blob.push_target_buffer,
            blob.push_target_count);

        // Actually do the pushing.  There will be lots of threads at
        // the end of this (> push_count) which aren't doing anything.
        v3_execute_tape_push<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.tiles.tiles + offset,
            blob.tape_data,
            blob.tape_index,

            blob.push_target_buffer,
            blob.push_target_count,
            blob.push_data);

        // Count up active tiles, to figure out how much memory needs to be
        // allocated in the next stage.
        cudaMemsetAsync(blob.num_active_tiles, 0, sizeof(int32_t));
        v3_assign_next_nodes<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.tiles.tiles + offset,
            std::min(stride, count - offset),
            blob.num_active_tiles);
    }

    // Make sure that the subtiles buffer has enough room
    CUDA_CHECK(cudaDeviceSynchronize());
    const int32_t num_active_tiles = *blob.num_active_tiles;
    if (num_active_tiles * 64 > blob.subtiles.tile_array_size) {
        blob.subtiles.tile_array_size = num_active_tiles * 64;
        CUDA_CHECK(cudaFree(blob.subtiles.tiles));
        blob.subtiles.tiles = CUDA_MALLOC(
                v3_tile_node_t, num_active_tiles * 64);
    }

    // Build the new tape from the active tiles in the previous tape
    for (unsigned offset=0; offset < count; offset += stride) {
        v3_subdivide_tiles<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.tiles.tiles + offset,
            std::min(stride, count - offset),
            blob.subtiles.tiles + offset * 64);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

}

// END OF EXPERIMENTAL ZONE
////////////////////////////////////////////////////////////////////////////////
