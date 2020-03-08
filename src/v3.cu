#include <cassert>

#include "libfive/tree/cache.hpp"

#include "v3.hpp"
#include "check.hpp"
#include "gpu_interval.hpp"
#include "gpu_opcode.hpp"

////////////////////////////////////////////////////////////////////////////////

// No need for parameters.hpp, we want to compile faster
// (without rebuilding everything else)
#define NUM_TILES (2)
#define NUM_THREADS (64 * NUM_TILES)
#define NUM_BLOCKS (512)
#define SUBTAPE_CHUNK_SIZE 64
#define NUM_SUBTAPES 320000

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

#define FUNCTION_PREAMBLE_IMM_RHS(name, T, suffix)          \
static __device__                                           \
void name##_imm_rhs_##suffix(const uint64_t data,           \
                    T* const __restrict__ slots)            \
{                                                           \
    const float lhs = IMM(&data);                           \
    const uint8_t i_rhs = I_RHS(&data);                     \
    const T rhs = slots[i_rhs];                             \
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
static inline __device__
int4 unpack(int32_t pos, int32_t tiles_per_side)
{
    return make_int4(pos % tiles_per_side,
                    (pos / tiles_per_side) % tiles_per_side,
                    (pos / tiles_per_side) / tiles_per_side,
                     pos % (tiles_per_side * tiles_per_side));
}

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
                            const uint32_t in_tile_count,
                            const uint32_t tiles_per_side,
                            const Eigen::Matrix4f mat,
                            Interval* const __restrict__ values)
{
    const uint32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count) {
        return;
    }

    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    const Interval ix = {(pos.x / (float)tiles_per_side - 0.5f) * 2.0f,
                   ((pos.x + 1) / (float)tiles_per_side - 0.5f) * 2.0f};
    const Interval iy = {(pos.y / (float)tiles_per_side - 0.5f) * 2.0f,
                   ((pos.y + 1) / (float)tiles_per_side - 0.5f) * 2.0f};
    const Interval iz = {(pos.z / (float)tiles_per_side - 0.5f) * 2.0f,
                   ((pos.z + 1) / (float)tiles_per_side - 0.5f) * 2.0f};

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
void v3_eval_tiles_i(uint64_t* const __restrict__ tape_data,
                     int32_t* const __restrict__ tape_index,
                     int32_t* const __restrict__ image,
                     const uint32_t tiles_per_side,

                     v3_tile_node_t* const __restrict__ in_tiles,
                     const int32_t in_tile_count,

                     const Interval* __restrict__ values)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count) {
        return;
    }

    // Check to see if we're masked
    if (in_tiles[tile_index].position == -1) {
        return;
    }

    Interval slots[128];
    slots[((const uint8_t*)tape_data)[1]] = values[tile_index * 3];
    slots[((const uint8_t*)tape_data)[2]] = values[tile_index * 3 + 1];
    slots[((const uint8_t*)tape_data)[3]] = values[tile_index * 3 + 2];

    // Pick out the tape based on the pointer stored in the tiles list
    const uint64_t* __restrict__ data = &tape_data[in_tiles[tile_index].tape];

    uint32_t choices[128] = {0};
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
#if 0
    printf("%u:%u: [%f %f] [%f %f] [%f %f] => [%f %f]\n",
            blockIdx.x, threadIdx.x,
            values[tile_index * 3].lower(),
            values[tile_index * 3].upper(),
            values[tile_index * 3 + 1].lower(),
            values[tile_index * 3 + 1].upper(),
            values[tile_index * 3 + 2].lower(),
            values[tile_index * 3 + 2].upper(),
            slots[i_out].lower(),
            slots[i_out].upper());
#endif

    if (slots[i_out].lower() > 0.0f) {
        in_tiles[tile_index].position = -1;
        return;
    }

    // Masked
    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    if (image[pos.w] > pos.z) {
        in_tiles[tile_index].position = -1;
        return;
    }

    // Filled
    if (slots[i_out].upper() < 0.0f) {
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        in_tiles[tile_index].position = -1;
        atomicMax(&image[pos.w], pos.z);
        return;
    }

    if (!has_any_choice) {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Tape pushing!
    // Use this array to track which slots are active
    int* const __restrict__ active = (int*)slots;
    for (unsigned i=0; i < 128; ++i) {
        active[i] = false;
    }
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
            const uint8_t i_lhs = I_LHS(data);
            if (i_lhs == i_out) {
                ++out_offset;
            } else {
                OP(&tape_data[out_index + out_offset]) = GPU_OP_COPY_LHS;
            }
            active[i_lhs] = true;
        } else if (choice == 2 /* RHS */) {
            const uint8_t i_rhs = I_RHS(data);
            if (i_rhs) {
                if (i_rhs == i_out) {
                    ++out_offset;
                } else {
                    OP(&tape_data[out_index + out_offset]) = GPU_OP_COPY_RHS;
                }
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

////////////////////////////////////////////////////////////////////////////////

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

    const int4 pos = unpack(tile, tiles_per_side);

    // If this tile is completely masked by the image, then skip it
    if (image[pos.w] > pos.z) {
        in_tiles[tile_index].position = -1;
    }
}

////////////////////////////////////////////////////////////////////////////////

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
void v3_subdivide_active_tiles(
        const v3_tile_node_t* const __restrict__ in_tiles,
        const int32_t in_tile_count,
        const int32_t tiles_per_side,
        v3_tile_node_t* const __restrict__ out_tiles)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count || in_tiles[tile_index].next == -1) {
        return;
    }

    const int t = in_tiles[tile_index].next * 64;
    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    const int32_t subtiles_per_side = tiles_per_side * 4;

    for (int i=0; i < 64; ++i) {
        const int4 sub = unpack(i, 4);
        const int32_t sx = pos.x * 4 + sub.x;
        const int32_t sy = pos.y * 4 + sub.y;
        const int32_t sz = pos.z * 4 + sub.z;
        const int32_t next_tile =
            sx +
            sy * subtiles_per_side +
            sz * subtiles_per_side * subtiles_per_side;

        out_tiles[t + i].position = next_tile;
        out_tiles[t + i].tape = in_tiles[tile_index].tape;
        out_tiles[t + i].next = -1;
    }
}

// Copies each active tile into the out_tiles list, clearing its `next` value.
// This is used right before per-pixel evaluation, which wants a compact list
// of active tiles, but doesn't need to subdivide them by 64 itself.
__global__
void v3_copy_active_tiles(v3_tile_node_t* const __restrict__ in_tiles,
                          const int32_t in_tile_count,
                          const int32_t tiles_per_side,
                          v3_tile_node_t* const __restrict__ out_tiles)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count || in_tiles[tile_index].next == -1) {
        return;
    }
    const int t = in_tiles[tile_index].next;
    out_tiles[t].position = in_tiles[tile_index].position;
    out_tiles[t].tape = in_tiles[tile_index].tape;
    out_tiles[t].next = -1;
    in_tiles[tile_index].next = -1;
}

////////////////////////////////////////////////////////////////////////////////

__global__
void v3_copy_filled(const int32_t* __restrict__ prev,
                    int32_t* __restrict__ image,
                    const int32_t image_size_px)
{
    const int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < image_size_px && y < image_size_px) {
        int32_t t = prev[x / 4 + y / 4 * (image_size_px / 4)];
        if (t) {
            image[x + y * image_size_px] = t * 4 + 3;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

__global__
void v3_eval_voxels_f(const uint64_t* const __restrict__ tape_data,
                      int32_t* const __restrict__ image,
                      const uint32_t tiles_per_side,

                      v3_tile_node_t* const __restrict__ in_tiles,
                      const int32_t in_tile_count,
                      Eigen::Matrix4f mat)
{
    // Each tile is executed by 64 threads (one for each voxel).
    //
    // This is different from the v3_eval_tiles_i function, which evaluates one
    // tile per thread, because the tiles are already expanded by 64x by the
    // time they're stored in the in_tiles list.
    const int32_t voxel_index = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t tile_index = voxel_index / 64;
    if (tile_index >= in_tile_count) {
        return;
    }

    float slots[128];

    {   // Load values into registers, subdividing by 4x on each axis
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        const int4 sub = unpack(threadIdx.x % 64, 4);

        const int32_t px = pos.x * 4 + sub.x;
        const int32_t py = pos.y * 4 + sub.y;
        const int32_t pz = pos.z * 4 + sub.z;

        // Early return if this pixel won't ever be filled
        if (image[px + py * tiles_per_side * 4] >= pz) {
            return;
        }

        const float size_recip = 1.0f / (tiles_per_side * 4);

        const float fx = ((px + 0.5f) * size_recip - 0.5f) * 2.0f;
        const float fy = ((py + 0.5f) * size_recip - 0.5f) * 2.0f;
        const float fz = ((pz + 0.5f) * size_recip - 0.5f) * 2.0f;

        // Otherwise, calculate the X/Y/Z values
        const float fw_ = mat(3, 0) * fx +
                          mat(3, 1) * fy +
                          mat(3, 2) * fz + mat(3, 3);
        for (unsigned i=0; i < 3; ++i) {
            slots[((const uint8_t*)tape_data)[i + 1]] =
                (mat(i, 0) * fx +
                 mat(i, 1) * fy +
                 mat(i, 2) * fz + mat(0, 3)) / fw_;
        }
    }

    // Pick out the tape based on the pointer stored in the tiles list
    const uint64_t* __restrict__ data = &tape_data[in_tiles[tile_index].tape];

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
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        const int4 sub = unpack(threadIdx.x % 64, 4);
        const int32_t px = pos.x * 4 + sub.x;
        const int32_t py = pos.y * 4 + sub.y;
        const int32_t pz = pos.z * 4 + sub.z;

        atomicMax(&image[px + py * tiles_per_side * 4], pz);
    }
}


////////////////////////////////////////////////////////////////////////////////

v3_blob_t build_v3_blob(libfive::Tree tree, const int32_t image_size_px) {
    v3_blob_t out = {0};

    for (unsigned i=0; i < 4; ++i) {
        const unsigned tile_size_px = 64 / (1 << (i * 2));
        out.stages[i].filled = CUDA_MALLOC(
                int32_t,
                pow(image_size_px / tile_size_px, 2));
    }

    out.image_size_px = image_size_px;

    out.tape_data = CUDA_MALLOC(uint64_t, NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE);
    out.tape_index = CUDA_MALLOC(int32_t, 1);
    *out.tape_index = 0;

    // Allocate an index to keep track of active tiles
    out.num_active_tiles = CUDA_MALLOC(int32_t, 1);

    // Allocate a bunch of scratch space for passing intervals around
    out.values = CUDA_MALLOC(Interval, NUM_THREADS * NUM_BLOCKS * 3);

    // The first array of tiles must have enough space to hold all of the
    // 64^3 tiles in the volume, which shouldn't be too much.
    out.stages[0].tiles = CUDA_MALLOC(
            v3_tile_node_t,
            pow(out.image_size_px / 64, 3));

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
    for (unsigned i=0; i < 4; ++i) {
        CUDA_FREE(blob.stages[i].filled);
        CUDA_FREE(blob.stages[i].tiles);
    }

    CUDA_FREE(blob.tape_data);
    CUDA_FREE(blob.tape_index);

    CUDA_FREE(blob.num_active_tiles);

    CUDA_FREE(blob.values);
}

////////////////////////////////////////////////////////////////////////////////

void render_v3_blob(v3_blob_t& blob, Eigen::Matrix4f mat) {
    // Reset the tape index
    *blob.tape_index = blob.tape_length;

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 64x64x64 tiles
    ////////////////////////////////////////////////////////////////////////////

    // Reset all of the data arrays
    for (unsigned i=0; i < 4; ++i) {
        const unsigned tile_size_px = 64 / (1 << (i * 2));
        CUDA_CHECK(cudaMemset(blob.stages[i].filled, 0, sizeof(int32_t) *
                              pow(blob.image_size_px / tile_size_px, 2)));
    }

    // Go the whole list of first-stage tiles, assigning each to
    // be [position, tape = 0, next = -1]
    unsigned stride = NUM_BLOCKS * NUM_THREADS;
    unsigned count = pow(blob.image_size_px / 64, 3);
    for (unsigned offset=0; offset < count; offset += stride) {
        v3_preload_tiles<<<NUM_BLOCKS, NUM_THREADS>>>(
            blob.stages[0].tiles + offset,
            std::min(stride, count - offset),
            offset);
    }

    // Iterate over 64^3, 16^3, 4^3 tiles
    for (unsigned i=0; i < 3; ++i) {
        //printf("BEGINNING STAGE %u\n", i);
        const unsigned tile_size_px = 64 / (1 << (i * 2));

        // Now loop through doing evaluation, one batch at a time
        for (unsigned offset=0; offset < count; offset += stride) {
            /*
            printf("    looping with offset = %u, stride = %u, count = %u\n",
                    offset, stride, count);
            */
            const int active_threads = std::min(stride, count - offset);
            const int active_blocks = (active_threads + NUM_THREADS - 1) / NUM_THREADS;

            // Unpack position values into interval X/Y/Z in the values array
            // This is done in a separate kernel to avoid bloating the
            // eval_tiles_i kernel with more registers, which is detrimental
            // to occupancy.
            v3_calculate_intervals<<<active_blocks, NUM_THREADS>>>(
                blob.stages[i].tiles + offset,
                active_threads,
                blob.image_size_px / tile_size_px,
                mat,
                (Interval*)blob.values);

            // Mark every tile which is covered in the image as masked,
            // which means it will be skipped later on.  We do this again below,
            // but it's basically free, so we should do it here and simplify
            // the logic in eval_tiles_i.
            v3_mask_filled_tiles<<<active_blocks, NUM_THREADS>>>(
                blob.stages[i].filled,
                blob.image_size_px / tile_size_px,
                blob.stages[i].tiles + offset,
                active_threads);

            // Do the actual tape evaluation, which is the expensive step
            v3_eval_tiles_i<<<active_blocks, NUM_THREADS>>>(
                blob.tape_data,
                blob.tape_index,
                blob.stages[i].filled,
                blob.image_size_px / tile_size_px,

                blob.stages[i].tiles + offset,
                active_threads,

                (Interval*)blob.values);
        }

        // Mark the total number of active tiles (from this stage) to 0
        cudaMemsetAsync(blob.num_active_tiles, 0, sizeof(int32_t));

        // Now that we have evaluated every tile at this level, we do one more
        // round of occlusion culling before accumulating tiles to render at
        // the next phase.
        for (unsigned offset=0; offset < count; offset += stride) {
            const int active_threads = std::min(stride, count - offset);
            const int active_blocks = (active_threads + NUM_THREADS - 1) / NUM_THREADS;

            v3_mask_filled_tiles<<<active_blocks, NUM_THREADS>>>(
                blob.stages[i].filled,
                blob.image_size_px / tile_size_px,
                blob.stages[i].tiles + offset,
                active_threads);

            // Count up active tiles, to figure out how much memory needs to be
            // allocated in the next stage.
            v3_assign_next_nodes<<<active_blocks, NUM_THREADS>>>(
                blob.stages[i].tiles + offset,
                active_threads,
                blob.num_active_tiles);
        }

        // Count the number of active tiles, which have been accumulated
        // through repeated calls to v3_assign_next_nodes
        int32_t num_active_tiles;
        cudaMemcpy(&num_active_tiles, blob.num_active_tiles, sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        if (i < 2) {
            num_active_tiles *= 64;
        }

        // Make sure that the subtiles buffer has enough room
        if (num_active_tiles > blob.stages[i + 1].tile_array_size) {
            blob.stages[i + 1].tile_array_size = num_active_tiles;
            CUDA_FREE(blob.stages[i + 1].tiles);
            blob.stages[i + 1].tiles = CUDA_MALLOC(
                    v3_tile_node_t, num_active_tiles);
        }

        if (i < 2) {
            // Build the new tile list from active tiles in the previous list
            for (unsigned offset=0; offset < count; offset += stride) {
                const int active_threads = std::min(stride, count - offset);
                const int active_blocks = (active_threads + NUM_THREADS - 1) / NUM_THREADS;

                v3_subdivide_active_tiles<<<active_blocks, NUM_THREADS>>>(
                    blob.stages[i].tiles + offset,
                    active_threads,
                    blob.image_size_px / tile_size_px,
                    blob.stages[i + 1].tiles);
            }
        } else {
            // Special case for per-pixel evaluation, which
            // doesn't unpack every single pixel (since that would take up
            // 64x extra space).
            for (unsigned offset=0; offset < count; offset += stride) {
                const int active_threads = std::min(stride, count - offset);
                const int active_blocks = (active_threads + NUM_THREADS - 1) / NUM_THREADS;

                v3_copy_active_tiles<<<active_blocks, NUM_THREADS>>>(
                    blob.stages[i].tiles + offset,
                    active_threads,
                    blob.image_size_px / tile_size_px,
                    blob.stages[i + 1].tiles);
            }
        }

        {   // Copy filled tiles into the next level's image (expanding them
            // by 64x).  This is cleaner that accumulating all of the levels
            // in a single pass, and could (possibly?) help with skipping
            // fully occluded tiles.
            const unsigned next_tile_size = tile_size_px / 4;
            const uint32_t u = ((blob.image_size_px / next_tile_size) / 32);
            v3_copy_filled<<<dim3(u + 1, u + 1), dim3(32, 32)>>>(
                    blob.stages[i].filled,
                    blob.stages[i + 1].filled,
                    blob.image_size_px / next_tile_size);
        }

        // Assign the next number of tiles to evaluate
        count = num_active_tiles;

        /*
        printf("------------------------------------------------------------\n");
        printf("Done with stage %u with %u tiles to do\n",
                i, count);
                */
    }

    // Time to render individual pixels!
    stride = NUM_BLOCKS * NUM_TILES;
    for (unsigned offset=0; offset < count; offset += stride) {
        //printf("Rendering pixels with offset %u, count %u\n", offset, count);
        const int active_tiles = std::min(stride, count - offset);
        const int active_blocks = (active_tiles*64 + NUM_THREADS - 1) / NUM_THREADS;

        v3_eval_voxels_f<<<active_blocks, NUM_THREADS>>>(
            blob.tape_data,
            blob.stages[3].filled,
            blob.image_size_px / 4,

            blob.stages[3].tiles + offset,
            active_tiles,

            mat);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// END OF EXPERIMENTAL ZONE
////////////////////////////////////////////////////////////////////////////////
