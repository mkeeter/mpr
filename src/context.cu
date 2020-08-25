/*
Reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include "clause.hpp"
#include "context.hpp"
#include "parameters.hpp"
#include "tape.hpp"

#include "gpu_deriv.hpp"
#include "gpu_interval.hpp"
#include "gpu_opcode.hpp"

using namespace mpr;

static inline __device__
int4 unpack(int32_t pos, int32_t tiles_per_side)
{
    return make_int4(pos % tiles_per_side,
                    (pos / tiles_per_side) % tiles_per_side,
                    (pos / tiles_per_side) / tiles_per_side,
                     pos % (tiles_per_side * tiles_per_side));
}

////////////////////////////////////////////////////////////////////////////////

/*
 *  preload_tiles
 *
 *  Fills the array at `in_tiles` with the values 0 through `in_tile_count`
 *  For each TileNode, sets its position to the appropriate value, its tape
 *  to 0 (for the default tape), and its `next` pointer to -1 (indicating
 *  that there is no following node, yet).
 *
 *  This function should be called before the first stage of per-tile
 *  evaluation, when we want to evaluate every single top-level tile.
 */
__global__
void preload_tiles(TileNode* const __restrict__ in_tiles,
                   const int32_t in_tile_count)
{
    const int32_t tile_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (tile_index >= in_tile_count) {
        return;
    }

    in_tiles[tile_index].position = tile_index;
    in_tiles[tile_index].tape = 0;
    in_tiles[tile_index].next = -1;
}

/*
 *  calculate_intervals
 *
 *  For tiles 0 through `in_tile_count` in the `in_tiles` array, calculates
 *  their position in render space (+/-1 on each axis, orthographic,
 *  screen-aligned), then applies the transform specified by `mat` and writes
 *  the results to the `values` array.
 *
 *  The values array is packed as triples, i.e. [X0 Y0 Z0 X1 Y1 Z1 ...]
 *
 *  This function could theoretically take place at the beginning of
 *  eval_tiles_i, but making it a separate kernel reduces register usage
 *  below the magic value of 32, which is needed to keep 100% occupancy.
 *
 *  One would be tempted to make this a __device__ __noinline__ function,
 *  then call it in eval_tiles_i (making it __noinline__ fixes the issue of
 *  register bloat).  Unfortunately, this reduces performance, at least on
 *  my laptop.
 */
__global__
void calculate_intervals_3d(const TileNode* const __restrict__ in_tiles,
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
void calculate_intervals_2d(const TileNode* const __restrict__ in_tiles,
                            const uint32_t in_tile_count,
                            const uint32_t tiles_per_side,
                            const Eigen::Matrix3f mat,
                            const float z,
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

    Interval ix_, iy_, iw_;
    ix_ = mat(0, 0) * ix +
          mat(0, 1) * iy +
          mat(0, 2);
    iy_ = mat(1, 0) * ix +
          mat(1, 1) * iy +
          mat(1, 2);
    iw_ = mat(2, 0) * ix +
          mat(2, 1) * iy +
          mat(2, 2);

    // Projection!
    ix_ = ix_ / iw_;
    iy_ = iy_ / iw_;

    values[tile_index * 3] = ix_;
    values[tile_index * 3 + 1] = iy_;
    values[tile_index * 3 + 2] = {z, z};
}

/*
 *  eval_tiles_i
 *
 *  This is the important one!
 *
 *  We take a bunch (`in_tile_count`) of tiles in the `in_tiles` array.  Their
 *  values must already be stored in `values` by `calculate_intervals`.
 *
 *  Each tile in the array specifies which tape to use, where tapes are stored
 *  as chunked linked lists in `tape_data`.  By construction, tiles evaluated
 *  by the same warp should have the same tape, which prevents divergence.
 *
 *  Each thread walks the tape for its tile values.  If the resulting interval
 *  is filled, then it records that result in the `image` output, using an
 *  `atomicMax` operation to prevent memory issues.  If the interval is empty,
 *  then it returns immediately.  In both of these cases, the tile's `position`
 *  is set to -1, to indicate that it does not require further processing.
 *
 *  Otherwise, it walks *backwards* through the tile's tape, creating a new
 *  tape which only contains active clauses.  This is done with an algorithm
 *  similiar to the "mark" phase of "mark-and-sweep": each clause marks its
 *  children as active, except for min/max clauses, which have the option to
 *  only mark one branch.
 *
 *  The new tape is written to the tile's `tape` variable, because it is valid
 *  for any evaluation which takes place within the tile.
 */
template <int DIMENSION>
__global__
void eval_tiles_i(uint64_t* const __restrict__ tape_data,
                  int32_t* const __restrict__ tape_index,
                  int32_t* const __restrict__ image,
                  const uint32_t tiles_per_side,

                  TileNode* const __restrict__ in_tiles,
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

    constexpr static int CHOICE_ARRAY_SIZE = 256;
    uint32_t choices[CHOICE_ARRAY_SIZE] = {0};
    int choice_index = 0;
    bool has_any_choice = false;

    while (1) {
        const uint64_t d = *++data;
        if (!OP(&d)) {
            break;
        }
        switch (OP(&d)) {
            case GPU_OP_JUMP: data += JUMP_TARGET(&d); continue;

#define lhs slots[I_LHS(&d)]
#define rhs slots[I_RHS(&d)]
#define imm IMM(&d)
#define out slots[I_OUT(&d)]

            case GPU_OP_SQUARE_LHS: out = square(lhs); break;
            case GPU_OP_SQRT_LHS:   out = sqrt(lhs); break;
            case GPU_OP_NEG_LHS:    out = -lhs; break;
            case GPU_OP_SIN_LHS:    out = sin(lhs); break;
            case GPU_OP_COS_LHS:    out = cos(lhs); break;
            case GPU_OP_ASIN_LHS:   out = asin(lhs); break;
            case GPU_OP_ACOS_LHS:   out = acos(lhs); break;
            case GPU_OP_ATAN_LHS:   out = atan(lhs); break;
            case GPU_OP_EXP_LHS:    out = exp(lhs); break;
            case GPU_OP_ABS_LHS:    out = abs(lhs); break;
            case GPU_OP_LOG_LHS:    out = log(lhs); break;

            // Commutative opcodes
            case GPU_OP_ADD_LHS_IMM: out = lhs + imm; break;
            case GPU_OP_ADD_LHS_RHS: out = lhs + rhs; break;
            case GPU_OP_MUL_LHS_IMM: out = lhs * imm; break;
            case GPU_OP_MUL_LHS_RHS: out = lhs * rhs; break;

#define CHOICE(f, a, b) {                                               \
    int c = 0;                                                          \
    out = f(a, b, c);                                                   \
    if (choice_index < CHOICE_ARRAY_SIZE * 16) {                        \
        choices[choice_index / 16] |= (c << ((choice_index % 16) * 2)); \
    }                                                                   \
    choice_index++;                                                     \
    has_any_choice |= (c != 0);                                         \
    break;                                                              \
}
            case GPU_OP_MIN_LHS_IMM: CHOICE(min, lhs, imm);
            case GPU_OP_MIN_LHS_RHS: CHOICE(min, lhs, rhs);
            case GPU_OP_MAX_LHS_IMM: CHOICE(max, lhs, imm);
            case GPU_OP_MAX_LHS_RHS: CHOICE(max, lhs, rhs);

            case GPU_OP_RAD_LHS_RHS: out = sqrt(square(lhs) + square(rhs)); break;

            // Non-commutative opcodes
            case GPU_OP_SUB_LHS_IMM: out = lhs - imm; break;
            case GPU_OP_SUB_IMM_RHS: out = imm - rhs; break;
            case GPU_OP_SUB_LHS_RHS: out = lhs - rhs; break;
            case GPU_OP_DIV_LHS_IMM: out = lhs / imm; break;
            case GPU_OP_DIV_IMM_RHS: out = imm / rhs; break;
            case GPU_OP_DIV_LHS_RHS: out = lhs / rhs; break;

            case GPU_OP_COPY_IMM: out = Interval(imm); break;
            case GPU_OP_COPY_LHS: out = lhs; break;
            case GPU_OP_COPY_RHS: out = rhs; break;

            default: assert(false);
        }
#undef lhs
#undef rhs
#undef imm
#undef out
    }

    // Check the result
    const uint8_t i_out = I_OUT(data);

    // Empty
    if (slots[i_out].lower() > 0.0f) {
        in_tiles[tile_index].position = -1;
        return;
    }

    // Masked
    if (DIMENSION == 3) {
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        if (image[pos.w] > pos.z) {
            in_tiles[tile_index].position = -1;
            return;
        }
    }

    // Filled
    if (slots[i_out].upper() < 0.0f) {
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        in_tiles[tile_index].position = -1;
        if (DIMENSION == 3) {
            atomicMax(&image[pos.w], pos.z);
        } else {
            image[pos.w] = 1;
        }
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

    // Check to make sure the tape isn't full
    // This doesn't mean that we'll successfully claim a chunk, because
    // other threads could claim chunks before us, but it's a way to check
    // quickly (and prevents tape_index from getting absurdly large).
    if (*tape_index >= NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE) {
        return;
    }

    // Claim a chunk of tape
    int32_t out_index = atomicAdd(tape_index, SUBTAPE_CHUNK_SIZE);
    int32_t out_offset = SUBTAPE_CHUNK_SIZE;

    // If we've run out of tape, then immediately return
    if (out_index + out_offset >= NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE) {
        return;
    }

    // Write out the end of the tape, which is the same as the ending
    // of the previous tape (0 opcode, with i_out as the last slot)
    out_offset--;
    tape_data[out_index + out_offset] = *data;

    while (1) {
        uint64_t d = *--data;
        if (!OP(&d)) {
            break;
        }
        const uint8_t op = OP(&d);
        if (op == GPU_OP_JUMP) {
            data += JUMP_TARGET(&d);
            continue;
        }

        const bool has_choice = op >= GPU_OP_MIN_LHS_IMM &&
                                op <= GPU_OP_MAX_LHS_RHS;
        choice_index -= has_choice;

        const uint8_t i_out = I_OUT(&d);
        if (!active[i_out]) {
            continue;
        }

        assert(!has_choice || choice_index >= 0);

        const int choice = (has_choice && choice_index < CHOICE_ARRAY_SIZE * 16)
            ? ((choices[choice_index / 16] >>
              ((choice_index % 16) * 2)) & 3)
            : 0;

        // If we're about to write a new piece of data to the tape,
        // (and are done with the current chunk), then we need to
        // add another link to the linked list.
        --out_offset;
        if (out_offset == 0) {
            const int32_t prev_index = out_index;

            // Early exit if we can't finish writing out this tape
            if (*tape_index >= NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE) {
                return;
            }
            out_index = atomicAdd(tape_index, SUBTAPE_CHUNK_SIZE);
            out_offset = SUBTAPE_CHUNK_SIZE;

            // Later exit if we claimed a chunk that exceeds the tape array
            if (out_index + out_offset >= NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE) {
                return;
            }
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
        if (choice == 0) {
            const uint8_t i_lhs = I_LHS(&d);
            if (i_lhs) {
                active[i_lhs] = true;
            }
            const uint8_t i_rhs = I_RHS(&d);
            if (i_rhs) {
                active[i_rhs] = true;
            }
        } else if (choice == 1 /* LHS */) {
            // The non-immediate is always the LHS in commutative ops, and
            // min/max (the only clauses that produce a choice) are commutative
            const uint8_t i_lhs = I_LHS(&d);
            active[i_lhs] = true;
            if (i_lhs == i_out) {
                ++out_offset;
                continue;
            } else {
                OP(&d) = GPU_OP_COPY_LHS;
            }
        } else if (choice == 2 /* RHS */) {
            const uint8_t i_rhs = I_RHS(&d);
            if (i_rhs) {
                active[i_rhs] = true;
                if (i_rhs == i_out) {
                    ++out_offset;
                    continue;
                } else {
                    OP(&d) = GPU_OP_COPY_RHS;
                }
            } else {
                OP(&d) = GPU_OP_COPY_IMM;
            }
        }
        tape_data[out_index + out_offset] = d;
    }

    // Write the beginning of the tape
    out_offset--;
    tape_data[out_index + out_offset] = *data;

    // Record the beginning of the tape in the output tile
    in_tiles[tile_index].tape = out_index + out_offset;
}

////////////////////////////////////////////////////////////////////////////////

/*
 *  mask_filled_tiles
 *
 *  For every tile in the `in_tiles` array, compares its z position against
 *  the image's z value at the tile's xy position.  If the tile is below the
 *  image, then it will never contribute, so its position is set to -1 to mark
 *  it as inactive.
 */
__global__
void mask_filled_tiles(int32_t* const __restrict__ image,
                       const uint32_t tiles_per_side,

                       TileNode* const __restrict__ in_tiles,
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

/*
 *  assign_next_nodes
 *
 *  For every tile in `in_tiles`, which is active (i.e. has a position that
 *  has not been set to -1), set its `next` value to a unique value.
 *
 *  The total number of active tiles is stored in `num_active_tiles`; `next`
 *  values range from 0 to `num_active_tiles - 1`.
 *
 *  Philosophically, this function packs sparse items (active tiles in
 *  `in_tiles`) tightly.  It could also be implemented as a scan, but this is
 *  far from the limiting factor.
 */
__global__
void assign_next_nodes(TileNode* const __restrict__ in_tiles,
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

/*
 *  subdivide_active_tiles
 *
 *  For each active tile in `in_tiles`, unpack it into 64 subtiles in
 *  `out_tiles`.  Subtiles are tightly packed using `next` indices assigned
 *  in `assign_next_nodes`.
 *
 *  Subtiles inherit the `tape` value from their parent tiles, since they're
 *  contained within the parent and can reuse its tape.  They are assigned
 *  `next` = -1, because we don't yet know whether they have children.
 */
__global__
void subdivide_active_tiles_3d(
        const TileNode* const __restrict__ in_tiles,
        const int32_t in_tile_count,
        const int32_t tiles_per_side,
        TileNode* const __restrict__ out_tiles)
{
    const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t subtile_index = index % 64;
    const int32_t tile_index = index / 64;
    if (tile_index >= in_tile_count || in_tiles[tile_index].next == -1) {
        return;
    }

    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    const int32_t subtiles_per_side = tiles_per_side * 4;

    const int4 sub = unpack(subtile_index, 4);
    const int32_t sx = pos.x * 4 + sub.x;
    const int32_t sy = pos.y * 4 + sub.y;
    const int32_t sz = pos.z * 4 + sub.z;
    const int32_t next_tile =
        sx +
        sy * subtiles_per_side +
        sz * subtiles_per_side * subtiles_per_side;

    const int t = in_tiles[tile_index].next * 64 + subtile_index;
    out_tiles[t].position = next_tile;
    out_tiles[t].tape = in_tiles[tile_index].tape;
    out_tiles[t].next = -1;
}

__global__
void subdivide_active_tiles_2d(
        const TileNode* const __restrict__ in_tiles,
        const int32_t in_tile_count,
        const int32_t tiles_per_side,
        TileNode* const __restrict__ out_tiles)
{
    const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t subtile_index = index % 64;
    const int32_t tile_index = index / 64;
    if (tile_index >= in_tile_count || in_tiles[tile_index].next == -1) {
        return;
    }

    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    assert(pos.z == 0);
    const int32_t subtiles_per_side = tiles_per_side * 8;

    const int4 sub = unpack(subtile_index, 8);
    const int32_t sx = pos.x * 8 + sub.x;
    const int32_t sy = pos.y * 8 + sub.y;
    const int32_t next_tile = sx + sy * subtiles_per_side;

    const int t = in_tiles[tile_index].next * 64 + subtile_index;
    out_tiles[t].position = next_tile;
    out_tiles[t].tape = in_tiles[tile_index].tape;
    out_tiles[t].next = -1;
}

/*
 *  copy_active_tiles
 *
 *  For each active tile in `in_tiles`, copy it into `out_tiles`.  This
 *  operation turns a sparse array of active tiles into a tightly packed array.
 *
 *  This is used right before per-pixel evaluation, which wants a compact list
 *  of active tiles, but doesn't want them to be subdivided by 64.
 *
 *  Tiles keep the `tape` value when copied, but `next` is assigned to -1
 *  (since we're at the bottom of the evaluation stack).
 */
__global__
void copy_active_tiles(TileNode* const __restrict__ in_tiles,
                       const int32_t in_tile_count,
                       TileNode* const __restrict__ out_tiles)
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

/*
 *  copy_filled
 *
 *  Copies a lower-resolution (4x undersampled) image into a higher-resolution
 *  image, expanding every active (non-zero) "pixel" by 4x.
 *
 *  The higher-resolution image must be empty (all 0) when this is called;
 *  no comparison of Z values is done.
 */
__global__
void copy_filled_3d(const int32_t* __restrict__ prev,
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
__global__
void copy_filled_2d(const int32_t* __restrict__ prev,
                    int32_t* __restrict__ image,
                    const int32_t image_size_px)
{
    const int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < image_size_px && y < image_size_px &&
        prev[x / 8 + y / 8 * (image_size_px / 8)])
    {
        image[x + y * image_size_px] = 1;
    }
}

////////////////////////////////////////////////////////////////////////////////

/*
 *  calculate_voxels
 *
 *  For a given set of input tiles, each is divided into 64 voxels.  Each
 *  voxel's position in (orthographic, screen-aligned, +/-1) render space is
 *  transformed by `mat`, then written to the `values` array.
 *
 *  For efficiency, we actually calculate two voxels per thread and store them
 *  in a float2, i.e. data is packed as
 *  [x0 x1 | y0 y1 | z0 z1 | x2 x3 | y2 y3 | z2 z3 | ...]
 */
__global__
void calculate_voxels(const TileNode* const __restrict__ in_tiles,
                      const uint32_t in_tile_count,
                      const uint32_t tiles_per_side,
                      const Eigen::Matrix4f mat,
                      float2* const __restrict__ values)
{
    // Each tile is executed by 32 threads (one for each pair of voxels).
    //
    // This is different from the eval_tiles_i function, which evaluates one
    // tile per thread, because the tiles are already expanded by 64x by the
    // time they're stored in the in_tiles list.
    const int32_t voxel_index = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t tile_index = voxel_index / 32;

    if (tile_index >= in_tile_count) {
        return;
    }
    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    const int4 sub = unpack(threadIdx.x % 32, 4);

    const int32_t px = pos.x * 4 + sub.x;
    const int32_t py = pos.y * 4 + sub.y;
    const int32_t pz_a = pos.z * 4 + sub.z;

    const float size_recip = 1.0f / (tiles_per_side * 4);

    const float fx = ((px + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fy = ((py + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fz_a = ((pz_a + 0.5f) * size_recip - 0.5f) * 2.0f;

    // Otherwise, calculate the X/Y/Z values
    const float fw_a = mat(3, 0) * fx +
                       mat(3, 1) * fy +
                       mat(3, 2) * fz_a + mat(3, 3);
    for (unsigned i=0; i < 3; ++i) {
        values[voxel_index * 3 + i].x =
            (mat(i, 0) * fx +
             mat(i, 1) * fy +
             mat(i, 2) * fz_a + mat(i, 3)) / fw_a;
    }

    // Do the same calculation for the second pixel
    const int32_t pz_b = pz_a + 2;
    const float fz_b = ((pz_b + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fw_b = mat(3, 0) * fx +
                       mat(3, 1) * fy +
                       mat(3, 2) * fz_b + mat(3, 3);

    for (unsigned i=0; i < 3; ++i) {
        values[voxel_index * 3 + i].y =
            (mat(i, 0) * fx +
             mat(i, 1) * fy +
             mat(i, 2) * fz_b + mat(i, 3)) / fw_b;
    }
}

__global__
void calculate_pixels(const TileNode* const __restrict__ in_tiles,
                      const uint32_t in_tile_count,
                      const uint32_t tiles_per_side,
                      const Eigen::Matrix3f mat, const float z,
                      float2* const __restrict__ values)
{
    // Each tile is executed by 32 threads (one for each pair of voxels).
    //
    // This is different from the eval_tiles_i function, which evaluates one
    // tile per thread, because the tiles are already expanded by 64x by the
    // time they're stored in the in_tiles list.
    const int32_t voxel_index = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t tile_index = voxel_index / 32;

    if (tile_index >= in_tile_count) {
        return;
    }
    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    const int4 sub = unpack(threadIdx.x % 32, 8);

    const int32_t px = pos.x * 8 + sub.x;
    const int32_t py_a = pos.y * 8 + sub.y;
    assert(sub.y < 4);
    assert(sub.z == 0);

    const float size_recip = 1.0f / (tiles_per_side * 8);

    const float fx = ((px + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fy_a = ((py_a + 0.5f) * size_recip - 0.5f) * 2.0f;

    // Otherwise, calculate the X/Y/Z values
    const float fw_a = mat(2, 0) * fx + mat(2, 1) * fy_a + mat(2, 2);
    for (unsigned i=0; i < 2; ++i) {
        values[voxel_index * 3 + i].x =
            (mat(i, 0) * fx + mat(i, 1) * fy_a + mat(i, 2)) / fw_a;
    }

    // Do the same calculation for the second pixel
    const int32_t py_b = py_a + 4;
    const float fy_b = ((py_b + 0.5f) * size_recip - 0.5f) * 2.0f;
    const float fw_b = mat(2, 0) * fx + mat(2, 1) * fy_b + mat(2, 2);

    for (unsigned i=0; i < 2; ++i) {
        values[voxel_index * 3 + i].y =
            (mat(i, 0) * fx + mat(i, 1) * fy_b + mat(i, 2)) / fw_b;
    }

    values[voxel_index * 3 + 2] = make_float2(z, z);
}

/*
 *  eval_voxels_f
 *
 *  Evaluates the 64 voxels which make up every tile in `in_tiles` (of which
 *  there should be `in_tile_count`.  This must be called after
 *  `calculate_voxels`, which writes voxel positions to the `values` array.
 *
 *  For efficiency, this function calculates two voxels per thread, reading and
 *  writing float2 data (which improves memory access patterns).
 *
 *  Filled voxels are written to `image`, using atomic operations to accumulate
 *  the voxel with the tallest Z value.
 */
template <unsigned DIMENSION>
__global__
void eval_voxels_f(const uint64_t* const __restrict__ tape_data,
                   int32_t* const __restrict__ image,
                   const uint32_t tiles_per_side,

                   TileNode* const __restrict__ in_tiles,
                   const int32_t in_tile_count,

                   const float2* const __restrict__ values)
{
    // Each tile is executed by 32 threads (one for each pair of voxels, so
    // we can do all of our load/stores as float2s and make memory happier).
    //
    // This is different from the eval_tiles_i function, which evaluates one
    // tile per thread, because the tiles are already expanded by 64x by the
    // time they're stored in the in_tiles list.
    const int32_t voxel_index = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t tile_index = voxel_index / 32;
    if (tile_index >= in_tile_count) {
        return;
    }

    // Check whether this pixel is masked in the output image
    if (DIMENSION == 3) {
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        const int4 sub = unpack(threadIdx.x % 32, 4);

        const int32_t px = pos.x * 4 + sub.x;
        const int32_t py = pos.y * 4 + sub.y;
        const int32_t pz = pos.z * 4 + sub.z;

        // Early return if this pixel won't ever be filled
        if (image[px + py * tiles_per_side * 4] >= pz + 2) {
            return;
        }
    }

    float2 slots[128];

    // Pick out the tape based on the pointer stored in the tiles list
    const uint64_t* __restrict__ data = &tape_data[in_tiles[tile_index].tape];
    slots[((const uint8_t*)tape_data)[1]] = values[voxel_index * 3];
    slots[((const uint8_t*)tape_data)[2]] = values[voxel_index * 3 + 1];
    slots[((const uint8_t*)tape_data)[3]] = values[voxel_index * 3 + 2];

    while (1) {
        const uint64_t d = *++data;
        if (!OP(&d)) {
            break;
        }
        switch (OP(&d)) {
            case GPU_OP_JUMP: data += JUMP_TARGET(&d); continue;

#define lhs slots[I_LHS(&d)]
#define rhs slots[I_RHS(&d)]
#define imm IMM(&d)
#define out slots[I_OUT(&d)]

            case GPU_OP_SQUARE_LHS: out = make_float2(lhs.x * lhs.x, lhs.y * lhs.y); break;
            case GPU_OP_SQRT_LHS: out = make_float2(sqrtf(lhs.x), sqrtf(lhs.y)); break;
            case GPU_OP_NEG_LHS: out = make_float2(-lhs.x, -lhs.y); break;
            case GPU_OP_SIN_LHS: out = make_float2(sinf(lhs.x), sinf(lhs.y)); break;
            case GPU_OP_COS_LHS: out = make_float2(cosf(lhs.x), cosf(lhs.y)); break;
            case GPU_OP_ASIN_LHS: out = make_float2(asinf(lhs.x), asinf(lhs.y)); break;
            case GPU_OP_ACOS_LHS: out = make_float2(acosf(lhs.x), acosf(lhs.y)); break;
            case GPU_OP_ATAN_LHS: out = make_float2(atanf(lhs.x), atanf(lhs.y)); break;
            case GPU_OP_EXP_LHS: out = make_float2(expf(lhs.x), expf(lhs.y)); break;
            case GPU_OP_ABS_LHS: out = make_float2(fabsf(lhs.x), fabsf(lhs.y)); break;
            case GPU_OP_LOG_LHS: out = make_float2(logf(lhs.x), logf(lhs.y)); break;

            // Commutative opcodes
            case GPU_OP_ADD_LHS_IMM: out = make_float2(lhs.x + imm, lhs.y + imm); break;
            case GPU_OP_ADD_LHS_RHS: out = make_float2(lhs.x + rhs.x, lhs.y + rhs.y); break;
            case GPU_OP_MUL_LHS_IMM: out = make_float2(lhs.x * imm, lhs.y * imm); break;
            case GPU_OP_MUL_LHS_RHS: out = make_float2(lhs.x * rhs.x, lhs.y * rhs.y); break;
            case GPU_OP_MIN_LHS_IMM: out = make_float2(fminf(lhs.x, imm), fminf(lhs.y, imm)); break;
            case GPU_OP_MIN_LHS_RHS: out = make_float2(fminf(lhs.x, rhs.x), fminf(lhs.y, rhs.y)); break;
            case GPU_OP_MAX_LHS_IMM: out = make_float2(fmaxf(lhs.x, imm), fmaxf(lhs.y, imm)); break;
            case GPU_OP_MAX_LHS_RHS: out = make_float2(fmaxf(lhs.x, rhs.x), fmaxf(lhs.y, rhs.y)); break;
            case GPU_OP_RAD_LHS_RHS: out = make_float2(sqrtf(lhs.x * lhs.x + rhs.x * rhs.x),
                                                       sqrtf(lhs.y * lhs.y + rhs.y * rhs.y)); break;

            // Non-commutative opcodes
            case GPU_OP_SUB_LHS_IMM: out = make_float2(lhs.x - imm, lhs.y - imm); break;
            case GPU_OP_SUB_IMM_RHS: out = make_float2(imm - rhs.x, imm - rhs.y); break;
            case GPU_OP_SUB_LHS_RHS: out = make_float2(lhs.x - rhs.x, lhs.y - rhs.y); break;

            case GPU_OP_DIV_LHS_IMM: out = make_float2(lhs.x / imm, lhs.y / imm); break;
            case GPU_OP_DIV_IMM_RHS: out = make_float2(imm / rhs.x, imm / rhs.y); break;
            case GPU_OP_DIV_LHS_RHS: out = make_float2(lhs.x / rhs.x, lhs.y / rhs.y); break;

            case GPU_OP_COPY_IMM: out = make_float2(imm, imm); break;
            case GPU_OP_COPY_LHS: out = make_float2(lhs.x, lhs.y); break;
            case GPU_OP_COPY_RHS: out = make_float2(rhs.x, rhs.y); break;

#undef lhs
#undef rhs
#undef imm
#undef out
        }
    }

    // Check the result
    const uint8_t i_out = I_OUT(data);

    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    if (DIMENSION == 3) {
        const int4 sub = unpack(threadIdx.x % 32, 4);
        // The second voxel is always higher in Z, so it masks the lower voxel
        if (slots[i_out].y < 0.0f) {
            const int32_t px = pos.x * 4 + sub.x;
            const int32_t py = pos.y * 4 + sub.y;
            const int32_t pz = pos.z * 4 + sub.z + 2;

            atomicMax(&image[px + py * tiles_per_side * 4], pz);
        } else if (slots[i_out].x < 0.0f) {
            const int32_t px = pos.x * 4 + sub.x;
            const int32_t py = pos.y * 4 + sub.y;
            const int32_t pz = pos.z * 4 + sub.z;

            atomicMax(&image[px + py * tiles_per_side * 4], pz);
        }
    } else if (DIMENSION == 2) {
        const int4 sub = unpack(threadIdx.x % 32, 8);
        if (slots[i_out].y < 0.0f) {
            const int32_t px = pos.x * 8 + sub.x;
            const int32_t py = pos.y * 8 + sub.y + 4;

            image[px + py * tiles_per_side * 8] = 1;
        }
        if (slots[i_out].x < 0.0f) {
            const int32_t px = pos.x * 8 + sub.x;
            const int32_t py = pos.y * 8 + sub.y;

            image[px + py * tiles_per_side * 8] = 1;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/*
 *  eval_pixels_d
 *
 *  For each active pixel in `image`, renders its partial derivatives
 *  (using automatic differentiation), interpreting the result as its normal
 *  and saving it to the `output` image.
 *
 *  We search through the `tiles`, `subtiles`, `microtiles` structure to
 *  find the shortest tape useful for each pixel, as an optimization.
 */
__global__
void eval_pixels_d(const uint64_t* const __restrict__ tape_data,
                   const int32_t* const __restrict__ image,
                   uint32_t* const __restrict__ output,
                   const uint32_t image_size_px,

                   Eigen::Matrix4f mat,

                   const TileNode* const __restrict__ tiles,
                   const TileNode* const __restrict__ subtiles,
                   const TileNode* const __restrict__ microtiles)
{
    const int32_t px = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t py = threadIdx.y + blockIdx.y * blockDim.y;
    if (px >= image_size_px || py >= image_size_px) {
        return;
    }

    const int32_t pxy = px + py * image_size_px;
    int32_t pz = image[pxy];
    if (pz == 0) {
        return;
    }
    // Move slightly in front of the surface, unless we're at the top of the
    // region (in which case moving would put us in an invalid tile)
    if (pz < image_size_px - 1) {
        pz += 1;
    }

    Deriv slots[128];

    {   // Calculate size and load into initial slots
        const float size_recip = 1.0f / image_size_px;

        const float fx = ((px + 0.5f) * size_recip - 0.5f) * 2.0f;
        const float fy = ((py + 0.5f) * size_recip - 0.5f) * 2.0f;
        const float fz = ((pz + 0.5f) * size_recip - 0.5f) * 2.0f;

        // Otherwise, calculate the X/Y/Z values
        const float fw_ = mat(3, 0) * fx +
                          mat(3, 1) * fy +
                          mat(3, 2) * fz + mat(3, 3);
        for (unsigned i=0; i < 3; ++i) {
            slots[((const uint8_t*)tape_data)[i + 1]] = Deriv(
                (mat(i, 0) * fx +
                 mat(i, 1) * fy +
                 mat(i, 2) * fz + mat(i, 3)) / fw_);
        }
        slots[((const uint8_t*)tape_data)[1]].v.x = 1.0f;
        slots[((const uint8_t*)tape_data)[2]].v.y = 1.0f;
        slots[((const uint8_t*)tape_data)[3]].v.z = 1.0f;
    }


    const uint64_t* __restrict__ data = tape_data;

    {   // Pick out the tape based on the pointer stored in the tiles list
        const int32_t tile_x = px / 64;
        const int32_t tile_y = py / 64;
        const int32_t tile_z = pz / 64;
        const int32_t tile = tile_x +
                             tile_y * (image_size_px / 64) +
                             tile_z * (image_size_px / 64) * (image_size_px / 64);

        if (tiles[tile].next == -1) {
            data = &tape_data[tiles[tile].tape];
        } else {
            const int32_t sx = (px % 64) / 16;
            const int32_t sy = (py % 64) / 16;
            const int32_t sz = (pz % 64) / 16;
            const int32_t subtile = tiles[tile].next * 64 +
                                    sx +
                                    sy * 4 +
                                    sz * 16;

            if (subtiles[subtile].next == -1) {
                data = &tape_data[subtiles[subtile].tape];
            } else {
                const int32_t ux = (px % 16) / 4;
                const int32_t uy = (py % 16) / 4;
                const int32_t uz = (pz % 16) / 4;
                const int32_t microtile = subtiles[subtile].next * 64 +
                                        ux +
                                        uy * 4 +
                                        uz * 16;
                data = &tape_data[microtiles[microtile].tape];
            }
        }
    }

    while (1) {
        const uint64_t d = *++data;
        if (!OP(&d)) {
            break;
        }
        switch (OP(&d)) {
            case GPU_OP_JUMP: data += JUMP_TARGET(&d); continue;

#define lhs slots[I_LHS(&d)]
#define rhs slots[I_RHS(&d)]
#define imm IMM(&d)
#define out slots[I_OUT(&d)]

            case GPU_OP_SQUARE_LHS: out = lhs * lhs; break;
            case GPU_OP_SQRT_LHS: out = sqrt(lhs); break;
            case GPU_OP_NEG_LHS: out = -lhs; break;
            case GPU_OP_SIN_LHS: out = sin(lhs); break;
            case GPU_OP_COS_LHS: out = cos(lhs); break;
            case GPU_OP_ASIN_LHS: out = asin(lhs); break;
            case GPU_OP_ACOS_LHS: out = acos(lhs); break;
            case GPU_OP_ATAN_LHS: out = atan(lhs); break;
            case GPU_OP_EXP_LHS: out = exp(lhs); break;
            case GPU_OP_ABS_LHS: out = abs(lhs); break;
            case GPU_OP_LOG_LHS: out = log(lhs); break;

            // Commutative opcodes
            case GPU_OP_ADD_LHS_IMM: out = lhs + imm; break;
            case GPU_OP_ADD_LHS_RHS: out = lhs + rhs; break;
            case GPU_OP_MUL_LHS_IMM: out = lhs * imm; break;
            case GPU_OP_MUL_LHS_RHS: out = lhs * rhs; break;
            case GPU_OP_MIN_LHS_IMM: out = min(lhs, imm); break;
            case GPU_OP_MIN_LHS_RHS: out = min(lhs, rhs); break;
            case GPU_OP_MAX_LHS_IMM: out = max(lhs, imm); break;
            case GPU_OP_MAX_LHS_RHS: out = max(lhs, rhs); break;
            case GPU_OP_RAD_LHS_RHS: out = sqrt(square(lhs) + square(rhs)); break;


            // Non-commutative opcodes
            case GPU_OP_SUB_LHS_IMM: out = lhs - imm; break;
            case GPU_OP_SUB_IMM_RHS: out = imm - rhs; break;
            case GPU_OP_SUB_LHS_RHS: out = lhs - rhs; break;

            case GPU_OP_DIV_LHS_IMM: out = lhs / imm; break;
            case GPU_OP_DIV_IMM_RHS: out = imm / rhs; break;
            case GPU_OP_DIV_LHS_RHS: out = lhs / rhs; break;

            case GPU_OP_COPY_IMM: out = Deriv(imm); break;
            case GPU_OP_COPY_LHS: out = lhs; break;
            case GPU_OP_COPY_RHS: out = rhs; break;

#undef lhs
#undef rhs
#undef imm
#undef out
        }
    }

    const uint8_t i_out = I_OUT(data);
    const Deriv result = slots[i_out];
    float norm = sqrtf(powf(result.dx(), 2) +
                       powf(result.dy(), 2) +
                       powf(result.dz(), 2));
    uint8_t dx = (result.dx() / norm) * 127 + 128;
    uint8_t dy = (result.dy() / norm) * 127 + 128;
    uint8_t dz = (result.dz() / norm) * 127 + 128;
    output[pxy] = (0xFF << 24) | (dz << 16) | (dy << 8) | dx;
}

////////////////////////////////////////////////////////////////////////////////

void Context::render2D(const Tape& tape, const Eigen::Matrix3f& mat, const float z) {
    // Reset the tape index and copy the tape to the beginning of the
    // context's tape buffer area.
    *tape_index = tape.length;
    cudaMemcpyAsync(tape_data.get(), tape.data.get(),
                    sizeof(uint64_t) * tape.length,
                    cudaMemcpyDeviceToDevice);

    // Reset all of the data arrays.  In 2D, we only use stages 0, 2, and 3
    // for 64^2, 8^2, and per-voxel evaluation steps.
    CUDA_CHECK(cudaMemsetAsync(stages[0].filled.get(), 0, sizeof(int32_t) *
                               pow(image_size_px / 64, 2)));
    CUDA_CHECK(cudaMemsetAsync(stages[2].filled.get(), 0, sizeof(int32_t) *
                               pow(image_size_px / 8, 2)));
    CUDA_CHECK(cudaMemsetAsync(stages[3].filled.get(), 0, sizeof(int32_t) *
                               pow(image_size_px, 2)));

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 64x64 tiles
    ////////////////////////////////////////////////////////////////////////////

    // Go the whole list of first-stage tiles, assigning each to
    // be [position, tape = 0, next = -1]
    unsigned count = pow(image_size_px / 64, 2);
    unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;
    preload_tiles<<<num_blocks, NUM_THREADS>>>(stages[0].tiles.get(), count);

    // Iterate over 64^2, 8^2 tiles
    for (unsigned i=0; i < 3; i += 2) {
        const unsigned tile_size_px = i ? 8 : 64;
        const unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;

        if (values_size < num_blocks * NUM_THREADS * 3) {
            values.reset(CUDA_MALLOC(Interval, num_blocks * NUM_THREADS * 3));
            values_size = num_blocks * NUM_THREADS * 3;
        }

        // Unpack position values into interval X/Y/Z in the values array
        // This is done in a separate kernel to avoid bloating the
        // eval_tiles_i kernel with more registers, which is detrimental
        // to occupancy.
        calculate_intervals_2d<<<num_blocks, NUM_THREADS>>>(
            stages[i].tiles.get(),
            count,
            image_size_px / tile_size_px,
            mat, z,
            reinterpret_cast<Interval*>(values.get()));

        // Do the actual tape evaluation, which is the expensive step
        eval_tiles_i<2><<<num_blocks, NUM_THREADS>>>(
            tape_data.get(),
            tape_index.get(),
            stages[i].filled.get(),
            image_size_px / tile_size_px,

            stages[i].tiles.get(),
            count,

            reinterpret_cast<Interval*>(values.get()));

        // Mark the total number of active tiles (from this stage) to 0
        cudaMemsetAsync(num_active_tiles.get(), 0, sizeof(int32_t));

        // Count up active tiles, to figure out how much memory needs to be
        // allocated in the next stage.
        assign_next_nodes<<<num_blocks, NUM_THREADS>>>(
            stages[i].tiles.get(),
            count,
            num_active_tiles.get());

        // Count the number of active tiles, which have been accumulated
        // through repeated calls to assign_next_nodes
        int32_t active_tile_count;
        cudaMemcpy(&active_tile_count, num_active_tiles.get(), sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        if (i == 0) {
            active_tile_count *= 64;
        }

        // Make sure that the subtiles buffer has enough room
        // This wastes a small amount of data for the per-pixel evaluation,
        // where the `next` indexes aren't used, but it's relatively small.
        const int next = i ? 3 : 2;
        if (active_tile_count > stages[next].tile_array_size) {
            stages[next].tile_array_size = active_tile_count;
            stages[next].tiles.reset(CUDA_MALLOC(TileNode, active_tile_count));
        }

        if (i < 2) {
            // Build the new tile list from active tiles in the previous list
            subdivide_active_tiles_2d<<<num_blocks*64, NUM_THREADS>>>(
                stages[i].tiles.get(),
                count,
                image_size_px / tile_size_px,
                stages[next].tiles.get());
        } else {
            // Special case for per-pixel evaluation, which
            // doesn't unpack every single pixel (since that would take up
            // 64x extra space).
            copy_active_tiles<<<num_blocks, NUM_THREADS>>>(
                stages[i].tiles.get(),
                count,
                stages[next].tiles.get());
        }

        {   // Copy filled tiles into the next level's image (expanding them
            // by 64x).  This is cleaner that accumulating all of the levels
            // in a single pass, and could (possibly?) help with skipping
            // fully occluded tiles.
            const unsigned next_tile_size = tile_size_px / 8;
            const uint32_t u = ((image_size_px / next_tile_size) / 32);
            copy_filled_2d<<<dim3(u + 1, u + 1), dim3(32, 32)>>>(
                    stages[i].filled.get(),
                    stages[next].filled.get(),
                    image_size_px / next_tile_size);
        }

        // Assign the next number of tiles to evaluate
        count = active_tile_count;
    }

    // Time to render individual pixels!
    num_blocks = (count + NUM_TILES - 1) / NUM_TILES;
    const size_t num_values = num_blocks * NUM_TILES * 32 * 3;
    if (values_size < num_values) {
        values.reset(CUDA_MALLOC(float2, num_values));
        values_size = num_values;
    }
    calculate_pixels<<<num_blocks, NUM_TILES * 32>>>(
        stages[3].tiles.get(),
        count,
        image_size_px / 8,
        mat, z,
        reinterpret_cast<float2*>(values.get()));
    eval_voxels_f<2><<<num_blocks, NUM_TILES * 32>>>(
        tape_data.get(),
        stages[3].filled.get(),
        image_size_px / 8,

        stages[3].tiles.get(),
        count,

        reinterpret_cast<float2*>(values.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Context::render3D(const Tape& tape, const Eigen::Matrix4f& mat) {
    // Reset the tape index and copy the tape to the beginning of the
    // context's tape buffer area.
    *tape_index = tape.length;
    cudaMemcpyAsync(tape_data.get(), tape.data.get(),
                    sizeof(uint64_t) * tape.length,
                    cudaMemcpyDeviceToDevice);

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 64x64x64 tiles
    ////////////////////////////////////////////////////////////////////////////

    // Reset all of the data arrays
    for (unsigned i=0; i < 4; ++i) {
        const unsigned tile_size_px = 64 / (1 << (i * 2));
        CUDA_CHECK(cudaMemsetAsync(stages[i].filled.get(), 0, sizeof(int32_t) *
                                   pow(image_size_px / tile_size_px, 2)));
    }
    CUDA_CHECK(cudaMemsetAsync(normals.get(), 0, sizeof(uint32_t) *
                               pow(image_size_px, 2)));

    // Go the whole list of first-stage tiles, assigning each to
    // be [position, tape = 0, next = -1]
    unsigned count = pow(image_size_px / 64, 3);
    unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;
    preload_tiles<<<num_blocks, NUM_THREADS>>>(stages[0].tiles.get(), count);

    // Iterate over 64^3, 16^3, 4^3 tiles
    for (unsigned i=0; i < 3; ++i) {
        //printf("BEGINNING STAGE %u\n", i);
        const unsigned tile_size_px = 64 / (1 << (i * 2));
        const unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;

        if (values_size < num_blocks * NUM_THREADS * 3) {
            values.reset(CUDA_MALLOC(Interval, num_blocks * NUM_THREADS * 3));
            values_size = num_blocks * NUM_THREADS * 3;
        }

        // Unpack position values into interval X/Y/Z in the values array
        // This is done in a separate kernel to avoid bloating the
        // eval_tiles_i kernel with more registers, which is detrimental
        // to occupancy.
        calculate_intervals_3d<<<num_blocks, NUM_THREADS>>>(
            stages[i].tiles.get(),
            count,
            image_size_px / tile_size_px,
            mat,
            reinterpret_cast<Interval*>(values.get()));

        // Mark every tile which is covered in the image as masked,
        // which means it will be skipped later on.  We do this again below,
        // but it's basically free, so we should do it here and simplify
        // the logic in eval_tiles_i.
        mask_filled_tiles<<<num_blocks, NUM_THREADS>>>(
            stages[i].filled.get(),
            image_size_px / tile_size_px,
            stages[i].tiles.get(),
            count);

        // Do the actual tape evaluation, which is the expensive step
        eval_tiles_i<3><<<num_blocks, NUM_THREADS>>>(
            tape_data.get(),
            tape_index.get(),
            stages[i].filled.get(),
            image_size_px / tile_size_px,

            stages[i].tiles.get(),
            count,

            reinterpret_cast<Interval*>(values.get()));

        // Mark the total number of active tiles (from this stage) to 0
        cudaMemsetAsync(num_active_tiles.get(), 0, sizeof(int32_t));

        // Now that we have evaluated every tile at this level, we do one more
        // round of occlusion culling before accumulating tiles to render at
        // the next phase.
        mask_filled_tiles<<<num_blocks, NUM_THREADS>>>(
            stages[i].filled.get(),
            image_size_px / tile_size_px,
            stages[i].tiles.get(),
            count);

        // Count up active tiles, to figure out how much memory needs to be
        // allocated in the next stage.
        assign_next_nodes<<<num_blocks, NUM_THREADS>>>(
            stages[i].tiles.get(),
            count,
            num_active_tiles.get());

        // Count the number of active tiles, which have been accumulated
        // through repeated calls to assign_next_nodes
        int32_t active_tile_count;
        cudaMemcpy(&active_tile_count, num_active_tiles.get(), sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        if (i < 2) {
            active_tile_count *= 64;
        }

        // Make sure that the subtiles buffer has enough room
        // This wastes a small amount of data for the per-pixel evaluation,
        // where the `next` indexes aren't used, but it's relatively small.
        if (active_tile_count > stages[i + 1].tile_array_size) {
            stages[i + 1].tile_array_size = active_tile_count;
            stages[i + 1].tiles.reset(CUDA_MALLOC(TileNode, active_tile_count));
        }

        if (i < 2) {
            // Build the new tile list from active tiles in the previous list
            subdivide_active_tiles_3d<<<num_blocks*64, NUM_THREADS>>>(
                stages[i].tiles.get(),
                count,
                image_size_px / tile_size_px,
                stages[i + 1].tiles.get());
        } else {
            // Special case for per-pixel evaluation, which
            // doesn't unpack every single pixel (since that would take up
            // 64x extra space).
            copy_active_tiles<<<num_blocks, NUM_THREADS>>>(
                stages[i].tiles.get(),
                count,
                stages[i + 1].tiles.get());
        }

        {   // Copy filled tiles into the next level's image (expanding them
            // by 64x).  This is cleaner that accumulating all of the levels
            // in a single pass, and could (possibly?) help with skipping
            // fully occluded tiles.
            const unsigned next_tile_size = tile_size_px / 4;
            const uint32_t u = ((image_size_px / next_tile_size) / 32);
            copy_filled_3d<<<dim3(u + 1, u + 1), dim3(32, 32)>>>(
                    stages[i].filled.get(),
                    stages[i + 1].filled.get(),
                    image_size_px / next_tile_size);
        }

        // Assign the next number of tiles to evaluate
        count = active_tile_count;
    }

    // Time to render individual pixels!
    num_blocks = (count + NUM_TILES - 1) / NUM_TILES;
    const size_t num_values = num_blocks * NUM_TILES * 32 * 3;
    if (values_size < num_values) {
        values.reset(CUDA_MALLOC(float2, num_values));
        values_size = num_values;
    }
    calculate_voxels<<<num_blocks, NUM_TILES * 32>>>(
        stages[3].tiles.get(),
        count,
        image_size_px / 4,
        mat,
        reinterpret_cast<float2*>(values.get()));
    eval_voxels_f<3><<<num_blocks, NUM_TILES * 32>>>(
        tape_data.get(),
        stages[3].filled.get(),
        image_size_px / 4,

        stages[3].tiles.get(),
        count,

        reinterpret_cast<float2*>(values.get()));

    {   // Then render normals into those pixels
        const uint32_t u = ((image_size_px + 15) / 16);
        eval_pixels_d<<<dim3(u, u), dim3(16, 16)>>>(
                tape_data.get(),
                stages[3].filled.get(),
                normals.get(),
                image_size_px,
                mat,
                stages[0].tiles.get(),
                stages[1].tiles.get(),
                stages[2].tiles.get());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}


void Context::render2D_brute(const Tape& tape,
                             const Eigen::Matrix3f& mat,
                             const float z)
{
    // Reset the tape index and copy the tape to the beginning of the
    // context's tape buffer area.
    *tape_index = tape.length;
    cudaMemcpyAsync(tape_data.get(), tape.data.get(),
                    sizeof(uint64_t) * tape.length,
                    cudaMemcpyDeviceToDevice);

    // Reset the final image array, since we'll be rendering directly to it
    CUDA_CHECK(cudaMemsetAsync(stages[3].filled.get(), 0, sizeof(int32_t) *
                               pow(image_size_px, 2)));

    // We'll only be evaluating 8x8 tiles, so preload all of them
    unsigned count = pow(image_size_px / 8, 2);
    if (count > stages[3].tile_array_size) {
        stages[3].tile_array_size = count;
        stages[3].tiles.reset(CUDA_MALLOC(TileNode, count));
    }
    unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;
    preload_tiles<<<num_blocks, NUM_THREADS>>>(stages[3].tiles.get(), count);

    // Time to render individual pixels!
    num_blocks = (count + NUM_TILES - 1) / NUM_TILES;
    const size_t num_values = num_blocks * NUM_TILES * 32 * 3;
    if (values_size < num_values) {
        values.reset(CUDA_MALLOC(float2, num_values));
        values_size = num_values;
    }
    calculate_pixels<<<num_blocks, NUM_TILES * 32>>>(
        stages[3].tiles.get(),
        count,
        image_size_px / 8,
        mat, z,
        reinterpret_cast<float2*>(values.get()));
    eval_voxels_f<2><<<num_blocks, NUM_TILES * 32>>>(
        tape_data.get(),
        stages[3].filled.get(),
        image_size_px / 8,

        stages[3].tiles.get(),
        count,

        reinterpret_cast<float2*>(values.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////


template <int DIMENSION>
__global__
void eval_tiles_i_heatmap(uint64_t* const __restrict__ tape_data,
                          int32_t* const __restrict__ tape_index,
                          int32_t* const __restrict__ image,
                          const uint32_t tiles_per_side,

                          TileNode* const __restrict__ in_tiles,
                          const int32_t in_tile_count,

                          const Interval* __restrict__ values,

                          const int tile_size_px,
                          float* __restrict__ const heatmap)
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

    constexpr static int CHOICE_ARRAY_SIZE = 256;
    uint32_t choices[CHOICE_ARRAY_SIZE] = {0};
    int choice_index = 0;
    bool has_any_choice = false;

    unsigned work = 0;
    while (1) {
        const uint64_t d = *++data;
        if (!OP(&d)) {
            break;
        }
        work++;
        switch (OP(&d)) {
            case GPU_OP_JUMP: data += JUMP_TARGET(&d); continue;

#define lhs slots[I_LHS(&d)]
#define rhs slots[I_RHS(&d)]
#define imm IMM(&d)
#define out slots[I_OUT(&d)]

            case GPU_OP_SQUARE_LHS: out = square(lhs); break;
            case GPU_OP_SQRT_LHS:   out = sqrt(lhs); break;
            case GPU_OP_NEG_LHS:    out = -lhs; break;
            case GPU_OP_SIN_LHS:    out = sin(lhs); break;
            case GPU_OP_COS_LHS:    out = cos(lhs); break;
            case GPU_OP_ASIN_LHS:   out = asin(lhs); break;
            case GPU_OP_ACOS_LHS:   out = acos(lhs); break;
            case GPU_OP_ATAN_LHS:   out = atan(lhs); break;
            case GPU_OP_EXP_LHS:    out = exp(lhs); break;
            case GPU_OP_ABS_LHS:    out = abs(lhs); break;
            case GPU_OP_LOG_LHS:    out = log(lhs); break;

            // Commutative opcodes
            case GPU_OP_ADD_LHS_IMM: out = lhs + imm; break;
            case GPU_OP_ADD_LHS_RHS: out = lhs + rhs; break;
            case GPU_OP_MUL_LHS_IMM: out = lhs * imm; break;
            case GPU_OP_MUL_LHS_RHS: out = lhs * rhs; break;

#define CHOICE(f, a, b) {                                               \
    int c = 0;                                                          \
    out = f(a, b, c);                                                   \
    if (choice_index < CHOICE_ARRAY_SIZE * 16) {                        \
        choices[choice_index / 16] |= (c << ((choice_index % 16) * 2)); \
    }                                                                   \
    choice_index++;                                                     \
    has_any_choice |= (c != 0);                                         \
    break;                                                              \
}
            case GPU_OP_MIN_LHS_IMM: CHOICE(min, lhs, imm);
            case GPU_OP_MIN_LHS_RHS: CHOICE(min, lhs, rhs);
            case GPU_OP_MAX_LHS_IMM: CHOICE(max, lhs, imm);
            case GPU_OP_MAX_LHS_RHS: CHOICE(max, lhs, rhs);
            case GPU_OP_RAD_LHS_RHS: out = sqrt(square(lhs) + square(rhs)); break;

            // Non-commutative opcodes
            case GPU_OP_SUB_LHS_IMM: out = lhs - imm; break;
            case GPU_OP_SUB_IMM_RHS: out = imm - rhs; break;
            case GPU_OP_SUB_LHS_RHS: out = lhs - rhs; break;
            case GPU_OP_DIV_LHS_IMM: out = lhs / imm; break;
            case GPU_OP_DIV_IMM_RHS: out = imm / rhs; break;
            case GPU_OP_DIV_LHS_RHS: out = lhs / rhs; break;

            case GPU_OP_COPY_IMM: out = Interval(imm); break;
            case GPU_OP_COPY_LHS: out = lhs; break;
            case GPU_OP_COPY_RHS: out = rhs; break;

            default: assert(false);
        }
#undef lhs
#undef rhs
#undef imm
#undef out
    }

    // Check the result
    const uint8_t i_out = I_OUT(data);

    {   // Write the work to the heatmap
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        for (int x=0; x < tile_size_px; ++x) {
            for (int y=0; y < tile_size_px; ++y) {
                int px = x + pos.x * tile_size_px;
                int py = y + pos.y * tile_size_px;
                atomicAdd(&heatmap[px + py * tile_size_px * tiles_per_side],
                          work / powf(tile_size_px, 2.0f));
            }
        }
        work = 0;
    }

    // Empty
    if (slots[i_out].lower() > 0.0f) {
        in_tiles[tile_index].position = -1;
        return;
    }

    // Masked
    if (DIMENSION == 3) {
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        if (image[pos.w] > pos.z) {
            in_tiles[tile_index].position = -1;
            return;
        }
    }

    // Filled
    if (slots[i_out].upper() < 0.0f) {
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        in_tiles[tile_index].position = -1;
        if (DIMENSION == 3) {
            atomicMax(&image[pos.w], pos.z);
        } else {
            image[pos.w] = 1;
        }
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

    // Check to make sure the tape isn't full
    // This doesn't mean that we'll successfully claim a chunk, because
    // other threads could claim chunks before us, but it's a way to check
    // quickly (and prevents tape_index from getting absurdly large).
    if (*tape_index >= NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE) {
        return;
    }

    // Claim a chunk of tape
    int32_t out_index = atomicAdd(tape_index, SUBTAPE_CHUNK_SIZE);
    int32_t out_offset = SUBTAPE_CHUNK_SIZE;

    // If we've run out of tape, then immediately return
    if (out_index + out_offset >= NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE) {
        return;
    }

    // Write out the end of the tape, which is the same as the ending
    // of the previous tape (0 opcode, with i_out as the last slot)
    out_offset--;
    tape_data[out_index + out_offset] = *data;

    while (1) {
        uint64_t d = *--data;
        if (!OP(&d)) {
            break;
        }
        work++;
        const uint8_t op = OP(&d);
        if (op == GPU_OP_JUMP) {
            data += JUMP_TARGET(&d);
            continue;
        }

        const bool has_choice = op >= GPU_OP_MIN_LHS_IMM &&
                                op <= GPU_OP_MAX_LHS_RHS;
        choice_index -= has_choice;

        const uint8_t i_out = I_OUT(&d);
        if (!active[i_out]) {
            continue;
        }

        assert(!has_choice || choice_index >= 0);

        const int choice = (has_choice && choice_index < CHOICE_ARRAY_SIZE * 16)
            ? ((choices[choice_index / 16] >>
              ((choice_index % 16) * 2)) & 3)
            : 0;

        // If we're about to write a new piece of data to the tape,
        // (and are done with the current chunk), then we need to
        // add another link to the linked list.
        --out_offset;
        if (out_offset == 0) {
            const int32_t prev_index = out_index;

            // Early exit if we can't finish writing out this tape
            if (*tape_index >= NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE) {
                const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
                for (int x=0; x < tile_size_px; ++x) {
                    for (int y=0; y < tile_size_px; ++y) {
                        int px = x + pos.x * tile_size_px;
                        int py = y + pos.y * tile_size_px;
                        atomicAdd(&heatmap[px + py * tile_size_px * tiles_per_side],
                                  work / powf(tile_size_px, 2.0f));
                    }
                }
                return;
            }
            out_index = atomicAdd(tape_index, SUBTAPE_CHUNK_SIZE);
            out_offset = SUBTAPE_CHUNK_SIZE;

            // Later exit if we claimed a chunk that exceeds the tape array
            if (out_index + out_offset >= NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE) {
                const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
                for (int x=0; x < tile_size_px; ++x) {
                    for (int y=0; y < tile_size_px; ++y) {
                        int px = x + pos.x * tile_size_px;
                        int py = y + pos.y * tile_size_px;
                        atomicAdd(&heatmap[px + py * tile_size_px * tiles_per_side],
                                  work / powf(tile_size_px, 2.0f));
                    }
                }
                return;
            }
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
        if (choice == 0) {
            const uint8_t i_lhs = I_LHS(&d);
            if (i_lhs) {
                active[i_lhs] = true;
            }
            const uint8_t i_rhs = I_RHS(&d);
            if (i_rhs) {
                active[i_rhs] = true;
            }
        } else if (choice == 1 /* LHS */) {
            // The non-immediate is always the LHS in commutative ops, and
            // min/max (the only clauses that produce a choice) are commutative
            const uint8_t i_lhs = I_LHS(&d);
            active[i_lhs] = true;
            if (i_lhs == i_out) {
                ++out_offset;
                continue;
            } else {
                OP(&d) = GPU_OP_COPY_LHS;
            }
        } else if (choice == 2 /* RHS */) {
            const uint8_t i_rhs = I_RHS(&d);
            if (i_rhs) {
                active[i_rhs] = true;
                if (i_rhs == i_out) {
                    ++out_offset;
                    continue;
                } else {
                    OP(&d) = GPU_OP_COPY_RHS;
                }
            } else {
                OP(&d) = GPU_OP_COPY_IMM;
            }
        }
        tape_data[out_index + out_offset] = d;
    }

    {   // Accumulate the work of walking backwards through the tape
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        for (int x=0; x < tile_size_px; ++x) {
            for (int y=0; y < tile_size_px; ++y) {
                int px = x + pos.x * tile_size_px;
                int py = y + pos.y * tile_size_px;
                atomicAdd(&heatmap[px + py * tile_size_px * tiles_per_side],
                          work / powf(tile_size_px, 2.0f));
            }
        }
    }

    // Write the beginning of the tape
    out_offset--;
    tape_data[out_index + out_offset] = *data;

    // Record the beginning of the tape in the output tile
    in_tiles[tile_index].tape = out_index + out_offset;
}

template <unsigned DIMENSION>
__global__
void eval_voxels_f_heatmap(const uint64_t* const __restrict__ tape_data,
                           int32_t* const __restrict__ image,
                           const uint32_t tiles_per_side,

                           TileNode* const __restrict__ in_tiles,
                           const int32_t in_tile_count,

                           const float2* const __restrict__ values,

                           float* __restrict__ const heatmap)
{
    // Each tile is executed by 32 threads (one for each pair of voxels, so
    // we can do all of our load/stores as float2s and make memory happier).
    //
    // This is different from the eval_tiles_i function, which evaluates one
    // tile per thread, because the tiles are already expanded by 64x by the
    // time they're stored in the in_tiles list.
    const int32_t voxel_index = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t tile_index = voxel_index / 32;
    if (tile_index >= in_tile_count) {
        return;
    }

    // Check whether this pixel is masked in the output image
    if (DIMENSION == 3) {
        const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
        const int4 sub = unpack(threadIdx.x % 32, 4);

        const int32_t px = pos.x * 4 + sub.x;
        const int32_t py = pos.y * 4 + sub.y;
        const int32_t pz = pos.z * 4 + sub.z;

        // Early return if this pixel won't ever be filled
        if (image[px + py * tiles_per_side * 4] >= pz + 2) {
            return;
        }
    }

    float2 slots[128];

    // Pick out the tape based on the pointer stored in the tiles list
    const uint64_t* __restrict__ data = &tape_data[in_tiles[tile_index].tape];
    slots[((const uint8_t*)tape_data)[1]] = values[voxel_index * 3];
    slots[((const uint8_t*)tape_data)[2]] = values[voxel_index * 3 + 1];
    slots[((const uint8_t*)tape_data)[3]] = values[voxel_index * 3 + 2];

    unsigned work = 0;
    while (1) {
        const uint64_t d = *++data;
        if (!OP(&d)) {
            break;
        }
        work++;
        switch (OP(&d)) {
            case GPU_OP_JUMP: data += JUMP_TARGET(&d); continue;

#define lhs slots[I_LHS(&d)]
#define rhs slots[I_RHS(&d)]
#define imm IMM(&d)
#define out slots[I_OUT(&d)]

            case GPU_OP_SQUARE_LHS: out = make_float2(lhs.x * lhs.x, lhs.y * lhs.y); break;
            case GPU_OP_SQRT_LHS: out = make_float2(sqrtf(lhs.x), sqrtf(lhs.y)); break;
            case GPU_OP_NEG_LHS: out = make_float2(-lhs.x, -lhs.y); break;
            case GPU_OP_SIN_LHS: out = make_float2(sinf(lhs.x), sinf(lhs.y)); break;
            case GPU_OP_COS_LHS: out = make_float2(cosf(lhs.x), cosf(lhs.y)); break;
            case GPU_OP_ASIN_LHS: out = make_float2(asinf(lhs.x), asinf(lhs.y)); break;
            case GPU_OP_ACOS_LHS: out = make_float2(acosf(lhs.x), acosf(lhs.y)); break;
            case GPU_OP_ATAN_LHS: out = make_float2(atanf(lhs.x), atanf(lhs.y)); break;
            case GPU_OP_EXP_LHS: out = make_float2(expf(lhs.x), expf(lhs.y)); break;
            case GPU_OP_ABS_LHS: out = make_float2(fabsf(lhs.x), fabsf(lhs.y)); break;
            case GPU_OP_LOG_LHS: out = make_float2(logf(lhs.x), logf(lhs.y)); break;

            // Commutative opcodes
            case GPU_OP_ADD_LHS_IMM: out = make_float2(lhs.x + imm, lhs.y + imm); break;
            case GPU_OP_ADD_LHS_RHS: out = make_float2(lhs.x + rhs.x, lhs.y + rhs.y); break;
            case GPU_OP_MUL_LHS_IMM: out = make_float2(lhs.x * imm, lhs.y * imm); break;
            case GPU_OP_MUL_LHS_RHS: out = make_float2(lhs.x * rhs.x, lhs.y * rhs.y); break;
            case GPU_OP_MIN_LHS_IMM: out = make_float2(fminf(lhs.x, imm), fminf(lhs.y, imm)); break;
            case GPU_OP_MIN_LHS_RHS: out = make_float2(fminf(lhs.x, rhs.x), fminf(lhs.y, rhs.y)); break;
            case GPU_OP_MAX_LHS_IMM: out = make_float2(fmaxf(lhs.x, imm), fmaxf(lhs.y, imm)); break;
            case GPU_OP_MAX_LHS_RHS: out = make_float2(fmaxf(lhs.x, rhs.x), fmaxf(lhs.y, rhs.y)); break;
            case GPU_OP_RAD_LHS_RHS: out = make_float2(sqrtf(lhs.x * lhs.x + rhs.x * rhs.x),
                                                       sqrtf(lhs.y * lhs.y + rhs.y * rhs.y)); break;

            // Non-commutative opcodes
            case GPU_OP_SUB_LHS_IMM: out = make_float2(lhs.x - imm, lhs.y - imm); break;
            case GPU_OP_SUB_IMM_RHS: out = make_float2(imm - rhs.x, imm - rhs.y); break;
            case GPU_OP_SUB_LHS_RHS: out = make_float2(lhs.x - rhs.x, lhs.y - rhs.y); break;

            case GPU_OP_DIV_LHS_IMM: out = make_float2(lhs.x / imm, lhs.y / imm); break;
            case GPU_OP_DIV_IMM_RHS: out = make_float2(imm / rhs.x, imm / rhs.y); break;
            case GPU_OP_DIV_LHS_RHS: out = make_float2(lhs.x / rhs.x, lhs.y / rhs.y); break;

            case GPU_OP_COPY_IMM: out = make_float2(imm, imm); break;
            case GPU_OP_COPY_LHS: out = make_float2(lhs.x, lhs.y); break;
            case GPU_OP_COPY_RHS: out = make_float2(rhs.x, rhs.y); break;

#undef lhs
#undef rhs
#undef imm
#undef out
        }
    }

    // Check the result
    const uint8_t i_out = I_OUT(data);

    const int4 pos = unpack(in_tiles[tile_index].position, tiles_per_side);
    if (DIMENSION == 3) {
        const int4 sub = unpack(threadIdx.x % 32, 4);
        // The second voxel is always higher in Z, so it masks the lower voxel
        if (slots[i_out].y < 0.0f) {
            const int32_t px = pos.x * 4 + sub.x;
            const int32_t py = pos.y * 4 + sub.y;
            const int32_t pz = pos.z * 4 + sub.z + 2;

            atomicMax(&image[px + py * tiles_per_side * 4], pz);
        } else if (slots[i_out].x < 0.0f) {
            const int32_t px = pos.x * 4 + sub.x;
            const int32_t py = pos.y * 4 + sub.y;
            const int32_t pz = pos.z * 4 + sub.z;

            atomicMax(&image[px + py * tiles_per_side * 4], pz);
        }
        const int px = pos.x * 4 + sub.x;
        const int py = pos.y * 4 + sub.y;
        atomicAdd(&heatmap[px + py * tiles_per_side * 4], work);
    } else if (DIMENSION == 2) {
        const int4 sub = unpack(threadIdx.x % 32, 8);
        if (slots[i_out].y < 0.0f) {
            const int32_t px = pos.x * 8 + sub.x;
            const int32_t py = pos.y * 8 + sub.y + 4;

            image[px + py * tiles_per_side * 8] = 1;
        }
        if (slots[i_out].x < 0.0f) {
            const int32_t px = pos.x * 8 + sub.x;
            const int32_t py = pos.y * 8 + sub.y;

            image[px + py * tiles_per_side * 8] = 1;
        }
        const int px = pos.x * 8 + sub.x;
        const int py = pos.y * 8 + sub.y;
        atomicAdd(&heatmap[px + py * tiles_per_side * 8], work / 2.0f);
        atomicAdd(&heatmap[px + (py + 4) * tiles_per_side * 8], work / 2.0f);
    }
}

Ptr<float[]> Context::render2D_heatmap(const Tape& tape,
                                       const Eigen::Matrix3f& mat,
                                       const float z)
{
    // Build the heatmap for this render
    Ptr<float[]> heatmap(CUDA_MALLOC(float, pow(image_size_px, 2)));
    cudaMemset(heatmap.get(), 0, sizeof(float) * pow(image_size_px, 2));

    // Reset the tape index and copy the tape to the beginning of the
    // context's tape buffer area.
    *tape_index = tape.length;
    cudaMemcpyAsync(tape_data.get(), tape.data.get(),
                    sizeof(uint64_t) * tape.length,
                    cudaMemcpyDeviceToDevice);

    // Reset all of the data arrays.  In 2D, we only use stages 0, 2, and 3
    // for 64^2, 8^2, and per-voxel evaluation steps.
    CUDA_CHECK(cudaMemsetAsync(stages[0].filled.get(), 0, sizeof(int32_t) *
                               pow(image_size_px / 64, 2)));
    CUDA_CHECK(cudaMemsetAsync(stages[2].filled.get(), 0, sizeof(int32_t) *
                               pow(image_size_px / 8, 2)));
    CUDA_CHECK(cudaMemsetAsync(stages[3].filled.get(), 0, sizeof(int32_t) *
                               pow(image_size_px, 2)));

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 64x64 tiles
    ////////////////////////////////////////////////////////////////////////////

    // Go the whole list of first-stage tiles, assigning each to
    // be [position, tape = 0, next = -1]
    unsigned count = pow(image_size_px / 64, 2);
    unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;
    preload_tiles<<<num_blocks, NUM_THREADS>>>(stages[0].tiles.get(), count);

    // Iterate over 64^2, 8^2 tiles
    for (unsigned i=0; i < 3; i += 2) {
        const unsigned tile_size_px = i ? 8 : 64;
        const unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;

        if (values_size < num_blocks * NUM_THREADS * 3) {
            values.reset(CUDA_MALLOC(Interval, num_blocks * NUM_THREADS * 3));
            values_size = num_blocks * NUM_THREADS * 3;
        }

        // Unpack position values into interval X/Y/Z in the values array
        // This is done in a separate kernel to avoid bloating the
        // eval_tiles_i kernel with more registers, which is detrimental
        // to occupancy.
        calculate_intervals_2d<<<num_blocks, NUM_THREADS>>>(
            stages[i].tiles.get(),
            count,
            image_size_px / tile_size_px,
            mat, z,
            reinterpret_cast<Interval*>(values.get()));

        // Do the actual tape evaluation, which is the expensive step
        eval_tiles_i_heatmap<2><<<num_blocks, NUM_THREADS>>>(
            tape_data.get(),
            tape_index.get(),
            stages[i].filled.get(),
            image_size_px / tile_size_px,

            stages[i].tiles.get(),
            count,

            reinterpret_cast<Interval*>(values.get()),

            tile_size_px,
            heatmap.get());

        // Mark the total number of active tiles (from this stage) to 0
        cudaMemsetAsync(num_active_tiles.get(), 0, sizeof(int32_t));

        // Count up active tiles, to figure out how much memory needs to be
        // allocated in the next stage.
        assign_next_nodes<<<num_blocks, NUM_THREADS>>>(
            stages[i].tiles.get(),
            count,
            num_active_tiles.get());

        // Count the number of active tiles, which have been accumulated
        // through repeated calls to assign_next_nodes
        int32_t active_tile_count;
        cudaMemcpy(&active_tile_count, num_active_tiles.get(), sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        if (i == 0) {
            active_tile_count *= 64;
        }

        // Make sure that the subtiles buffer has enough room
        // This wastes a small amount of data for the per-pixel evaluation,
        // where the `next` indexes aren't used, but it's relatively small.
        const int next = i ? 3 : 2;
        if (active_tile_count > stages[next].tile_array_size) {
            stages[next].tile_array_size = active_tile_count;
            stages[next].tiles.reset(CUDA_MALLOC(TileNode, active_tile_count));
        }

        if (i < 2) {
            // Build the new tile list from active tiles in the previous list
            subdivide_active_tiles_2d<<<num_blocks*64, NUM_THREADS>>>(
                stages[i].tiles.get(),
                count,
                image_size_px / tile_size_px,
                stages[next].tiles.get());
        } else {
            // Special case for per-pixel evaluation, which
            // doesn't unpack every single pixel (since that would take up
            // 64x extra space).
            copy_active_tiles<<<num_blocks, NUM_THREADS>>>(
                stages[i].tiles.get(),
                count,
                stages[next].tiles.get());
        }

        {   // Copy filled tiles into the next level's image (expanding them
            // by 64x).  This is cleaner that accumulating all of the levels
            // in a single pass, and could (possibly?) help with skipping
            // fully occluded tiles.
            const unsigned next_tile_size = tile_size_px / 8;
            const uint32_t u = ((image_size_px / next_tile_size) / 32);
            copy_filled_2d<<<dim3(u + 1, u + 1), dim3(32, 32)>>>(
                    stages[i].filled.get(),
                    stages[next].filled.get(),
                    image_size_px / next_tile_size);
        }

        // Assign the next number of tiles to evaluate
        count = active_tile_count;
    }

    // Time to render individual pixels!
    num_blocks = (count + NUM_TILES - 1) / NUM_TILES;
    const size_t num_values = num_blocks * NUM_TILES * 32 * 3;
    if (values_size < num_values) {
        values.reset(CUDA_MALLOC(float2, num_values));
        values_size = num_values;
    }
    calculate_pixels<<<num_blocks, NUM_TILES * 32>>>(
        stages[3].tiles.get(),
        count,
        image_size_px / 8,
        mat, z,
        reinterpret_cast<float2*>(values.get()));
    eval_voxels_f_heatmap<2><<<num_blocks, NUM_TILES * 32>>>(
        tape_data.get(),
        stages[3].filled.get(),
        image_size_px / 8,

        stages[3].tiles.get(),
        count,

        reinterpret_cast<float2*>(values.get()),
        heatmap.get());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (unsigned x=0; x < image_size_px; ++x) {
        for (unsigned y=0; y < image_size_px; ++y) {
            heatmap[x + y * image_size_px] /= tape.length - 2;
        }
    }
    return heatmap;
}

Ptr<float[]> Context::render3D_heatmap(const Tape& tape,
                                       const Eigen::Matrix4f& mat)
{
    // Build the heatmap for this render
    Ptr<float[]> heatmap(CUDA_MALLOC(float, pow(image_size_px, 2)));
    cudaMemset(heatmap.get(), 0, sizeof(float) * pow(image_size_px, 2));

    // Reset the tape index and copy the tape to the beginning of the
    // context's tape buffer area.
    *tape_index = tape.length;
    cudaMemcpyAsync(tape_data.get(), tape.data.get(),
                    sizeof(uint64_t) * tape.length,
                    cudaMemcpyDeviceToDevice);

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation of 64x64x64 tiles
    ////////////////////////////////////////////////////////////////////////////

    // Reset all of the data arrays
    for (unsigned i=0; i < 4; ++i) {
        const unsigned tile_size_px = 64 / (1 << (i * 2));
        CUDA_CHECK(cudaMemsetAsync(stages[i].filled.get(), 0, sizeof(int32_t) *
                                   pow(image_size_px / tile_size_px, 2)));
    }
    CUDA_CHECK(cudaMemsetAsync(normals.get(), 0, sizeof(uint32_t) *
                               pow(image_size_px, 2)));

    // Go the whole list of first-stage tiles, assigning each to
    // be [position, tape = 0, next = -1]
    unsigned count = pow(image_size_px / 64, 3);
    unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;
    preload_tiles<<<num_blocks, NUM_THREADS>>>(stages[0].tiles.get(), count);

    // Iterate over 64^3, 16^3, 4^3 tiles
    for (unsigned i=0; i < 3; ++i) {
        //printf("BEGINNING STAGE %u\n", i);
        const unsigned tile_size_px = 64 / (1 << (i * 2));
        const unsigned num_blocks = (count + NUM_THREADS - 1) / NUM_THREADS;

        if (values_size < num_blocks * NUM_THREADS * 3) {
            values.reset(CUDA_MALLOC(Interval, num_blocks * NUM_THREADS * 3));
            values_size = num_blocks * NUM_THREADS * 3;
        }

        // Unpack position values into interval X/Y/Z in the values array
        // This is done in a separate kernel to avoid bloating the
        // eval_tiles_i kernel with more registers, which is detrimental
        // to occupancy.
        calculate_intervals_3d<<<num_blocks, NUM_THREADS>>>(
            stages[i].tiles.get(),
            count,
            image_size_px / tile_size_px,
            mat,
            reinterpret_cast<Interval*>(values.get()));

        // Mark every tile which is covered in the image as masked,
        // which means it will be skipped later on.  We do this again below,
        // but it's basically free, so we should do it here and simplify
        // the logic in eval_tiles_i.
        mask_filled_tiles<<<num_blocks, NUM_THREADS>>>(
            stages[i].filled.get(),
            image_size_px / tile_size_px,
            stages[i].tiles.get(),
            count);

        // Do the actual tape evaluation, which is the expensive step
        eval_tiles_i_heatmap<3><<<num_blocks, NUM_THREADS>>>(
            tape_data.get(),
            tape_index.get(),
            stages[i].filled.get(),
            image_size_px / tile_size_px,

            stages[i].tiles.get(),
            count,

            reinterpret_cast<Interval*>(values.get()),
            tile_size_px,
            heatmap.get());

        // Mark the total number of active tiles (from this stage) to 0
        cudaMemsetAsync(num_active_tiles.get(), 0, sizeof(int32_t));

        // Now that we have evaluated every tile at this level, we do one more
        // round of occlusion culling before accumulating tiles to render at
        // the next phase.
        mask_filled_tiles<<<num_blocks, NUM_THREADS>>>(
            stages[i].filled.get(),
            image_size_px / tile_size_px,
            stages[i].tiles.get(),
            count);

        // Count up active tiles, to figure out how much memory needs to be
        // allocated in the next stage.
        assign_next_nodes<<<num_blocks, NUM_THREADS>>>(
            stages[i].tiles.get(),
            count,
            num_active_tiles.get());

        // Count the number of active tiles, which have been accumulated
        // through repeated calls to assign_next_nodes
        int32_t active_tile_count;
        cudaMemcpy(&active_tile_count, num_active_tiles.get(), sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        if (i < 2) {
            active_tile_count *= 64;
        }

        // Make sure that the subtiles buffer has enough room
        // This wastes a small amount of data for the per-pixel evaluation,
        // where the `next` indexes aren't used, but it's relatively small.
        if (active_tile_count > stages[i + 1].tile_array_size) {
            stages[i + 1].tile_array_size = active_tile_count;
            stages[i + 1].tiles.reset(CUDA_MALLOC(TileNode, active_tile_count));
        }

        if (i < 2) {
            // Build the new tile list from active tiles in the previous list
            subdivide_active_tiles_3d<<<num_blocks*64, NUM_THREADS>>>(
                stages[i].tiles.get(),
                count,
                image_size_px / tile_size_px,
                stages[i + 1].tiles.get());
        } else {
            // Special case for per-pixel evaluation, which
            // doesn't unpack every single pixel (since that would take up
            // 64x extra space).
            copy_active_tiles<<<num_blocks, NUM_THREADS>>>(
                stages[i].tiles.get(),
                count,
                stages[i + 1].tiles.get());
        }

        {   // Copy filled tiles into the next level's image (expanding them
            // by 64x).  This is cleaner that accumulating all of the levels
            // in a single pass, and could (possibly?) help with skipping
            // fully occluded tiles.
            const unsigned next_tile_size = tile_size_px / 4;
            const uint32_t u = ((image_size_px / next_tile_size) / 32);
            copy_filled_3d<<<dim3(u + 1, u + 1), dim3(32, 32)>>>(
                    stages[i].filled.get(),
                    stages[i + 1].filled.get(),
                    image_size_px / next_tile_size);
        }

        // Assign the next number of tiles to evaluate
        count = active_tile_count;
    }

    // Time to render individual pixels!
    num_blocks = (count + NUM_TILES - 1) / NUM_TILES;
    const size_t num_values = num_blocks * NUM_TILES * 32 * 3;
    if (values_size < num_values) {
        values.reset(CUDA_MALLOC(float2, num_values));
        values_size = num_values;
    }
    calculate_voxels<<<num_blocks, NUM_TILES * 32>>>(
        stages[3].tiles.get(),
        count,
        image_size_px / 4,
        mat,
        reinterpret_cast<float2*>(values.get()));
    eval_voxels_f_heatmap<3><<<num_blocks, NUM_TILES * 32>>>(
        tape_data.get(),
        stages[3].filled.get(),
        image_size_px / 4,

        stages[3].tiles.get(),
        count,

        reinterpret_cast<float2*>(values.get()),
        heatmap.get());

    {   // Then render normals into those pixels
        const uint32_t u = ((image_size_px + 15) / 16);
        eval_pixels_d<<<dim3(u, u), dim3(16, 16)>>>(
                tape_data.get(),
                stages[3].filled.get(),
                normals.get(),
                image_size_px,
                mat,
                stages[0].tiles.get(),
                stages[1].tiles.get(),
                stages[2].tiles.get());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (unsigned x=0; x < image_size_px; ++x) {
        for (unsigned y=0; y < image_size_px; ++y) {
            heatmap[x + y * image_size_px] /= tape.length - 2;
        }
    }
    return heatmap;
}
