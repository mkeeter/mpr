#include <cassert>
#include "renderable.hpp"

////////////////////////////////////////////////////////////////////////////////

template <typename R, unsigned T, unsigned D>
__device__ void storeAxes(const uint32_t tile,
                          const View& v, const Tiles<T, D>& tiles, const Tape& tape,
                          R* const __restrict__ regs)
{
   // Prepopulate axis values
    const float3 lower = tiles.tileToLowerPos(tile);
    const float3 upper = tiles.tileToUpperPos(tile);

    Interval X(lower.x, upper.x);
    Interval Y(lower.y, upper.y);
    Interval Z(lower.z, upper.z);

    if (tape.axes.reg[0] != UINT16_MAX) {
        regs[tape.axes.reg[0]][threadIdx.x] = X * v.scale - v.center[0];
    }
    if (tape.axes.reg[1] != UINT16_MAX) {
        regs[tape.axes.reg[1]][threadIdx.x] = Y * v.scale - v.center[1];
    }
    if (tape.axes.reg[2] != UINT16_MAX) {
        regs[tape.axes.reg[2]][threadIdx.x] = (D == 3)
            ? (Z * v.scale - v.center[2])
            : Interval{v.center[2], v.center[2]};
    }
}

template <typename A, typename B>
__device__ inline Interval intervalOp(uint8_t op, A lhs, B rhs,
                                      uint64_t& choice, uint8_t choice_index)
{
    using namespace libfive::Opcode;
    switch (op) {
        case OP_SQUARE: return square(lhs);
        case OP_SQRT: return sqrt(lhs);
        case OP_NEG: return -lhs;
        // Skipping transcendental functions for now

        case OP_ADD: return lhs + rhs;
        case OP_MUL: return lhs * rhs;
        case OP_DIV: return lhs / rhs;
        case OP_MIN: if (upper(lhs) < lower(rhs)) {
                         choice |= (1UL << choice_index);
                         return lhs;
                     } else if (upper(rhs) < lower(lhs)) {
                         choice |= (2UL << choice_index);
                         return rhs;
                     } else {
                         return min(lhs, rhs);
                     }
        case OP_MAX: if (lower(lhs) > upper(rhs)) {
                         choice |= (1UL << choice_index);
                         return lhs;
                     } else if (lower(rhs) > upper(lhs)) {
                         choice |= (2UL << choice_index);
                         return rhs;
                     } else {
                         return max(lhs, rhs);
                     }
        case OP_SUB: return lhs - rhs;

        // Skipping various hard functions here
        default: break;
    }
    return {0.0f, 0.0f};
}

template <typename A, typename B>
__device__ inline Deriv derivOp(uint8_t op, A lhs, B rhs)
{
    using namespace libfive::Opcode;
    switch (op) {
        case OP_SQUARE: return lhs * lhs;
        case OP_SQRT: return sqrt(lhs);
        case OP_NEG: return -lhs;
        // Skipping transcendental functions for now

        case OP_ADD: return lhs + rhs;
        case OP_MUL: return lhs * rhs;
        case OP_DIV: return lhs / rhs;
        case OP_MIN: return min(lhs, rhs);
        case OP_MAX: return max(lhs, rhs);
        case OP_SUB: return lhs - rhs;

        // Skipping various hard functions here
        default: break;
    }
    return {0.0f, 0.0f, 0.0f, 0.0f};
}

////////////////////////////////////////////////////////////////////////////////

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
TileRenderer<TILE_SIZE_PX, DIMENSION>::TileRenderer(
        const Tape& tape, Subtapes& subtapes, Image& image)
    : tape(tape), subtapes(subtapes), image(image),
      tiles(image.size_px),

      regs(CUDA_MALLOC(Registers, LIBFIVE_CUDA_TILE_BLOCKS *
                                      tape.num_regs)),
      active(CUDA_MALLOC(ActiveArray, LIBFIVE_CUDA_TILE_BLOCKS *
                                      tape.num_regs)),
      choices(tape.num_csg_choices ?
              CUDA_MALLOC(ChoiceArray,
                    LIBFIVE_CUDA_TILE_BLOCKS *
                    ((tape.num_csg_choices + 31) / 32))
              : nullptr)
{
    // Nothing to do here
}

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
TileRenderer<TILE_SIZE_PX, DIMENSION>::~TileRenderer()
{
    CUDA_CHECK(cudaFree(regs));
    CUDA_CHECK(cudaFree(active));
    CUDA_CHECK(cudaFree(choices));
}

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
__device__
void TileRenderer<TILE_SIZE_PX, DIMENSION>::check(
        const uint32_t tile, const View& v)
{
    auto regs = this->regs + tape.num_regs * blockIdx.x;
    storeAxes(tile, v, tiles, tape, regs);

    // Unpack a 1D offset into the data arrays
    auto choices = this->choices + ((tape.num_csg_choices + 31) / 32) * blockIdx.x;
    uint64_t choice = 0;
    uint8_t choice_index = 0;

    const Clause* __restrict__ clause_ptr = &tape[0];
    const float* __restrict__ constant_ptr = &tape.constant(0);
    const auto num_clauses = tape.num_clauses;

    for (uint32_t i=0; i < num_clauses; ++i) {
        using namespace libfive::Opcode;

        const Clause c = clause_ptr[i];
        Interval out;
        switch (c.banks) {
            case 0: // Interval op Interval
                out = intervalOp<Interval, Interval>(c.opcode,
                        regs[c.lhs][threadIdx.x],
                        regs[c.rhs][threadIdx.x],
                        choice, choice_index);
                break;
            case 1: // Constant op Interval
                out = intervalOp<float, Interval>(c.opcode,
                        constant_ptr[c.lhs],
                        regs[c.rhs][threadIdx.x],
                        choice, choice_index);
                break;
            case 2: // Interval op Constant
                out = intervalOp<Interval, float>(c.opcode,
                        regs[c.lhs][threadIdx.x],
                        constant_ptr[c.rhs],
                        choice, choice_index);
                break;
            case 3: // Constant op Constant
                out = intervalOp<float, float>(c.opcode,
                        constant_ptr[c.lhs],
                        constant_ptr[c.rhs],
                        choice, choice_index);
                break;
        }

        if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            choice_index += 2;
            if (choice_index == sizeof(choice) * 8) {
                (*(choices++))[threadIdx.x] = choice;
                choice = 0;
                choice_index = 0;
            }
        }

        regs[c.out][threadIdx.x] = out;
    }

    const Clause c = clause_ptr[num_clauses - 1];
    const Interval result = regs[c.out][threadIdx.x];

    // If this tile is unambiguously filled, then mark it at the end
    // of the tiles list
    if (result.upper() < 0.0f) {
        tiles.insertFilled(tile);
        return;
    }

    // If the tile is ambiguous, then record it as needing further refinement
    else if ((result.lower() <= 0.0f && result.upper() >= 0.0f)
            || isnan(result.lower())
            || isnan(result.upper()))
    {
        tiles.insertActive(tile);
    }
    else {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Now, we build a tape for this tile (if it's active).  If it isn't active,
    // then we use the thread to help copy stuff to shared memory, but don't
    // write any tape data out.

    // Pick a subset of the active array to use for this block
    auto active = this->active + blockIdx.x * tape.num_regs;

    for (uint32_t r=0; r < tape.num_regs; ++r) {
        active[r][threadIdx.x] = false;
    }

    // Mark the root of the tree as true
    active[tape[num_clauses - 1].out][threadIdx.x] = true;

    uint32_t subtape_index = 0;
    uint32_t s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;

    // Claim a subtape to populate
    subtape_index = subtapes.claim();

    // Since we're reversing the tape, this is going to be the
    // end of the linked list (i.e. next = 0)
    subtapes.next[subtape_index] = 0;

    // Walk from the root of the tape downwards
    Clause* __restrict__ out = subtapes.data[subtape_index];

    bool terminal = true;
    for (uint32_t i=0; i < num_clauses; i++) {
        using namespace libfive::Opcode;
        Clause c = clause_ptr[num_clauses - i - 1];

        if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            if (choice_index == 0) {
                choice_index = sizeof(choice) * 8;
                choice = (*(--choices))[threadIdx.x];
            }
            choice_index -= 2;
        }

        if (active[c.out][threadIdx.x]) {
            active[c.out][threadIdx.x] = false;
            if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
                const uint8_t choice_ = (choice >> choice_index) & 3;
                if (choice_ == 1) {
                    if (!(c.banks & 1)) {
                        active[c.lhs][threadIdx.x] = true;
                        if (c.lhs == c.out) {
                            continue;
                        }
                        c.rhs = c.lhs;
                        c.banks = 0;
                    } else {
                        c.rhs = c.lhs;
                        c.banks = 3;
                    }
                } else if (choice_ == 2) {
                    if (!(c.banks & 2)) {
                        active[c.rhs][threadIdx.x] = true;
                        if (c.rhs == c.out) {
                            continue;
                        }
                        c.lhs = c.rhs;
                        c.banks = 0;
                    } else {
                        c.lhs = c.rhs;
                        c.banks = 3;
                    }
                } else if (choice_ == 0) {
                    terminal = false;
                    if (!(c.banks & 1)) {
                        active[c.lhs][threadIdx.x] = true;
                    }
                    if (!(c.banks & 2)) {
                        active[c.rhs][threadIdx.x] = true;
                    }
                } else {
                    assert(false);
                }
            } else {
                if (!(c.banks & 1)) {
                    active[c.lhs][threadIdx.x] = true;
                }
                if (c.opcode >= OP_ADD && !(c.banks & 2)) {
                    active[c.rhs][threadIdx.x] = true;
                }
            }

            // Allocate a new subtape and begin writing to it
            if (s == 0) {
                auto next_subtape_index = subtapes.claim();
                subtapes.start[subtape_index] = 0;
                subtapes.next[next_subtape_index] = subtape_index;

                subtape_index = next_subtape_index;
                s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                out = subtapes.data[subtape_index];
            }
            out[--s] = c;
        }
    }

    // The last subtape may not be completely filled
    subtapes.start[subtape_index] = s;
    tiles.head(tile) = subtape_index;
    tiles.terminal[tile] = terminal;
}

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
__global__ void TileRenderer_check(TileRenderer<TILE_SIZE_PX, DIMENSION>* r,
                                   const uint32_t offset, View v)
{
    // This should be a 1D kernel
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t tile = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (tile < r->tiles.total) {
        if (!r->tiles.isMasked(tile)) {
            r->check(tile, v);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <unsigned TILE_SIZE_PX, unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
SubtileRenderer<TILE_SIZE_PX, SUBTILE_SIZE_PX, DIMENSION>::SubtileRenderer(
        const Tape& tape, Subtapes& subtapes, Image& image,
        Tiles<TILE_SIZE_PX, DIMENSION>& prev)
    : tape(tape), subtapes(subtapes), image(image), tiles(prev),
      subtiles(image.size_px),

      regs(CUDA_MALLOC(Registers,
        LIBFIVE_CUDA_SUBTILE_BLOCKS * tape.num_regs)),

      active(CUDA_MALLOC(ActiveArray,
                  LIBFIVE_CUDA_SUBTILE_BLOCKS * tape.num_regs)),
      choices(tape.num_csg_choices ?
              CUDA_MALLOC(ChoiceArray,
                  LIBFIVE_CUDA_SUBTILE_BLOCKS *
                  ((tape.num_csg_choices + 31) / 32))
              : nullptr)
{
    // Nothing to do here
}

template <unsigned TILE_SIZE_PX, unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
SubtileRenderer<TILE_SIZE_PX, SUBTILE_SIZE_PX, DIMENSION>::~SubtileRenderer()
{
    CUDA_CHECK(cudaFree(regs));
    CUDA_CHECK(cudaFree(active));
    CUDA_CHECK(cudaFree(choices));
}

template <unsigned TILE_SIZE_PX, unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__device__
void SubtileRenderer<TILE_SIZE_PX, SUBTILE_SIZE_PX, DIMENSION>::check(
        const uint32_t subtile, const uint32_t tile, const View& v)
{
    auto regs = this->regs + tape.num_regs * blockIdx.x;
    storeAxes(subtile, v, subtiles, tape, regs);

    auto choices = this->choices + ((tape.num_csg_choices + 31) / 32) * blockIdx.x;
    uint64_t choice = 0;
    uint32_t choice_index = 0;

    // Run actual evaluation
    uint32_t subtape_index = tiles.head(tile);
    uint32_t s = subtapes.start[subtape_index];
    const Clause* __restrict__ tape = subtapes.data[subtape_index];
    const float* __restrict__ constant_ptr = &this->tape.constant(0);

    Interval result;
    while (true) {
        using namespace libfive::Opcode;

        if (s == LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE) {
            uint32_t next = subtapes.next[subtape_index];
            if (next) {
                subtape_index = next;
                s = subtapes.start[subtape_index];
                tape = subtapes.data[subtape_index];
            } else {
                result = regs[tape[s - 1].out][threadIdx.x];
                break;
            }
        }
        const Clause c = tape[s++];

        Interval out;
        switch (c.banks) {
            case 0: // Interval op Interval
                out = intervalOp<Interval, Interval>(c.opcode,
                        regs[c.lhs][threadIdx.x],
                        regs[c.rhs][threadIdx.x], choice, choice_index);
                break;
            case 1: // Constant op Interval
                out = intervalOp<float, Interval>(c.opcode,
                        constant_ptr[c.lhs],
                        regs[c.rhs][threadIdx.x], choice, choice_index);
                break;
            case 2: // Interval op Constant
                out = intervalOp<Interval, float>(c.opcode,
                         regs[c.lhs][threadIdx.x],
                         constant_ptr[c.rhs], choice, choice_index);
                break;
            case 3: // Constant op Constant
                out = intervalOp<float, float>(c.opcode,
                        constant_ptr[c.lhs],
                        constant_ptr[c.rhs], choice, choice_index);
                break;
        }
        if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            choice_index += 2;
            if (choice_index == sizeof(choice) * 8) {
                (*(choices++))[threadIdx.x] = choice;
                choice = 0;
                choice_index = 0;
            }
        }

        regs[c.out][threadIdx.x] = out;
    }

    ////////////////////////////////////////////////////////////////////////////

    // Reverse the tape if it isn't terminal
    bool terminal = tiles.terminal[tile];
    __syncthreads();
    if ((threadIdx.x % subtilesPerTile()) == 0 && !terminal) {
        uint32_t subtape_index = tiles.head(tile);
        uint32_t prev = 0;

        while (true) {
            const uint32_t next = subtapes.next[subtape_index];
            subtapes.next[subtape_index] = prev;
            if (next == 0) {
                break;
            } else {
                prev = subtape_index;
                subtape_index = next;
            }
        }
        tiles.head(tile) = subtape_index;
    }
    __syncthreads();

    // If this tile is unambiguously filled, then mark it at the end
    // of the tiles list
    if (result.upper() < 0.0f) {
        subtiles.insertFilled(subtile);
        return;
    }

    // If the tile is ambiguous, then record it as needing further refinement
    else if ((result.lower() <= 0.0f && result.upper() >= 0.0f)
            || isnan(result.lower())
            || isnan(result.upper()))
    {
        subtiles.insertActive(subtile);
    }

    else {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////

    // Re-use the previous tape and return immediately if the previous
    // tape was terminal (i.e. having no min/max clauses to specialize)
    if (terminal) {
        subtiles.head(subtile) = tiles.head(tile);
        subtiles.terminal[subtile] = true;
        return;
    }

    // Pick a subset of the active array to use for this block
    auto active = this->active + blockIdx.x * this->tape.num_regs;

    for (uint32_t r=0; r < this->tape.num_regs; ++r) {
        active[r][threadIdx.x] = false;
    }

    // The tape chunks must be reversed by this point!
    uint32_t in_subtape_index = tiles.head(tile);
    uint32_t in_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
    uint32_t in_s_end = subtapes.start[in_subtape_index];
    const Clause* __restrict__ in_tape = subtapes.data[in_subtape_index];

    // Mark the head of the tape as active
    active[in_tape[in_s - 1].out][threadIdx.x] = true;

    // Claim a subtape to populate
    uint32_t out_subtape_index = subtapes.claim();
    assert(out_subtape_index < LIBFIVE_CUDA_NUM_SUBTAPES);
    uint32_t out_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
    Clause* __restrict__ out_tape = subtapes.data[out_subtape_index];

    // Since we're reversing the tape, this is going to be the
    // end of the linked list (i.e. next = 0)
    subtapes.next[out_subtape_index] = 0;

    terminal = true;
    while (true) {
        using namespace libfive::Opcode;

        // If we've reached the end of an input tape chunk, then
        // either move on to the next one or escape the loop
        if (in_s == in_s_end) {
            const uint32_t next = subtapes.next[in_subtape_index];
            if (next) {
                in_subtape_index = next;
                in_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                in_s_end = subtapes.start[in_subtape_index];
                in_tape = subtapes.data[in_subtape_index];
            } else {
                break;
            }
        }
        Clause c = in_tape[--in_s];

        if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            if (choice_index == 0) {
                choice_index = sizeof(choice) * 8;
                choice = (*(--choices))[threadIdx.x];
            }
            choice_index -= 2;
        }

        if (active[c.out][threadIdx.x]) {
            active[c.out][threadIdx.x] = false;
            if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
                const uint8_t choice_ = (choice >> choice_index) & 3;
                if (choice_ == 1) {
                    if (!(c.banks & 1)) {
                        active[c.lhs][threadIdx.x] = true;
                        if (c.lhs == c.out) {
                            continue;
                        }
                        c.rhs = c.lhs;
                        c.banks = 0;
                    } else {
                        c.rhs = c.lhs;
                        c.banks = 3;
                    }
                } else if (choice_ == 2) {
                    if (!(c.banks & 2)) {
                        active[c.rhs][threadIdx.x] = true;
                        if (c.rhs == c.out) {
                            continue;
                        }
                        c.lhs = c.rhs;
                        c.banks = 0;
                    } else {
                        c.lhs = c.rhs;
                        c.banks = 3;
                    }
                } else if (choice_ == 0) {
                    if (!(c.banks & 1)) {
                        active[c.lhs][threadIdx.x] = true;
                    }
                    if (!(c.banks & 2)) {
                        active[c.rhs][threadIdx.x] = true;
                    }
                } else {
                    assert(false);
                }
            } else {
                terminal = false;
                if (!(c.banks & 1)) {
                    active[c.lhs][threadIdx.x] = true;
                }
                if (c.opcode >= OP_ADD && !(c.banks & 2)) {
                    active[c.rhs][threadIdx.x] = true;
                }
            }

            // If we've reached the end of the output tape, then
            // allocate a new one and keep going
            if (out_s == 0) {
                const auto next = subtapes.claim();
                subtapes.start[out_subtape_index] = 0;
                subtapes.next[next] = out_subtape_index;

                out_subtape_index = next;
                out_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                out_tape = subtapes.data[out_subtape_index];
            }

            out_tape[--out_s] = c;
        }
    }

    // The last subtape may not be completely filled, so write its size here
    subtapes.start[out_subtape_index] = out_s;
    subtiles.head(subtile) = out_subtape_index;
    subtiles.terminal[subtile] = terminal;
}

template <unsigned TILE_SIZE_PX, unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__global__
void SubtileRenderer_check(
        SubtileRenderer<TILE_SIZE_PX, SUBTILE_SIZE_PX, DIMENSION>* r,
        const uint32_t offset, View v)
{
    assert(blockDim.x % r->subtilesPerTile() == 0);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    // Pick an active tile from the list.  Each block executes multiple tiles!
    const uint32_t stride = blockDim.x / r->subtilesPerTile();
    const uint32_t sub = threadIdx.x / r->subtilesPerTile();
    const uint32_t i = offset + blockIdx.x * stride + sub;

    if (i < r->tiles.num_active) {
        // Pick out the next active tile
        // (this will be the same for every thread in a block)
        const uint32_t tile = r->tiles.active(i);

        // Convert from tile position to pixels
        const uint3 p = r->tiles.lowerCornerVoxel(tile);

        // Calculate the subtile's offset within the tile
        const uint32_t q = threadIdx.x % r->subtilesPerTile();
        const uint3 d = make_uint3(
             q % r->subtilesPerTileSide(),
             (q / r->subtilesPerTileSide()) % r->subtilesPerTileSide(),
             (q / r->subtilesPerTileSide()) / r->subtilesPerTileSide());

        const uint32_t tx = p.x / SUBTILE_SIZE_PX + d.x;
        const uint32_t ty = p.y / SUBTILE_SIZE_PX + d.y;
        const uint32_t tz = p.z / SUBTILE_SIZE_PX + d.z;
        if (DIMENSION == 2) {
            assert(tz == 0);
        }

        // Finally, unconvert back into a single index
        const uint32_t subtile = tx + ty * r->subtiles.per_side
             + tz * r->subtiles.per_side * r->subtiles.per_side;

        if (!r->tiles.isMasked(tile) && !r->subtiles.isMasked(subtile)) {
            r->check(subtile, tile, v);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
PixelRenderer<SUBTILE_SIZE_PX, DIMENSION>::PixelRenderer(
        const Tape& tape, const Subtapes& subtapes, Image& image,
        const Tiles<SUBTILE_SIZE_PX, DIMENSION>& prev)
    : tape(tape), subtapes(subtapes), image(image), subtiles(prev),
      regs(CUDA_MALLOC(FloatRegisters,
                       tape.num_regs * LIBFIVE_CUDA_RENDER_BLOCKS))
{
    // Nothing to do here
}

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
PixelRenderer<SUBTILE_SIZE_PX, DIMENSION>::~PixelRenderer()
{
    CUDA_CHECK(cudaFree(regs));
}

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__device__ void PixelRenderer<SUBTILE_SIZE_PX, DIMENSION>::draw(
        const uint32_t subtile, const View& v)
{
    const uint32_t pixel = threadIdx.x % pixelsPerSubtile();
    const uint3 d = make_uint3(
            pixel % SUBTILE_SIZE_PX,
            (pixel / SUBTILE_SIZE_PX) % SUBTILE_SIZE_PX,
            (pixel / SUBTILE_SIZE_PX) / SUBTILE_SIZE_PX);

    // Pick an index into the register array
    auto regs = this->regs + tape.num_regs * blockIdx.x;

    // Convert from tile position to pixels
    const uint3 p = subtiles.lowerCornerVoxel(subtile);

    // Skip this pixel if it's already below the image
    if (DIMENSION == 3 && image(p.x + d.x, p.y + d.y) >= p.z + d.z) {
        return;
    }

    {   // Prepopulate axis values
        float3 f = image.voxelPos(make_uint3(
                    p.x + d.x, p.y + d.y, p.z + d.z));
        if (tape.axes.reg[0] != UINT16_MAX) {
            regs[tape.axes.reg[0]][threadIdx.x] = f.x * v.scale - v.center[0];
        }
        if (tape.axes.reg[1] != UINT16_MAX) {
            regs[tape.axes.reg[1]][threadIdx.x] = f.y * v.scale - v.center[1];
        }
        if (tape.axes.reg[2] != UINT16_MAX) {
            regs[tape.axes.reg[2]][threadIdx.x] = (DIMENSION == 3)
                ? (f.z * v.scale)
                : v.center[2];
        }
    }

    uint32_t subtape_index = subtiles.head(subtile);
    uint32_t s = subtapes.start[subtape_index];
    const float* __restrict__ constant_ptr = &tape.constant(0);
    const Clause* __restrict__ tape = subtapes.data[subtape_index];

    while (true) {
        using namespace libfive::Opcode;

        // Move to the next subtape if this one is finished
        if (s == LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE) {
            const uint32_t next = subtapes.next[subtape_index];
            if (next) {
                subtape_index = next;
                s = subtapes.start[subtape_index];
                tape = subtapes.data[subtape_index];
            } else {
                if (regs[tape[s - 1].out][threadIdx.x] < 0.0f) {
                    if (DIMENSION == 2) {
                        image(p.x + d.x, p.y + d.y) = 255;
                    } else {
                        atomicMax(&image(p.x + d.x, p.y + d.y), p.z + d.z);
                    }
                }
                return;
            }
        }
        const Clause c = tape[s++];

        // All clauses must have at least one argument, since constants
        // and VAR_X/Y/Z are handled separately.
        float lhs;
        if (c.banks & 1) {
            lhs = constant_ptr[c.lhs];
        } else {
            lhs = regs[c.lhs][threadIdx.x];
        }

        float rhs;
        if (c.banks & 2) {
            rhs = constant_ptr[c.rhs];
        } else if (c.opcode >= OP_ADD) {
            rhs = regs[c.rhs][threadIdx.x];
        }

        float out;
        switch (c.opcode) {
            case OP_SQUARE: out = lhs * lhs; break;
            case OP_SQRT: out = sqrtf(lhs); break;
            case OP_NEG: out = -lhs; break;
            // Skipping transcendental functions for now

            case OP_ADD: out = lhs + rhs; break;
            case OP_MUL: out = lhs * rhs; break;
            case OP_DIV: out = lhs / rhs; break;
            case OP_MIN: out = fminf(lhs, rhs); break;
            case OP_MAX: out = fmaxf(lhs, rhs); break;
            case OP_SUB: out = lhs - rhs; break;

            // Skipping various hard functions here
            default: break;
        }
        regs[c.out][threadIdx.x] = out;
    }
}

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__global__ void PixelRenderer_draw(
        PixelRenderer<SUBTILE_SIZE_PX, DIMENSION>* r,
        const uint32_t offset, View v)
{
    // We assume one thread per pixel in a set of tiles
    assert(blockDim.x % SUBTILE_SIZE_PX == 0);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    // Pick an active tile from the list.  Each block executes multiple tiles!
    const uint32_t stride = blockDim.x / r->pixelsPerSubtile();
    const uint32_t sub = threadIdx.x / r->pixelsPerSubtile();
    const uint32_t i = offset + blockIdx.x * stride + sub;

    if (i < r->subtiles.num_active) {
        const uint32_t subtile = r->subtiles.active(i);
        if (!r->subtiles.isMasked(subtile)) {
            r->draw(subtile, v);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#if LIBFIVE_CUDA_3D
NormalRenderer::NormalRenderer(const Tape& tape, const Renderable& parent,
                               Image& norm)
    : tape(tape), parent(parent), norm(norm),
      regs(CUDA_MALLOC(DerivRegisters,
                       tape.num_regs * LIBFIVE_CUDA_RENDER_BLOCKS))
{
    // Nothing to do here
}

NormalRenderer::~NormalRenderer()
{
    CUDA_CHECK(cudaFree(regs));
}

__device__ void NormalRenderer::draw(const uint2 pixel, const View& v)
{
    const uint32_t pz = parent.heightAt(pixel.x, pixel.y);
    if (!pz) {
        return;
    }

    const uint3 p = make_uint3(pixel.x, pixel.y, pz + 1);
    const float3 f = norm.voxelPos(p);

    auto regs = this->regs + tape.num_regs * blockIdx.x;

    {   // Prepopulate axis values
        if (tape.axes.reg[0] != UINT16_MAX) {
            const float x = f.x * v.scale - v.center[0];
            regs[tape.axes.reg[0]][threadIdx.x] = Deriv(x, 1.0f, 0.0f, 0.0f);
        }
        if (tape.axes.reg[1] != UINT16_MAX) {
            const float y = f.y * v.scale - v.center[1];
            regs[tape.axes.reg[1]][threadIdx.x] = Deriv(y, 0.0f, 1.0f, 0.0f);
        }
        if (tape.axes.reg[2] != UINT16_MAX) {
            const float z = (f.z * v.scale);
            regs[tape.axes.reg[2]][threadIdx.x] = Deriv(z, 0.0f, 0.0f, 1.0f);
        }
    }

    const Clause* __restrict__ clause_ptr = &tape[0];
    const float* __restrict__ constant_ptr = &tape.constant(0);
    const auto num_clauses = tape.num_clauses;

    for (uint32_t i=0; i < num_clauses; ++i) {
        using namespace libfive::Opcode;
        const Clause c = clause_ptr[i];
        Deriv out;
        switch (c.banks) {
            case 0: // Deriv op Deriv
                out = derivOp<Deriv, Deriv>(c.opcode,
                        regs[c.lhs][threadIdx.x],
                        regs[c.rhs][threadIdx.x]);
                break;
            case 1: // Constant op Deriv
                out = derivOp<float, Deriv>(c.opcode,
                        constant_ptr[c.lhs],
                        regs[c.rhs][threadIdx.x]);
                break;
            case 2: // Deriv op Constant
                out = derivOp<Deriv, float>(c.opcode,
                        regs[c.lhs][threadIdx.x],
                        constant_ptr[c.rhs]);
                break;
            case 3: // Constant op Constant
                out = derivOp<float, float>(c.opcode,
                        constant_ptr[c.lhs],
                        constant_ptr[c.rhs]);
                break;
        }
        regs[c.out][threadIdx.x] = out;
    }

    const Clause c = clause_ptr[num_clauses - 1];
    const Deriv result = regs[c.out][threadIdx.x];
    float norm = sqrtf(powf(result.dx(), 2) +
                       powf(result.dy(), 2) +
                       powf(result.dz(), 2));
    uint8_t dx = (result.dx() / norm) * 127 + 128;
    uint8_t dy = (result.dy() / norm) * 127 + 128;
    uint8_t dz = (result.dz() / norm) * 127 + 128;
    this->norm(p.x, p.y) = (0xFF << 24) | (dz << 16) | (dy << 8) | dx;
}

__global__ void NormalRenderer_draw(
        NormalRenderer* r, const uint32_t offset, View v)
{
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t pixel = threadIdx.x % (16 * 16);

    const uint32_t i = offset + (threadIdx.x + blockIdx.x * blockDim.x) /
                                (16 * 16);
    const uint32_t px = (i % (r->norm.size_px / 16)) * 16 +
                        (pixel % 16);
    const uint32_t py = (i / (r->norm.size_px / 16)) * 16 +
                        (pixel / 16);
    if (px < r->norm.size_px && py < r->norm.size_px) {
        r->draw(make_uint2(px, py), v);
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
__host__ __device__
uint32_t Renderable::heightAt(const uint32_t px, const uint32_t py) const
{
    const uint32_t c = image(px, py);
    const uint32_t t = tile_renderer.tiles.filledAt(px, py);
    const uint32_t s = subtile_renderer.subtiles.filledAt(px, py);

    // This is the same as the subtile renderer in 2D, but that's okay
    const uint32_t u = pixel_renderer.subtiles.filledAt(px, py);

    if (pixel_renderer.is3D()) {
        return max(max(c, t), max(s, u));
    } else {
        return (c || t || s) ? 65535 : 0;
    }
}

__device__
void Renderable::copyToSurface(bool append, cudaSurfaceObject_t surf)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    const unsigned size = image.size_px;
    if (x < size && y < size) {
        const auto h = heightAt(x, size - y - 1);
        if (h) {
#if LIBFIVE_CUDA_3D
            if (has_normals) {
                surf2Dwrite(norm(x, size - y - 1), surf, x*4, y);
            } else
#endif
            {
                surf2Dwrite(0x00FFFFFF | (h << 24), surf, x*4, y);
            }
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__global__
void Renderable_copyToSurface(Renderable* r, bool append,
                              cudaSurfaceObject_t surf)
{
    r->copyToSurface(append, surf);
}

////////////////////////////////////////////////////////////////////////////////

void Renderable::Deleter::operator()(Renderable* r)
{
    for (unsigned i=0; i < LIBFIVE_CUDA_NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(r->streams[i]));
    }
    r->~Renderable();
    CUDA_CHECK(cudaFree(r));
}

Renderable::Handle Renderable::build(libfive::Tree tree, uint32_t image_size_px)
{
    auto out = CUDA_MALLOC(Renderable, 1);
    new (out) Renderable(tree, image_size_px);
    cudaDeviceSynchronize();
    return Handle(out);
}

Renderable::Renderable(libfive::Tree tree, uint32_t image_size_px)
    : image(image_size_px), norm(image_size_px),
      tape(std::move(Tape::build(tree))),

      tile_renderer(tape, subtapes, image),
      subtile_renderer(tape, subtapes, image, tile_renderer.tiles),
#if LIBFIVE_CUDA_3D
      microtile_renderer(tape, subtapes, image, subtile_renderer.subtiles),
      pixel_renderer(tape, subtapes, image, microtile_renderer.subtiles),
      normal_renderer(tape, *this, norm)
#else
      pixel_renderer(tape, subtapes, image, subtile_renderer.subtiles)
#endif
{
    for (unsigned i=0; i < LIBFIVE_CUDA_NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
}

void Renderable::run(const View& view)
{
    // Reset everything in preparation for a render
    tile_renderer.tiles.reset();
    subtile_renderer.subtiles.reset();
    subtapes.reset();
    image.reset();
    norm.reset();
#if LIBFIVE_CUDA_3D
    microtile_renderer.subtiles.reset();
    has_normals = false;
#endif

    // Record this local variable because otherwise it looks up memory
    // that has been loaned to the GPU and not synchronized.
    auto tile_renderer = &this->tile_renderer;
    auto subtile_renderer = &this->subtile_renderer;
    auto pixel_renderer = &this->pixel_renderer;
#if LIBFIVE_CUDA_3D
    auto microtile_renderer = &this->microtile_renderer;
    auto normal_renderer = &this->normal_renderer;
#endif

    cudaStream_t streams[LIBFIVE_CUDA_NUM_STREAMS];
    for (unsigned i=0; i < LIBFIVE_CUDA_NUM_STREAMS; ++i) {
        streams[i] = this->streams[i];
    }

    {   // Do per-tile evaluation to get filled / ambiguous tiles
        const uint32_t stride = LIBFIVE_CUDA_TILE_THREADS *
                                LIBFIVE_CUDA_TILE_BLOCKS;
        const uint32_t total_tiles = tile_renderer->tiles.total;
        for (unsigned i=0; i < total_tiles; i += stride) {
            TileRenderer_check<<<LIBFIVE_CUDA_TILE_BLOCKS,
                                 LIBFIVE_CUDA_TILE_THREADS, 0,
                                 streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                                         tile_renderer, i, view);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    {   // Refine ambiguous tiles from their subtapes
        const uint32_t active = tile_renderer->tiles.num_active;
        const uint32_t stride = LIBFIVE_CUDA_SUBTILE_BLOCKS *
                                LIBFIVE_CUDA_REFINE_TILES;
        for (unsigned i=0; i < active; i += stride) {
            SubtileRenderer_check<<<LIBFIVE_CUDA_SUBTILE_BLOCKS,
                    subtile_renderer->subtilesPerTile() *
                    LIBFIVE_CUDA_REFINE_TILES, 0,
                    streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                        subtile_renderer, i, view);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

#if LIBFIVE_CUDA_3D
    {   // Refine ambiguous tiles from their subtapes
        const uint32_t active = subtile_renderer->subtiles.num_active;
        const uint32_t stride = LIBFIVE_CUDA_SUBTILE_BLOCKS *
                                LIBFIVE_CUDA_REFINE_TILES;
        for (unsigned i=0; i < active; i += stride) {
            SubtileRenderer_check<<<LIBFIVE_CUDA_SUBTILE_BLOCKS,
                    microtile_renderer->subtilesPerTile() *
                    LIBFIVE_CUDA_REFINE_TILES, 0,
                    streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                        microtile_renderer, i, view);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif

    {   // Do pixel-by-pixel rendering for active subtiles
        const uint32_t active = pixel_renderer->subtiles.num_active;
        const uint32_t stride = LIBFIVE_CUDA_RENDER_BLOCKS *
                                LIBFIVE_CUDA_RENDER_SUBTILES;
        for (unsigned i=0; i < active; i += stride) {
            PixelRenderer_draw<<<LIBFIVE_CUDA_RENDER_BLOCKS,
                                 pixel_renderer->pixelsPerSubtile() *
                                 LIBFIVE_CUDA_RENDER_SUBTILES, 0,
                                 streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                pixel_renderer, i, view);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

#if LIBFIVE_CUDA_3D && 0
    {   // Do pixel-by-pixel rendering for normals
        const uint32_t active = pow(image.size_px / 16, 2);
        const uint32_t stride = LIBFIVE_CUDA_NORMAL_BLOCKS *
                                LIBFIVE_CUDA_NORMAL_TILES;
        for (unsigned i=0; i < active; i += stride) {
            NormalRenderer_draw<<<
                LIBFIVE_CUDA_NORMAL_BLOCKS,
                pow(16, 2) * LIBFIVE_CUDA_NORMAL_TILES,
                0, streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    normal_renderer, i, view);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    has_normals = true;
#endif
}

cudaGraphicsResource* Renderable::registerTexture(GLuint t)
{
    cudaGraphicsResource* gl_tex;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&gl_tex, t, GL_TEXTURE_2D,
                                      cudaGraphicsMapFlagsWriteDiscard));
    return gl_tex;
}

void Renderable::copyToTexture(cudaGraphicsResource* gl_tex, bool append)
{
    cudaArray* array;
    CUDA_CHECK(cudaGraphicsMapResources(1, &gl_tex));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, gl_tex, 0, 0));

    // Specify texture
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    // Surface object??!
    cudaSurfaceObject_t surf = 0;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &res_desc));

    CUDA_CHECK(cudaDeviceSynchronize());
    Renderable_copyToSurface<<<dim3(256, 256), dim3(16, 16)>>>(
            this, append, surf);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDestroySurfaceObject(surf));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &gl_tex));
}
