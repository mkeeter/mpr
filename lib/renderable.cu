#include <cassert>
#include "renderable.hpp"

////////////////////////////////////////////////////////////////////////////////

template <typename R>
__device__ void storeAxes(const uint32_t index, const uint32_t tile,
                          const View& v, const Tiles& tiles, const Tape& tape,
                          R* const __restrict__ lower,
                          R* const __restrict__ upper)
{
   // Prepopulate axis values
    const float x = tile / tiles.per_side;
    const float y = tile % tiles.per_side;

    Interval vs[3];
    const Interval X = {x / tiles.per_side, (x + 1) / tiles.per_side};
    vs[0].lower = 2.0f * (X.lower - 0.5f) * v.scale - v.center[0];
    vs[0].upper = 2.0f * (X.upper - 0.5f) * v.scale - v.center[0];

    const Interval Y = {y / tiles.per_side, (y + 1) / tiles.per_side};
    vs[1].lower = 2.0f * (Y.lower - 0.5f) * v.scale - v.center[1];
    vs[1].upper = 2.0f * (Y.upper - 0.5f) * v.scale - v.center[1];

    vs[2].lower = 0.0f;
    vs[2].upper = 0.0f;

    for (unsigned i=0; i < 3; ++i) {
        if (tape.axes.reg[i] != UINT16_MAX) {
            lower[tape.axes.reg[i]][index] = vs[i].lower;
            upper[tape.axes.reg[i]][index] = vs[i].upper;
        }
    }
}

template <typename A, typename B, typename C>
__device__ inline Interval intervalOp(uint8_t op, A lhs, B rhs, C*& choices)
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
                         (*choices++)[threadIdx.x] = 1;
                         return lhs;
                     } else if (upper(rhs) < lower(lhs)) {
                         (*choices++)[threadIdx.x] = 2;
                         return rhs;
                     } else {
                         (*choices++)[threadIdx.x] = 0;
                         return min(lhs, rhs);
                     }
        case OP_MAX: if (lower(lhs) > upper(rhs)) {
                         (*choices++)[threadIdx.x] = 1;
                         return lhs;
                     } else if (lower(rhs) > upper(lhs)) {
                         (*choices++)[threadIdx.x] = 2;
                         return rhs;
                     } else {
                         (*choices++)[threadIdx.x] = 0;
                         return max(lhs, rhs);
                     }
        case OP_SUB: return lhs - rhs;

        // Skipping various hard functions here
        default: break;
    }
    return {0.0f, 0.0f};
}

////////////////////////////////////////////////////////////////////////////////

TileRenderer::TileRenderer(const Tape& tape, Image& image)
    : tape(tape), image(image),
      tiles(image.size_px, LIBFIVE_CUDA_TILE_SIZE_PX),

      regs_lower(CUDA_MALLOC(Registers, LIBFIVE_CUDA_TILE_BLOCKS *
                                        tape.num_regs * 2)),
      regs_upper(regs_lower + LIBFIVE_CUDA_TILE_BLOCKS * tape.num_regs),
      active(CUDA_MALLOC(ActiveArray, LIBFIVE_CUDA_TILE_BLOCKS *
                                      tape.num_regs)),
      choices(tape.num_csg_choices ?
              CUDA_MALLOC(ChoiceArray,
                LIBFIVE_CUDA_TILE_BLOCKS * tape.num_csg_choices)
              : nullptr)
{
    // Nothing to do here
}

TileRenderer::~TileRenderer()
{
    CHECK(cudaFree(regs_lower));
    CHECK(cudaFree(active));
    CHECK(cudaFree(choices));
}

__device__
void TileRenderer::check(const uint32_t tile, const View& v)
{
    auto regs_lower = this->regs_lower + tape.num_regs * blockIdx.x;
    auto regs_upper = this->regs_upper + tape.num_regs * blockIdx.x;
    storeAxes(threadIdx.x, tile, v, tiles, tape, regs_lower, regs_upper);

    // Unpack a 1D offset into the data arrays
    auto choices = this->choices + tape.num_csg_choices * blockIdx.x;

    const Clause* __restrict__ clause_ptr = &tape[0];
    const float* __restrict__ constant_ptr = &tape.constant(0);
    const auto num_clauses = tape.num_clauses;

    // We copy a chunk of the tape from constant to shared memory
    constexpr unsigned SHARED_CLAUSE_SIZE = LIBFIVE_CUDA_TILE_THREADS;
    __shared__ Clause clauses[SHARED_CLAUSE_SIZE];
    __shared__ float constant_lhs[SHARED_CLAUSE_SIZE];
    __shared__ float constant_rhs[SHARED_CLAUSE_SIZE];

    for (uint32_t i=0; i < num_clauses; ++i) {
        using namespace libfive::Opcode;

        if ((i % SHARED_CLAUSE_SIZE) == 0) {
            __syncthreads();
            if (i + threadIdx.x < num_clauses) {
                const Clause c = clause_ptr[i + threadIdx.x];
                if (c.banks & 1) {
                    constant_lhs[threadIdx.x] = constant_ptr[c.lhs];
                }
                if (c.banks & 2) {
                    constant_rhs[threadIdx.x] = constant_ptr[c.rhs];
                }
                clauses[threadIdx.x] = c;
            }
            __syncthreads();
        }

        // Skip unused tiles
        if (tile == UINT32_MAX) {
            continue;
        }

        const Clause c = clauses[i % SHARED_CLAUSE_SIZE];
        // All clauses must have at least one argument, since constants
        // and VAR_X/Y/Z are handled separately.
        Interval lhs;
        if (c.banks & 1) {
            const float f = constant_lhs[i % SHARED_CLAUSE_SIZE];
            lhs.lower = f;
            lhs.upper = f;
        } else {
            lhs.lower = regs_lower[c.lhs][threadIdx.x];
            lhs.upper = regs_upper[c.lhs][threadIdx.x];
        }

        Interval rhs;
        if (c.banks & 2) {
            const float f = constant_rhs[i % SHARED_CLAUSE_SIZE];
            rhs.lower = f;
            rhs.upper = f;
        } else if (c.opcode >= OP_ADD) {
            rhs.lower = regs_lower[c.rhs][threadIdx.x];
            rhs.upper = regs_upper[c.rhs][threadIdx.x];
        }

        Interval out = intervalOp(c.opcode, lhs, rhs, choices);

        regs_lower[c.out][threadIdx.x] = out.lower;
        regs_upper[c.out][threadIdx.x] = out.upper;
    }

    uint32_t build_tape_tile = UINT32_MAX;
    if (tile != UINT32_MAX) {
        // Copy output to standard register before exiting
        const Clause c = clause_ptr[num_clauses - 1];
        const Interval result = {regs_lower[c.out][threadIdx.x],
                                 regs_upper[c.out][threadIdx.x]};

        // If this tile is unambiguously filled, then mark it at the end
        // of the tiles list
        if (result.upper < 0.0f) {
            tiles.insert_filled(tile);
        }

        // If the tile is ambiguous, then record it as needing further refinement
        else if ((result.lower <= 0.0f && result.upper >= 0.0f)
                || isnan(result.lower)
                || isnan(result.upper))
        {
            tiles.insert_active(tile);
            build_tape_tile = tile;
        }
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
    if (build_tape_tile != UINT32_MAX) {
        subtape_index = atomicAdd(&tiles.num_subtapes, 1);
        assert(subtape_index < LIBFIVE_CUDA_NUM_SUBTAPES);

        // Since we're reversing the tape, this is going to be the
        // end of the linked list (i.e. next = 0)
        tiles.subtapes.next[subtape_index] = 0;
    }

    // Walk from the root of the tape downwards
    Clause* __restrict__ out = tiles.subtapes.data[subtape_index];

    for (uint32_t i=0; i < num_clauses; i++) {
        using namespace libfive::Opcode;

        if ((i % SHARED_CLAUSE_SIZE) == 0) {
            __syncthreads();
            const uint32_t j = num_clauses - i - 1 - threadIdx.x;
            if (j < num_clauses) {
                clauses[SHARED_CLAUSE_SIZE - threadIdx.x - 1] = clause_ptr[j];
            }
            __syncthreads();
        }

        // Skip dummy tiles which don't actually do things
        if (build_tape_tile == UINT32_MAX) {
            continue;
        }
        Clause c = clauses[SHARED_CLAUSE_SIZE - (i % SHARED_CLAUSE_SIZE) - 1];

        if (active[c.out][threadIdx.x]) {
            active[c.out][threadIdx.x] = false;
            if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
                const uint8_t choice = (*(--choices))[threadIdx.x];
                if (choice == 1) {
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
                } else if (choice == 2) {
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
                } else if (choice == 0) {
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
                auto next_subtape_index = atomicAdd(&tiles.num_subtapes, 1);
                tiles.subtapes.start[subtape_index] = 0;
                tiles.subtapes.next[next_subtape_index] = subtape_index;

                subtape_index = next_subtape_index;
                s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                out = tiles.subtapes.data[subtape_index];
            }
            out[--s] = c;
        } else if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            --choices;
        }
    }

    if (build_tape_tile != UINT32_MAX) {
        // The last subtape may not be completely filled
        tiles.subtapes.start[subtape_index] = s;
        tiles.head(build_tape_tile) = subtape_index;
    }
}

__global__ void TileRenderer_check(TileRenderer* r,
                                   const uint32_t offset,
                                   View v)
{
    // This should be a 1D kernel
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t tile = threadIdx.x + blockIdx.x * blockDim.x + offset;
    r->check(tile < r->tiles.total ? tile : UINT32_MAX, v);
}

__device__ void TileRenderer::drawFilled(const uint32_t tile)
{
    static_assert(LIBFIVE_CUDA_TILE_SIZE_PX >= 16, "Tiles are too small");
    static_assert(LIBFIVE_CUDA_TILE_SIZE_PX % 16 == 0, "Invalid tile size");

    // Convert from tile position to pixels
    const uint32_t px = (tile / tiles.per_side) * LIBFIVE_CUDA_TILE_SIZE_PX;
    const uint32_t py = (tile % tiles.per_side) * LIBFIVE_CUDA_TILE_SIZE_PX;

    uint4* pix = reinterpret_cast<uint4*>(&image[px + py * image.size_px]);
    const uint4 fill = make_uint4(0xB0B0B0B0, 0xB0B0B0B0, 0xB0B0B0B0, 0xB0B0B0B0);
    for (unsigned y=0; y < LIBFIVE_CUDA_TILE_SIZE_PX; y++) {
        for (unsigned x=0; x < LIBFIVE_CUDA_TILE_SIZE_PX; x += 16) {
            *pix = fill;
            pix++;
        }
        pix += (image.size_px - LIBFIVE_CUDA_TILE_SIZE_PX) / 16;
    }
}

__global__ void TileRenderer_drawFilled(TileRenderer* r, const uint32_t offset)
{
    // Each thread picks a block and fills in the whole thing
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t start = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t i = start + offset;
    if (i < r->tiles.num_filled) {
        const uint32_t tile = r->tiles.filled(i);
        r->drawFilled(tile);
    }
}

////////////////////////////////////////////////////////////////////////////////

SubtileRenderer::SubtileRenderer(const Tape& tape, Image& image,
                                 TileRenderer& prev)
    : tape(tape), image(image), tiles(prev.tiles),
      subtiles(image.size_px, LIBFIVE_CUDA_SUBTILE_SIZE_PX),

      regs_lower(CUDA_MALLOC(Registers,
        LIBFIVE_CUDA_SUBTILE_BLOCKS * tape.num_regs * 2)),
      regs_upper(regs_lower + LIBFIVE_CUDA_SUBTILE_BLOCKS * tape.num_regs),

      active(CUDA_MALLOC(ActiveArray,
                  LIBFIVE_CUDA_SUBTILE_BLOCKS * tape.num_regs)),
      choices(tape.num_csg_choices ?
              CUDA_MALLOC(ChoiceArray,
                  LIBFIVE_CUDA_SUBTILE_BLOCKS * tape.num_csg_choices)
              : nullptr)
{
    // Nothing to do here
}

SubtileRenderer::~SubtileRenderer()
{
    CHECK(cudaFree(regs_lower));
    CHECK(cudaFree(active));
    CHECK(cudaFree(choices));
}

__device__
void SubtileRenderer::check(const uint32_t subtile,
                            const uint32_t tile,
                            const View& v)
{
    auto regs_lower = this->regs_lower + tape.num_regs * blockIdx.x;
    auto regs_upper = this->regs_upper + tape.num_regs * blockIdx.x;
    storeAxes(threadIdx.x, subtile, v, subtiles, tape, regs_lower, regs_upper);

    auto choices = this->choices + tape.num_csg_choices * blockIdx.x;

    // Run actual evaluation
    uint32_t subtape_index = tiles.head(tile);
    uint32_t s = tiles.subtapes.start[subtape_index];
    const Clause* __restrict__ tape = tiles.subtapes.data[subtape_index];
    const float* __restrict__ constant_ptr = &this->tape.constant(0);

    uint32_t next = tiles.subtapes.next[subtape_index];
    uint32_t next_start = tiles.subtapes.start[next];
    uint32_t length = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;

    // We copy LIBFIVE_CUDA_SUBTILES_PER_TILE clauses from each active tape
    // into shared memory, to speed up the first pass a little bit.  Beyond
    // that point, tapes diverge in size, so we can't realiably sync threads.
    __shared__ Clause local[LIBFIVE_CUDA_REFINE_TILES]
                           [LIBFIVE_CUDA_SUBTILES_PER_TILE];
    const auto u = threadIdx.x / LIBFIVE_CUDA_SUBTILES_PER_TILE;
    const uint32_t q = threadIdx.x % LIBFIVE_CUDA_SUBTILES_PER_TILE;
    if (s + q < LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE) {
        local[u][q] = tape[s + q];
    }

    {   // If this chunk is larger than the short cached tape, then
        // we'll set the next chunk to re-enter this chunk at a
        // later point to finish it up.
        const auto chunk_length = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE - s;
        if (chunk_length > LIBFIVE_CUDA_SUBTILES_PER_TILE) {
            length = LIBFIVE_CUDA_SUBTILES_PER_TILE;
            next = subtape_index;
            next_start = s + length;
        } else {
            // Otherwise, we'll finish the entire cached subtape
            length = chunk_length;
        }
    }

    // Reassign the first tape to our chunk of shared memory
    tape = local[u];
    s = 0;
    __syncthreads();

    Interval result;
    while (true) {
        using namespace libfive::Opcode;

        if (s == length) {
            if (next) {
                subtape_index = next;
                s = next_start;
                tape = tiles.subtapes.data[subtape_index];

                // Preload these values
                next = tiles.subtapes.next[subtape_index];
                next_start = tiles.subtapes.start[next];
                length = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
            } else {
                result.lower = regs_lower[tape[s - 1].out][threadIdx.x];
                result.upper = regs_upper[tape[s - 1].out][threadIdx.x];
                break;
            }
        }
        const Clause c = tape[s++];

        // All clauses must have at least one argument, since constants
        // and VAR_X/Y/Z are handled separately.
        Interval lhs;
        if (c.banks & 1) {
            const float f = constant_ptr[c.lhs];
            lhs.lower = f;
            lhs.upper = f;
        } else {
            lhs.lower = regs_lower[c.lhs][threadIdx.x];
            lhs.upper = regs_upper[c.lhs][threadIdx.x];
        }

        Interval rhs;
        if (c.banks & 2) {
            const float f = constant_ptr[c.rhs];
            rhs.lower = f;
            rhs.upper = f;
        } else if (c.opcode >= OP_ADD) {
            rhs.lower = regs_lower[c.rhs][threadIdx.x];
            rhs.upper = regs_upper[c.rhs][threadIdx.x];
        }

        Interval out = intervalOp(c.opcode, lhs, rhs, choices);
        regs_lower[c.out][threadIdx.x] = out.lower;
        regs_upper[c.out][threadIdx.x] = out.upper;
    }

    ////////////////////////////////////////////////////////////////////////////

    // Reverse the tape
    if ((threadIdx.x % LIBFIVE_CUDA_SUBTILES_PER_TILE) == 0) {
        uint32_t subtape_index = tiles.head(tile);
        uint32_t prev = 0;

        while (true) {
            const uint32_t next = tiles.subtapes.next[subtape_index];
            tiles.subtapes.next[subtape_index] = prev;
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
    if (result.upper < 0.0f) {
        subtiles.insert_filled(subtile);
        return;
    }

    // If the tile is ambiguous, then record it as needing further refinement
    else if ((result.lower <= 0.0f && result.upper >= 0.0f)
            || isnan(result.lower)
            || isnan(result.upper))
    {
        subtiles.insert_active(subtile);
    }

    else {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////

    // Pick a subset of the active array to use for this block
    auto active = this->active + blockIdx.x * this->tape.num_regs;

    for (uint32_t r=0; r < this->tape.num_regs; ++r) {
        active[r][threadIdx.x] = false;
    }

    // The tape chunks must be reversed by this point!
    uint32_t in_subtape_index = tiles.head(tile);
    uint32_t in_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
    uint32_t in_s_end = tiles.subtapes.start[in_subtape_index];
    const Clause* __restrict__ in_tape = tiles.subtapes.data[in_subtape_index];

    // Mark the head of the tape as active
    active[in_tape[in_s - 1].out][threadIdx.x] = true;

    // Claim a subtape to populate
    uint32_t out_subtape_index = atomicAdd(&subtiles.num_subtapes, 1);
    assert(out_subtape_index < LIBFIVE_CUDA_NUM_SUBTAPES);
    uint32_t out_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
    Clause* __restrict__ out_tape = subtiles.subtapes.data[out_subtape_index];

    // Since we're reversing the tape, this is going to be the
    // end of the linked list (i.e. next = 0)
    subtiles.subtapes.next[out_subtape_index] = 0;

    while (true) {
        using namespace libfive::Opcode;

        // If we've reached the end of an input tape chunk, then
        // either move on to the next one or escape the loop
        if (in_s == in_s_end) {
            const uint32_t next = tiles.subtapes.next[in_subtape_index];
            if (next) {
                in_subtape_index = next;
                in_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                in_s_end = tiles.subtapes.start[in_subtape_index];
                in_tape = tiles.subtapes.data[in_subtape_index];
            } else {
                break;
            }
        }
        Clause c = in_tape[--in_s];

        if (active[c.out][threadIdx.x]) {
            active[c.out][threadIdx.x] = false;
            if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
                const uint8_t choice = (*(--choices))[threadIdx.x];
                if (choice == 1) {
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
                } else if (choice == 2) {
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
                } else if (choice == 0) {
                    if (!(c.banks & 1)) {
                        active[c.lhs][threadIdx.x] = true;
                    }
                    if (!(c.banks & 2)) {
                        active[c.rhs][threadIdx.x] = true;
                    }
                } else {
                    printf("Bad choice %u\n", choice);
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

            // If we've reached the end of the output tape, then
            // allocate a new one and keep going
            if (out_s == 0) {
                const auto next = atomicAdd(&subtiles.num_subtapes, 1);
                subtiles.subtapes.start[out_subtape_index] = 0;
                subtiles.subtapes.next[next] = out_subtape_index;

                out_subtape_index = next;
                out_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                out_tape = subtiles.subtapes.data[out_subtape_index];
            }

            out_tape[--out_s] = c;
        } else if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            --choices;
        }
    }

    // The last subtape may not be completely filled, so write its size here
    subtiles.subtapes.start[out_subtape_index] = out_s;
    subtiles.head(subtile) = out_subtape_index;
}

__global__
void SubtileRenderer_check(SubtileRenderer* r,
                           const uint32_t offset,
                           View v)
{
    assert(blockDim.x % LIBFIVE_CUDA_SUBTILES_PER_TILE == 0);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    // Pick an active tile from the list.  Each block executes multiple tiles!
    const uint32_t stride = blockDim.x / LIBFIVE_CUDA_SUBTILES_PER_TILE;
    const uint32_t sub = threadIdx.x / LIBFIVE_CUDA_SUBTILES_PER_TILE;
    const uint32_t i = offset + blockIdx.x * stride + sub;

    if (i < r->tiles.num_active) {
        // Pick out the next active tile
        // (this will be the same for every thread in a block)
        const uint32_t tile = r->tiles.active(i);

        // Convert from tile position to pixels
        const uint32_t px = (tile / r->tiles.per_side) *
                            LIBFIVE_CUDA_TILE_SIZE_PX;
        const uint32_t py = (tile % r->tiles.per_side) *
                            LIBFIVE_CUDA_TILE_SIZE_PX;

        // Then convert from pixels into subtiles
        const uint32_t p = threadIdx.x % LIBFIVE_CUDA_SUBTILES_PER_TILE;
        const uint32_t dx = p % LIBFIVE_CUDA_SUBTILES_PER_TILE_SIDE;
        const uint32_t dy = p / LIBFIVE_CUDA_SUBTILES_PER_TILE_SIDE;

        const uint32_t tx = px / LIBFIVE_CUDA_SUBTILE_SIZE_PX + dx;
        const uint32_t ty = py / LIBFIVE_CUDA_SUBTILE_SIZE_PX + dy;

        // Finally, unconvert back into a single index
        const uint32_t subtile = ty + tx * r->subtiles.per_side;

        r->check(subtile, tile, v);
    }
}

__device__ void SubtileRenderer::drawFilled(const uint32_t tile)
{
    static_assert(LIBFIVE_CUDA_TILE_SIZE_PX >= 8, "Tiles are too small");
    static_assert(LIBFIVE_CUDA_TILE_SIZE_PX % 8 == 0, "Invalid tile size");

    // Convert from tile position to pixels
    const uint32_t px = (tile / subtiles.per_side) * LIBFIVE_CUDA_SUBTILE_SIZE_PX;
    const uint32_t py = (tile % subtiles.per_side) * LIBFIVE_CUDA_SUBTILE_SIZE_PX;

    uint2* pix = reinterpret_cast<uint2*>(&image[px + py * image.size_px]);
    const uint2 fill = make_uint2(0xD0D0D0D0, 0xD0D0D0D0);
    for (unsigned y=0; y < LIBFIVE_CUDA_SUBTILE_SIZE_PX; y++) {
        for (unsigned x=0; x < LIBFIVE_CUDA_SUBTILE_SIZE_PX; x += 8) {
            *pix = fill;
            pix++;
        }
        pix += (image.size_px - LIBFIVE_CUDA_SUBTILE_SIZE_PX) / 8;
    }
}

__global__ void SubtileRenderer_drawFilled(SubtileRenderer* r, const uint32_t offset)
{
    // Each thread picks a block and fills in the whole thing
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t start = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t i = start + offset;
    if (i < r->subtiles.num_filled) {
        const uint32_t tile = r->subtiles.filled(i);
        r->drawFilled(tile);
    }
}

////////////////////////////////////////////////////////////////////////////////

PixelRenderer::PixelRenderer(const Tape& tape, Image& image,
                             const SubtileRenderer& prev)
    : tape(tape), image(image), subtiles(prev.subtiles),
      regs(CUDA_MALLOC(FloatRegisters,
                       tape.num_regs * LIBFIVE_CUDA_RENDER_BLOCKS))
{
    // Nothing to do here
}

PixelRenderer::~PixelRenderer()
{
    CHECK(cudaFree(regs));
}

__device__ void PixelRenderer::draw(const uint32_t subtile, const View& v)
{
    const uint32_t pixel = threadIdx.x % LIBFIVE_CUDA_PIXELS_PER_SUBTILE;
    const uint32_t dx = pixel % LIBFIVE_CUDA_SUBTILE_SIZE_PX;
    const uint32_t dy = pixel / LIBFIVE_CUDA_SUBTILE_SIZE_PX;

    // Pick an index into the register array
    auto regs = this->regs + tape.num_regs * blockIdx.x;

    // Convert from tile position to pixels
    uint32_t px = (subtile / subtiles.per_side) *
                   LIBFIVE_CUDA_SUBTILE_SIZE_PX + dx;
    uint32_t py = (subtile % subtiles.per_side) *
                   LIBFIVE_CUDA_SUBTILE_SIZE_PX + dy;

    {   // Prepopulate axis values
        const float x = px / (image.size_px - 1.0f);
        const float y = py / (image.size_px - 1.0f);
        float vs[3];
        vs[0] = 2.0f * (x - 0.5f) * v.scale - v.center[0];
        vs[1] = 2.0f * (y - 0.5f) * v.scale - v.center[1];
        vs[2] = 0.0f;
        for (unsigned i=0; i < 3; ++i) {
            if (tape.axes.reg[i] != UINT16_MAX) {
                regs[tape.axes.reg[i]][threadIdx.x] = vs[i];
            }
        }
    }

    uint32_t subtape_index = subtiles.head(subtile);
    uint32_t s = subtiles.subtapes.start[subtape_index];
    const float* __restrict__ constant_ptr = &tape.constant(0);
    const Clause* __restrict__ tape = subtiles.subtapes.data[subtape_index];

    while (true) {
        using namespace libfive::Opcode;

        // Move to the next subtape if this one is finished
        if (s == LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE) {
            const uint32_t next = subtiles.subtapes.next[subtape_index];
            if (next) {
                subtape_index = next;
                s = subtiles.subtapes.start[subtape_index];
                tape = subtiles.subtapes.data[subtape_index];
            } else {
                if (regs[tape[s - 1].out][threadIdx.x] < 0.0f) {
                    image(px, py) = 255;
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

__global__ void PixelRenderer_draw(PixelRenderer* r,
                                   const Tiles* subtiles,
                                   const uint32_t offset, View v)
{
    // We assume one thread per pixel in a tile
    assert(blockDim.x % LIBFIVE_CUDA_SUBTILE_SIZE_PX == 0);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    // Pick an active tile from the list.  Each block executes multiple tiles!
    const uint32_t stride = blockDim.x / LIBFIVE_CUDA_PIXELS_PER_SUBTILE;
    const uint32_t sub = threadIdx.x / LIBFIVE_CUDA_PIXELS_PER_SUBTILE;
    const uint32_t i = offset + blockIdx.x * stride + sub;

    if (i < subtiles->num_active) {
        const uint32_t subtile = subtiles->active(i);
        r->draw(subtile, v);
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__
void Renderable_copyToTexture(Renderable* r, bool append, cudaSurfaceObject_t surf)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    const unsigned size = r->image.size_px;
    if (x < size && y < size) {
        const uint8_t c = r->image(x, size - y - 1);
        if (c) {
            surf2Dwrite(0x00FFFFFF | (c << 24), surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void Renderable::Deleter::operator()(Renderable* r)
{
    r->~Renderable();
    CHECK(cudaFree(r));
}

Renderable::~Renderable()
{
    for (auto& s : streams) {
        CHECK(cudaStreamDestroy(s));
    }
}

Renderable::Handle Renderable::build(libfive::Tree tree, uint32_t image_size_px)
{
    auto out = CUDA_MALLOC(Renderable, 1);
    new (out) Renderable(tree, image_size_px);
    cudaDeviceSynchronize();
    return Handle(out);
}

Renderable::Renderable(libfive::Tree tree, uint32_t image_size_px)
    : image(image_size_px),
      tape(std::move(Tape::build(tree))),

      tile_renderer(tape, image),
      subtile_renderer(tape, image, tile_renderer),
      pixel_renderer(tape, image, subtile_renderer)
{
    CHECK(cudaStreamCreate(&streams[0]));
    CHECK(cudaStreamCreate(&streams[1]));
}

void Renderable::run(const View& view)
{
    cudaStream_t streams[2] = {this->streams[0], this->streams[1]};

    // Record this local variable because otherwise it looks up memory
    // that has been loaned to the GPU and not synchronized.
    TileRenderer* tile_renderer = &this->tile_renderer;
    const uint32_t total_tiles = tile_renderer->tiles.total;
    const uint32_t tile_stride = LIBFIVE_CUDA_TILE_THREADS *
                                 LIBFIVE_CUDA_TILE_BLOCKS;
    SubtileRenderer* subtile_renderer = &this->subtile_renderer;
    PixelRenderer* pixel_renderer = &this->pixel_renderer;
    auto tiles = &tile_renderer->tiles;
    auto subtiles = &subtile_renderer->subtiles;

    // Reset everything in preparation for a render
    tiles->reset();
    subtiles->reset();
    cudaMemset(image.data, 0, image.size_px * image.size_px);

    // Do per-tile evaluation to get filled / ambiguous tiles
    for (unsigned i=0; i < total_tiles; i += tile_stride) {
        TileRenderer_check<<<LIBFIVE_CUDA_TILE_BLOCKS,
                             LIBFIVE_CUDA_TILE_THREADS,
                             0, streams[0]>>>(tile_renderer, i, view);
    }
    cudaDeviceSynchronize();

    // Pull a few variables back from the GPU
    const uint32_t filled_tiles = tiles->num_filled;
    const uint32_t active_tiles = tiles->num_active;

    for (unsigned i=0; i < filled_tiles; i += tile_stride) {
        // Drawing filled and ambiguous tiles can happen simultaneously,
        // so we assign each one to a separate stream.
        TileRenderer_drawFilled<<<LIBFIVE_CUDA_TILE_BLOCKS,
                                  LIBFIVE_CUDA_TILE_THREADS,
                                  0, streams[1]>>>(tile_renderer, i);
    }

    // Refine ambiguous tiles from their subtapes
    const uint32_t subtile_check_stride = LIBFIVE_CUDA_SUBTILE_BLOCKS *
                                          LIBFIVE_CUDA_REFINE_TILES;
    for (unsigned i=0; i < active_tiles; i += subtile_check_stride) {
        SubtileRenderer_check<<<LIBFIVE_CUDA_SUBTILE_BLOCKS,
            LIBFIVE_CUDA_SUBTILES_PER_TILE *
            LIBFIVE_CUDA_REFINE_TILES,
            0, streams[0]>>>(
                    subtile_renderer, i, view);
    }

    cudaDeviceSynchronize();

    const uint32_t filled_subtiles = subtile_renderer->subtiles.num_filled;
    const uint32_t active_subtiles = subtile_renderer->subtiles.num_active;
    const uint32_t subtile_stride = LIBFIVE_CUDA_SUBTILE_BLOCKS *
                                    LIBFIVE_CUDA_SUBTILE_THREADS;
    for (unsigned i=0; i < filled_subtiles; i += subtile_stride) {
        SubtileRenderer_drawFilled<<<LIBFIVE_CUDA_SUBTILE_BLOCKS,
                                     LIBFIVE_CUDA_SUBTILE_THREADS,
                                     0, streams[1]>>>(
            subtile_renderer, i);
    }

    // Do pixel-by-pixel rendering for active subtiles
    const uint32_t subtile_render_stride = LIBFIVE_CUDA_RENDER_BLOCKS *
                                           LIBFIVE_CUDA_RENDER_SUBTILES;
    for (unsigned i=0; i < active_subtiles; i += subtile_render_stride) {
        PixelRenderer_draw<<<LIBFIVE_CUDA_RENDER_BLOCKS,
                             LIBFIVE_CUDA_PIXELS_PER_SUBTILE *
                             LIBFIVE_CUDA_RENDER_SUBTILES, 0, streams[0]>>>(
            pixel_renderer, subtiles, i, view);
    }
    cudaDeviceSynchronize();
}

cudaGraphicsResource* Renderable::registerTexture(GLuint t)
{
    cudaGraphicsResource* gl_tex;
    CHECK(cudaGraphicsGLRegisterImage(&gl_tex, t, GL_TEXTURE_2D,
                                      cudaGraphicsMapFlagsWriteDiscard));
    return gl_tex;
}

void Renderable::copyToTexture(cudaGraphicsResource* gl_tex, bool append)
{
    cudaArray* array;
    CHECK(cudaGraphicsMapResources(1, &gl_tex));
    CHECK(cudaGraphicsSubResourceGetMappedArray(&array, gl_tex, 0, 0));

    // Specify texture
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    // Surface object??!
    cudaSurfaceObject_t surf = 0;
    CHECK(cudaCreateSurfaceObject(&surf, &res_desc));

    CHECK(cudaDeviceSynchronize());
    Renderable_copyToTexture<<<dim3(256, 256), dim3(16, 16)>>>(
            this, append, surf);
    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaDestroySurfaceObject(surf));
    CHECK(cudaGraphicsUnmapResources(1, &gl_tex));
}
