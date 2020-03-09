#include <cassert>
#include "renderable.hpp"

#include "gpu_affine.hpp"
#include "gpu_deriv.hpp"
#include "gpu_interval.hpp"

#ifdef USE_AFFINE
#define IntervalType Affine
#else
#define IntervalType Interval
#endif

// Copy-and paste the result of benchmark/dump_tape into this block
// to test out a kernel without the overhead of the interpreter
__global__ void evalRawTape(Image* image, View v)
{
    uint32_t px = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t py = threadIdx.y + blockIdx.y * blockDim.y;

    if (px >= image->size_px && py >= image->size_px) {
        return;
    }
    assert(false);
}
////////////////////////////////////////////////////////////////////////////////

template <typename R, unsigned T, unsigned D>
__device__ void storeAxes(const uint32_t tile,
                          const View& v, const Tiles<T, D>& tiles, const Tape& tape,
                          R* rx, R* ry, R* rz)
{
   // Prepopulate axis values
    const float3 lower = tiles.tileToLowerPos(tile);
    const float3 upper = tiles.tileToUpperPos(tile);

    IntervalType X = IntervalType::X(Interval{lower.x, upper.x});
    IntervalType Y = IntervalType::Y(Interval{lower.y, upper.y});
    IntervalType Z = IntervalType::Z(Interval{lower.z, upper.z});

    IntervalType X_, Y_, Z_, W_;
    X_ = v.mat(0, 0) * X +
         v.mat(0, 1) * Y +
         v.mat(0, 2) * Z + v.mat(0, 3);
    Y_ = v.mat(1, 0) * X +
         v.mat(1, 1) * Y +
         v.mat(1, 2) * Z + v.mat(1, 3);
    if (D == 3) {
        Z_ = v.mat(2, 0) * X +
             v.mat(2, 1) * Y +
             v.mat(2, 2) * Z + v.mat(2, 3);
        W_ = v.mat(3, 0) * X +
             v.mat(3, 1) * Y +
             v.mat(3, 2) * Z + v.mat(3, 3);

        // Projection!
        X_ = X_ / W_;
        Y_ = Y_ / W_;
        Z_ = Z_ / W_;
    } else {
        Z_ = IntervalType(Interval(v.mat(2,3), v.mat(2,3)));
    }

    if (rx) {
        *rx = X_;
    }
    if (ry) {
        *ry = Y_;
    }
    if (rz) {
        *rz = Z_;
    }

}

template <typename A, typename B>
__device__ inline IntervalType intervalOp(uint8_t op, A lhs, B rhs, int& choice)
{
    using namespace libfive::Opcode;
    switch (op) {
        case OP_SQUARE: return IntervalType(square(lhs));
        case OP_SQRT: return IntervalType(sqrt(lhs));
        case OP_NEG: return IntervalType(-lhs);
        case OP_ABS: return IntervalType(abs(lhs));

        case OP_ASIN: return IntervalType(asin(lhs));
        case OP_ACOS: return IntervalType(acos(lhs));
        case OP_ATAN: return IntervalType(atan(lhs));
        case OP_EXP: return IntervalType(exp(lhs));
        case OP_SIN: return IntervalType(sin(lhs));
        case OP_COS: return IntervalType(cos(lhs));
        case OP_LOG: return IntervalType(log(lhs));
        // Skipping other transcendental functions for now

        case OP_ADD: return IntervalType(lhs + rhs);
        case OP_MUL: return IntervalType(lhs * rhs);
        case OP_DIV: return IntervalType(lhs / rhs);
        case OP_MIN: return IntervalType(min(lhs, rhs, choice));
        case OP_MAX: return IntervalType(max(lhs, rhs, choice));
        case OP_SUB: return IntervalType(lhs - rhs);

        // Skipping various hard functions here
        default: break;
    }
    return IntervalType(0.0f);
}

template <typename A, typename B>
__device__ inline Deriv derivOp(uint8_t op, A lhs, B rhs)
{
    using namespace libfive::Opcode;
    switch (op) {
        case OP_SQUARE: return lhs * lhs;
        case OP_SQRT: return sqrt(lhs);
        case OP_NEG: return -lhs;
        case OP_ABS: return abs(lhs);

        case OP_ASIN: return asin(lhs);
        case OP_ACOS: return acos(lhs);
        case OP_ATAN: return atan(lhs);
        case OP_EXP: return exp(lhs);
        case OP_SIN: return sin(lhs);
        case OP_COS: return cos(lhs);
        case OP_LOG: return log(lhs);
        // Skipping other transcendental functions for now

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
    : tape(tape), subtapes(subtapes), tiles(image.size_px)
{
    // Nothing to do here
}

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
__device__
TileResult TileRenderer<TILE_SIZE_PX, DIMENSION>::check(
        const uint32_t tile, const View& v)
{
    IntervalType regs[128];
    storeAxes(tile, v, tiles, tape,
        (tape.axes.reg[0] == UINT16_MAX) ? NULL : &regs[tape.axes.reg[0]],
        (tape.axes.reg[1] == UINT16_MAX) ? NULL : &regs[tape.axes.reg[1]],
        (tape.axes.reg[2] == UINT16_MAX) ? NULL : &regs[tape.axes.reg[2]]);

    // Unpack a 1D offset into the data arrays
    uint32_t choices[256];
    memset(choices, 0, sizeof(choices));
    uint32_t choice_index = 0;

    const Clause* __restrict__ clause_ptr = &tape[0];
    const float* __restrict__ constant_ptr = &tape.constant(0);
    const auto num_clauses = tape.num_clauses;

    for (uint32_t i=0; i < num_clauses; ++i) {
        using namespace libfive::Opcode;

        const Clause c = clause_ptr[i];
        IntervalType out;
        int choice = 0;
        switch (c.banks) {
            case 0: // Interval op Interval
                out = intervalOp<IntervalType, IntervalType>(c.opcode,
                        regs[c.lhs],
                        regs[c.rhs],
                        choice);
                break;
            case 1: // Constant op Interval
                out = intervalOp<float, IntervalType>(c.opcode,
                        constant_ptr[c.lhs],
                        regs[c.rhs],
                        choice);
                break;
            case 2: // Interval op Constant
                out = intervalOp<IntervalType, float>(c.opcode,
                        regs[c.lhs],
                        constant_ptr[c.rhs],
                        choice);
                break;
            case 3: // Constant op Constant
                out = intervalOp<float, float>(c.opcode,
                        constant_ptr[c.lhs],
                        constant_ptr[c.rhs],
                        choice);
                break;
        }

        if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            choices[choice_index / 16] |= (choice << ((choice_index % 16) * 2));
            choice_index++;
        }

        regs[c.out] = out;
    }

    const Clause c = clause_ptr[num_clauses - 1];
    const IntervalType result = regs[c.out];

    // If this tile is unambiguously filled, then mark it at the end
    // of the tiles list
    if (result.upper() < 0.0f) {
        return TILE_FILLED;
    }

    // If the tile is empty, then return immediately
    else if (result.lower() > 0.0f)
    {
        return TILE_EMPTY;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Now, we build a tape for this tile (if it's active).  If it isn't active,
    // then we use the thread to help copy stuff to shared memory, but don't
    // write any tape data out.

    // Pick a subset of the active array to use for this block
    uint8_t* __restrict__ active = reinterpret_cast<uint8_t*>(regs);
    memset(active, 0, tape.num_regs);

    // Mark the root of the tree as true
    active[tape[num_clauses - 1].out] = true;

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

        int choice = 0;
        if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            --choice_index;
            choice = (choices[choice_index / 16] >> ((choice_index % 16) * 2)) & 3;
        }

        if (active[c.out]) {
            active[c.out] = false;
            if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
                if (choice == 1) {
                    if (!(c.banks & 1)) {
                        active[c.lhs] = true;
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
                        active[c.rhs] = true;
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
                    if (c.banks == 1 || c.banks == 2 || c.lhs != c.rhs) {
                        terminal = false;
                    }
                    active[c.lhs] |= !(c.banks & 1);
                    active[c.rhs] |= !(c.banks & 2);
                } else {
                    assert(false);
                }
            } else {
                active[c.lhs] |= !(c.banks & 1);
                active[c.rhs] |= (c.opcode >= OP_ADD && !(c.banks & 2));
            }

            // Allocate a new subtape and begin writing to it
            if (s == 0) {
                auto next_subtape_index = subtapes.claim();
                subtapes.start[subtape_index] = 0;
                subtapes.next[next_subtape_index] = subtape_index;
                subtapes.prev[subtape_index] = next_subtape_index;

                subtape_index = next_subtape_index;
                s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                out = subtapes.data[subtape_index];
            }
            out[--s] = c;
        }
    }

    // The last subtape may not be completely filled
    subtapes.start[subtape_index] = s;
    subtapes.prev[subtape_index] = 0;
    tiles.setHead(tile, subtape_index, terminal);

    return TILE_AMBIGUOUS;
}

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
__global__ void TileRenderer_check(
        TileRenderer<TILE_SIZE_PX, DIMENSION>* r,
        Queue* __restrict__ active_tiles,
        Filled<TILE_SIZE_PX>* __restrict__ filled_tiles,
        const uint32_t offset, View v)
{
    // This should be a 1D kernel
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t tile = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (tile < r->tiles.size &&
        !filled_tiles->isMasked(tile))
    {
        switch (r->check(tile, v)) {
            case TILE_FILLED:
                filled_tiles->insert(tile);
                break;
            case TILE_AMBIGUOUS:
                r->tiles.index(tile) = active_tiles->insert(tile);
                break;
            case TILE_EMPTY:
                break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <unsigned TILE_SIZE_PX, unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
SubtileRenderer<TILE_SIZE_PX, SUBTILE_SIZE_PX, DIMENSION>::SubtileRenderer(
        const Tape& tape, Subtapes& subtapes, Image& image,
        Tiles<TILE_SIZE_PX, DIMENSION>& prev)
    : tape(tape), subtapes(subtapes), tiles(prev),
      subtiles(image.size_px)
{
    // Nothing to do here
}

template <unsigned TILE_SIZE_PX, unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__device__
TileResult SubtileRenderer<TILE_SIZE_PX, SUBTILE_SIZE_PX, DIMENSION>::check(
        const uint32_t subtile, const uint32_t tile, const View& v)
{
#define FAST_SLOT_COUNT (4 * LIBFIVE_CUDA_SM_SCALE)
    IntervalType slots_slow[128 - FAST_SLOT_COUNT];
#if FAST_SLOT_COUNT > 0
    // We'd like to use subtilesPerTile(), but CUDA doesn't allow for
    // (even a constexpr) function to be used when sizing a __shared__ array
    // Instead, we hard-code 64 with a static assertion to check for it.
    static_assert(subtilesPerTile() == 64, "Incorect subdivision for __shared__ array");
    __shared__ IntervalType slots_fast[FAST_SLOT_COUNT]
                                      [64*LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK];
#define SLOT(i) ((i < FAST_SLOT_COUNT) ? slots_fast[i][threadIdx.x] \
                                       : slots_slow[i - FAST_SLOT_COUNT])
#else
#define SLOT(i) slots_slow[i]
#endif

    storeAxes(subtile, v, subtiles, tape,
        (tape.axes.reg[0] == UINT16_MAX) ? NULL : &SLOT(tape.axes.reg[0]),
        (tape.axes.reg[1] == UINT16_MAX) ? NULL : &SLOT(tape.axes.reg[1]),
        (tape.axes.reg[2] == UINT16_MAX) ? NULL : &SLOT(tape.axes.reg[2]));

    uint32_t choices[256];
    memset(choices, 0, sizeof(choices));
    uint32_t choice_index = 0;

    // Run actual evaluation
    uint32_t subtape_index = tiles.head(tile);
    uint32_t s = subtapes.start[subtape_index];
    const Clause* __restrict__ tape = subtapes.data[subtape_index];
    const float* __restrict__ constant_ptr = &this->tape.constant(0);

    IntervalType result;
    bool has_any_choices = false;
    while (true) {
        using namespace libfive::Opcode;

        if (s == LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE) {
            uint32_t next = subtapes.next[subtape_index];
            if (next) {
                subtape_index = next;
                s = subtapes.start[subtape_index];
                tape = subtapes.data[subtape_index];
            } else {
                result = SLOT(tape[s - 1].out);
                break;
            }
        }
        const Clause c = tape[s++];

        IntervalType out;
        int choice = 0;
        switch (c.banks) {
            case 0: // Interval op Interval
                out = intervalOp<IntervalType, IntervalType>(c.opcode,
                        SLOT(c.lhs),
                        SLOT(c.rhs), choice);
                break;
            case 1: // Constant op Interval
                out = intervalOp<float, IntervalType>(c.opcode,
                        constant_ptr[c.lhs],
                        SLOT(c.rhs), choice);
                break;
            case 2: // Interval op Constant
                out = intervalOp<IntervalType, float>(c.opcode,
                         SLOT(c.lhs),
                         constant_ptr[c.rhs], choice);
                break;
            case 3: // Constant op Constant
                out = intervalOp<float, float>(c.opcode,
                        constant_ptr[c.lhs],
                        constant_ptr[c.rhs], choice);
                break;
        }
        if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            has_any_choices |= (choice != 0);
            choices[choice_index / 16] |= (choice << ((choice_index % 16) * 2));
            choice_index++;
        }

        SLOT(c.out) = out;
    }
#undef SLOT
#undef FAST_SLOT_COUNT

    ////////////////////////////////////////////////////////////////////////////
    // If this tile is unambiguously filled, then mark it at the end
    // of the tiles list
    if (result.upper() < 0.0f) {
        return TILE_FILLED;
    }

    // If the tile is empty, then return right away
    else if (result.lower() > 0.0f)
    {
        return TILE_EMPTY;
    }

    ////////////////////////////////////////////////////////////////////////////

    // Re-use the previous tape and return immediately if the previous
    // tape was terminal (i.e. having no min/max clauses to specialize)
    bool terminal = tiles.terminal(tile);
    if (terminal || !has_any_choices) {
        subtiles.setHead(subtile, tiles.head(tile), true);
        return TILE_AMBIGUOUS;
    }

    // Pick a subset of the active array to use for this block
    uint8_t* __restrict__ active_slow = reinterpret_cast<uint8_t*>(slots_slow);
#if FAST_SLOT_COUNT > 0
    auto* __restrict__ active_fast =
        (uint8_t(*)[64*LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK])(slots_fast);
#define ACTIVE(i) ((i < (FAST_SLOT_COUNT * 8)) \
        ? active_fast[i][threadIdx.x] \
        : active_slow[i - (FAST_SLOT_COUNT * 8)])
#else
#define ACTIVE(i) active_slow[i]
#endif
    for (unsigned i=0; i < this->tape.num_regs; ++i) {
        ACTIVE(i) = false;
    }

    // At this point, subtape_index is pointing to the last chunk, so we'll
    // use the prev pointers to walk backwards (where "backwards" means
    // from the root of the tree to its leaves).
    uint32_t in_subtape_index = subtape_index;
    uint32_t in_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
    uint32_t in_s_end = subtapes.start[in_subtape_index];
    const Clause* __restrict__ in_tape = subtapes.data[in_subtape_index];

    // Mark the head of the tape as active
    ACTIVE(in_tape[in_s - 1].out) = true;

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
            const uint32_t prev = subtapes.prev[in_subtape_index];
            if (prev) {
                in_subtape_index = prev;
                in_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                in_s_end = subtapes.start[in_subtape_index];
                in_tape = subtapes.data[in_subtape_index];
            } else {
                break;
            }
        }
        Clause c = in_tape[--in_s];

        int choice = 0;
        if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            --choice_index;
            choice = (choices[choice_index / 16] >> ((choice_index % 16) * 2)) & 3;
        }

        if (ACTIVE(c.out)) {
            ACTIVE(c.out) = false;
            if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
                if (choice == 1) {
                    if (!(c.banks & 1)) {
                        ACTIVE(c.lhs) = true;
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
                        ACTIVE(c.rhs) = true;
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
                    if (c.banks == 1 || c.banks == 2 || c.lhs != c.rhs) {
                        terminal = false;
                    }
                    ACTIVE(c.lhs) |= (!(c.banks & 1));
                    ACTIVE(c.rhs) |= (!(c.banks & 2));
                } else {
                    assert(false);
                }
            } else {
                ACTIVE(c.lhs) |= (!(c.banks & 1));
                ACTIVE(c.rhs) |= (c.opcode >= OP_ADD && !(c.banks & 2));
            }

            // If we've reached the end of the output tape, then
            // allocate a new one and keep going
            if (out_s == 0) {
                const auto next = subtapes.claim();
                subtapes.start[out_subtape_index] = 0;
                subtapes.next[next] = out_subtape_index;
                subtapes.prev[out_subtape_index] = next;

                out_subtape_index = next;
                out_s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                out_tape = subtapes.data[out_subtape_index];
            }

            out_tape[--out_s] = c;
        }
    }

    // The last subtape may not be completely filled, so write its size here
    subtapes.start[out_subtape_index] = out_s;
    subtapes.prev[out_subtape_index] = 0;
    subtiles.setHead(subtile, out_subtape_index, terminal);

    return TILE_AMBIGUOUS;
}

template <unsigned TILE_SIZE_PX, unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__global__
void SubtileRenderer_check(
        SubtileRenderer<TILE_SIZE_PX, SUBTILE_SIZE_PX, DIMENSION>* r,

        const Queue* __restrict__ active_tiles,
        Queue* __restrict__ active_subtiles,

        const Filled<TILE_SIZE_PX>* __restrict__ filled_tiles,
        Filled<SUBTILE_SIZE_PX>* __restrict__ filled_subtiles,

        const uint32_t offset, View v)
{
    assert(blockDim.x % r->subtilesPerTile() == 0);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    constexpr uint32_t subtiles_per_tile =
        std::remove_pointer<decltype(r)>::type::subtilesPerTile();
    constexpr uint32_t subtiles_per_tile_side =
        std::remove_pointer<decltype(r)>::type::subtilesPerTileSide();

    // Pick an active tile from the list.  Each block executes multiple tiles!
    const uint32_t stride = blockDim.x / subtiles_per_tile;
    const uint32_t sub = threadIdx.x / subtiles_per_tile;
    const uint32_t i = offset + blockIdx.x * stride + sub;

    if (i < active_tiles->count) {
        // Pick out the next active tile
        // (this will be the same for every thread in a block)
        const uint32_t tile = (*active_tiles)[i];
        const uint32_t tile_pos = r->tiles.pos(tile);

        // Convert from tile position to pixels
        const uint3 p = r->tiles.lowerCornerVoxel(tile);

        // Calculate the subtile's offset within the tile
        const uint32_t q = threadIdx.x % subtiles_per_tile;
        const uint3 d = make_uint3(
                q % subtiles_per_tile_side,
                (q / subtiles_per_tile_side) % subtiles_per_tile_side,
                (q / subtiles_per_tile_side) / subtiles_per_tile_side);

        const uint32_t tx = p.x / SUBTILE_SIZE_PX + d.x;
        const uint32_t ty = p.y / SUBTILE_SIZE_PX + d.y;
        const uint32_t tz = p.z / SUBTILE_SIZE_PX + d.z;
        if (DIMENSION == 2) {
            assert(tz == 0);
        }

        // This is the subtile index, which isn't the same as its
        // absolute position
        const uint32_t subtile = i * subtiles_per_tile + q;

        // Absolute position of the subtile
        const uint32_t subtiles_per_side = r->subtiles.per_side;
        const uint32_t pos = tx + ty * subtiles_per_side
             + tz * subtiles_per_side * subtiles_per_side;

        // Record the absolute position of the tile
        r->subtiles.pos(subtile) = pos;

        if (!filled_tiles->isMasked(tile_pos) &&
            !filled_subtiles->isMasked(pos))
        {
            switch (r->check(subtile, tile, v)) {
                case TILE_FILLED:
                    filled_subtiles->insert(pos);
                    break;
                case TILE_AMBIGUOUS:
                    r->subtiles.index(subtile) = active_subtiles->insert(subtile);
                    break;
                case TILE_EMPTY:
                    break;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
PixelRenderer<SUBTILE_SIZE_PX, DIMENSION>::PixelRenderer(
        const Tape& tape, const Subtapes& subtapes, Image& image,
        const Tiles<SUBTILE_SIZE_PX, DIMENSION>& prev)
    : tape(tape), subtapes(subtapes), image(image), subtiles(prev)
{
    // Nothing to do here
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

#define FAST_SLOT_COUNT (8 * LIBFIVE_CUDA_SM_SCALE)
    float slots_slow[128 - FAST_SLOT_COUNT];
#if FAST_SLOT_COUNT > 0
    static_assert(pixelsPerSubtile() == 64, "Invalid __shared__ slots size");
    __shared__ float slots_fast[FAST_SLOT_COUNT]
                               [64*LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK];
#define SLOT(i) ((i < FAST_SLOT_COUNT) ? slots_fast[i][threadIdx.x] \
                                       : slots_slow[i - FAST_SLOT_COUNT])
#else
#define SLOT(i) slots_slow[i]
#endif

    // Convert from tile position to pixels
    const uint3 q = subtiles.lowerCornerVoxel(subtile);
    const uint3 p = make_uint3(d.x + q.x, d.y + q.y, d.z + q.z);

    // Skip this pixel if it's already below the image
    if (DIMENSION == 3 && image(p.x, p.y) >= p.z) {
        return;
    }

    {   // Prepopulate axis values
        const float3 f = image.voxelPos(p);
        const Eigen::Vector4f pos_(f.x, f.y, f.z, 1.0f);
        const Eigen::Vector3f pos = (v.mat * pos_).hnormalized();
        if (tape.axes.reg[0] != UINT16_MAX) {
            SLOT(tape.axes.reg[0]) = pos.x();
        }
        if (tape.axes.reg[1] != UINT16_MAX) {
            SLOT(tape.axes.reg[1]) = pos.y();
        }
        if (tape.axes.reg[2] != UINT16_MAX) {
            SLOT(tape.axes.reg[2]) = (DIMENSION == 3)
                ? pos.z()
                : v.mat(2,3);
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
                if (SLOT(tape[s - 1].out) < 0.0f) {
                    if (DIMENSION == 2) {
                        image(p.x, p.y) = 255;
                    } else {
                        atomicMax(&image(p.x, p.y), p.z);
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
            lhs = SLOT(c.lhs);
        }

        float rhs;
        if (c.banks & 2) {
            rhs = constant_ptr[c.rhs];
        } else if (c.opcode >= OP_ADD) {
            rhs = SLOT(c.rhs);
        }

        float out;
        switch (c.opcode) {
            case OP_SQUARE: out = lhs * lhs; break;
            case OP_SQRT: out = sqrtf(lhs); break;
            case OP_NEG: out = -lhs; break;
            case OP_ABS: out = fabsf(lhs); break;

            case OP_ASIN: out = asinf(lhs); break;
            case OP_ACOS: out = acosf(lhs); break;
            case OP_ATAN: out = atanf(lhs); break;
            case OP_EXP: out = expf(lhs); break;
            case OP_SIN: out = sinf(lhs); break;
            case OP_COS: out = cosf(lhs); break;
            case OP_LOG: out = logf(lhs); break;
            // Skipping other transcendental functions for now

            case OP_ADD: out = lhs + rhs; break;
            case OP_MUL: out = lhs * rhs; break;
            case OP_DIV: out = lhs / rhs; break;
            case OP_MIN: out = fminf(lhs, rhs); break;
            case OP_MAX: out = fmaxf(lhs, rhs); break;
            case OP_SUB: out = lhs - rhs; break;

            // Skipping various hard functions here
            default: break;
        }
        SLOT(c.out) = out;
    }
#undef SLOT
#undef FAST_SLOT_COUNT
}

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__global__ void PixelRenderer_draw(
        PixelRenderer<SUBTILE_SIZE_PX, DIMENSION>* r,
        const Queue* __restrict__ active,
        const Filled<SUBTILE_SIZE_PX>* __restrict__ filled,
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

    if (i < active->count) {
        const uint32_t subtile = (*active)[i];
        const uint32_t subtile_pos = r->subtiles.pos(subtile);
        if (!filled->isMasked(subtile_pos)) {
            r->draw(subtile, v);
        }
    }
}

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__device__ void PixelRenderer<SUBTILE_SIZE_PX, DIMENSION>::drawBrute(
        const uint2 p, const View& v)
{
    float regs[128];

    {   // Prepopulate axis values
        const float3 f = image.voxelPos(make_uint3(p.x, p.y, 0));
        const Eigen::Vector4f pos_(f.x, f.y, f.z, 1.0f);
        const Eigen::Vector3f pos = (v.mat * pos_).hnormalized();
        if (tape.axes.reg[0] != UINT16_MAX) {
            regs[tape.axes.reg[0]] = pos.x();
        }
        if (tape.axes.reg[1] != UINT16_MAX) {
            regs[tape.axes.reg[1]] = pos.y();
        }
        if (tape.axes.reg[2] != UINT16_MAX) {
            regs[tape.axes.reg[2]] = (DIMENSION == 3)
                ? pos.z()
                : v.mat(2,3);
        }
    }

    const Clause* __restrict__ clause_ptr = &tape[0];
    const float* __restrict__ constant_ptr = &tape.constant(0);
    const auto num_clauses = tape.num_clauses;

    for (uint32_t i=0; i < num_clauses; ++i) {
        using namespace libfive::Opcode;
        const Clause c = clause_ptr[i];

        // All clauses must have at least one argument, since constants
        // and VAR_X/Y/Z are handled separately.
        float lhs;
        if (c.banks & 1) {
            lhs = constant_ptr[c.lhs];
        } else {
            lhs = regs[c.lhs];
        }

        float rhs;
        if (c.banks & 2) {
            rhs = constant_ptr[c.rhs];
        } else if (c.opcode >= OP_ADD) {
            rhs = regs[c.rhs];
        }

        float out;
        switch (c.opcode) {
            case OP_SQUARE: out = lhs * lhs; break;
            case OP_SQRT: out = sqrtf(lhs); break;
            case OP_NEG: out = -lhs; break;
            case OP_ABS: out = fabsf(lhs); break;

            case OP_ASIN: out = asinf(lhs); break;
            case OP_ACOS: out = acosf(lhs); break;
            case OP_ATAN: out = atanf(lhs); break;
            case OP_EXP: out = expf(lhs); break;
            case OP_SIN: out = sinf(lhs); break;
            case OP_COS: out = cosf(lhs); break;
            case OP_LOG: out = logf(lhs); break;
            // Skipping other transcendental functions for now

            case OP_ADD: out = lhs + rhs; break;
            case OP_MUL: out = lhs * rhs; break;
            case OP_DIV: out = lhs / rhs; break;
            case OP_MIN: out = fminf(lhs, rhs); break;
            case OP_MAX: out = fmaxf(lhs, rhs); break;
            case OP_SUB: out = lhs - rhs; break;

            // Skipping various hard functions here
            default: break;
        }
        regs[c.out] = out;
    }

    const Clause c = clause_ptr[num_clauses - 1];
    const float result = regs[c.out];
    if (result < 0.0f) {
        image(p.x, p.y) = 255;
    }
}

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
__global__ void PixelRenderer_drawBrute(
        PixelRenderer<SUBTILE_SIZE_PX, DIMENSION>* r,
        View v)
{
    uint32_t px = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t py = threadIdx.y + blockIdx.y * blockDim.y;

    if (px < r->image.size_px && py < r->image.size_px) {
        r->drawBrute(make_uint2(px, py), v);
    }
}

////////////////////////////////////////////////////////////////////////////////

NormalRenderer::NormalRenderer(const Tape& tape,
                               const Subtapes& subtapes,
                               Image& norm)
    : tape(tape), subtapes(subtapes), norm(norm)
{
    // Nothing to do here
}

__device__ uint32_t NormalRenderer::draw(const float3 f,
                                         uint32_t subtape_index,
                                         const View& v)
{
#define FAST_SLOT_COUNT (4 * LIBFIVE_CUDA_SM_SCALE)
    Deriv slots_slow[128 - FAST_SLOT_COUNT];
#if FAST_SLOT_COUNT > 0
    __shared__ Deriv slots_fast[FAST_SLOT_COUNT]
                               [64*LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK];
#define SLOT(i) ((i < FAST_SLOT_COUNT) ? slots_fast[i][threadIdx.x] \
                                       : slots_slow[i - FAST_SLOT_COUNT])
#else
#define SLOT(i) slots_slow[i]
#endif

    {   // Prepopulate axis values
        const Eigen::Vector4f pos_(f.x, f.y, f.z, 1.0f);
        const Eigen::Vector3f pos = (v.mat * pos_).hnormalized();
        if (tape.axes.reg[0] != UINT16_MAX) {
            SLOT(tape.axes.reg[0]) = Deriv(pos.x(), v.mat(0, 0), v.mat(0, 1), v.mat(0, 2));
        }
        if (tape.axes.reg[1] != UINT16_MAX) {
            SLOT(tape.axes.reg[1]) = Deriv(pos.y(), v.mat(1, 0), v.mat(1, 1), v.mat(1, 2));
        }
        if (tape.axes.reg[2] != UINT16_MAX) {
            SLOT(tape.axes.reg[2]) = Deriv(pos.z(), v.mat(2, 0), v.mat(2, 1), v.mat(2, 2));
        }
    }

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
                break;
            }
        }
        const Clause c = tape[s++];

        Deriv out;
        switch (c.banks) {
            case 0: // Deriv op Deriv
                out = derivOp<Deriv, Deriv>(c.opcode,
                        SLOT(c.lhs),
                        SLOT(c.rhs));
                break;
            case 1: // Constant op Deriv
                out = derivOp<float, Deriv>(c.opcode,
                        constant_ptr[c.lhs],
                        SLOT(c.rhs));
                break;
            case 2: // Deriv op Constant
                out = derivOp<Deriv, float>(c.opcode,
                        SLOT(c.lhs),
                        constant_ptr[c.rhs]);
                break;
            case 3: // Constant op Constant
                out = derivOp<float, float>(c.opcode,
                        constant_ptr[c.lhs],
                        constant_ptr[c.rhs]);
                break;
        }
        SLOT(c.out) = out;
    }

    const Deriv result = SLOT(tape[s - 1].out);
    float norm = sqrtf(powf(result.dx(), 2) +
                       powf(result.dy(), 2) +
                       powf(result.dz(), 2));
    uint8_t dx = (result.dx() / norm) * 127 + 128;
    uint8_t dy = (result.dy() / norm) * 127 + 128;
    uint8_t dz = (result.dz() / norm) * 127 + 128;
    return (0xFF << 24) | (dz << 16) | (dy << 8) | dx;
#undef SLOT
#undef FAST_SLOT_COUNT
}

__global__ void Renderable3D_drawNormals(
        Renderable3D* r, const uint32_t offset, View v)
{
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t pixel = threadIdx.x % 64;

    const uint32_t i = offset + (threadIdx.x + blockIdx.x * blockDim.x) /
                                64;
    const uint32_t px = (i % (r->norm.size_px / 8)) * 8 +
                        (pixel % 8);
    const uint32_t py = (i / (r->norm.size_px / 8)) * 8 +
                        (pixel / 8);
    if (px < r->norm.size_px && py < r->norm.size_px) {
        const uint32_t pz = min(r->image(px, py) + 1, r->image.size_px - 1);
        if (pz) {
            const uint3 p = make_uint3(px, py, pz);
            const float3 f = r->norm.voxelPos(p);
            const uint32_t h = r->subtapeHeadAt(p);
            if (h) {
                const uint32_t n = r->drawNormals(f, h, v);
                r->norm(p.x, p.y) = n;
            }
        }
    }
}

__global__ void Renderable3D_drawSSAO(Renderable3D* r, const float radius)
{
    r->drawSSAO(radius);
}

__global__ void Renderable3D_blurSSAO(Renderable3D* r)
{
    r->blurSSAO();
}

__global__ void Renderable3D_shade(Renderable3D* r)
{
    r->shade();
}

__device__
uint32_t Renderable3D::drawNormals(const float3 f,
                                   const uint32_t subtape_index,
                                   const View& v)
{
    return normal_renderer.draw(f, subtape_index, v);
}

__device__
uint32_t Renderable3D::subtapeHeadAt(const uint3 v) const
{
    constexpr auto size = decltype(tile_renderer.tiles)::sizePx();
    const uint32_t tx = v.x / size;
    const uint32_t ty = v.y / size;
    const uint32_t tz = v.z / size;

    const uint32_t tile = tx
        + ty * tile_renderer.tiles.per_side
        + tz * pow(tile_renderer.tiles.per_side, 2);
    if (auto h = tile_renderer.tiles.head(tile)) {
        // Map the subtile within the tile
        constexpr auto subsize = decltype(subtile_renderer.subtiles)::sizePx();
        uint32_t sx = (v.x - tx * size) / subsize;
        uint32_t sy = (v.y - ty * size) / subsize;
        uint32_t sz = (v.z - tz * size) / subsize;

        const uint32_t index = tile_renderer.tiles.index(tile);
        const uint32_t subtile = index * subtile_renderer.subtilesPerTile()
            + sx
            + sy * subtile_renderer.subtilesPerTileSide()
            + sz * pow(subtile_renderer.subtilesPerTileSide(), 2);

        if (auto sub_h = subtile_renderer.subtiles.head(subtile)) {
            // Map the microtile within the subtile
            constexpr auto microsize = decltype(microtile_renderer.subtiles)::sizePx();
            uint32_t mx = (v.x - tx * size - sx * subsize) / microsize;
            uint32_t my = (v.y - ty * size - sy * subsize) / microsize;
            uint32_t mz = (v.z - tz * size - sz * subsize) / microsize;

            const uint32_t index = subtile_renderer.subtiles.index(subtile);
            const uint32_t microtile =
                index * microtile_renderer.subtilesPerTile()
                + mx
                + my * microtile_renderer.subtilesPerTileSide()
                + mz * pow(microtile_renderer.subtilesPerTileSide(), 2);

            if (auto micro_h = microtile_renderer.subtiles.head(microtile)) {
                return micro_h;
            } else {
                return sub_h;
            }
        } else {
            return h;
        }
    }
    return 0;
}

__device__
void Renderable3D::copyDepthToImage()
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    const unsigned size = image.size_px;
    if (x < size && y < size) {
        const uint32_t c = image(x, y);
        const uint32_t t = filled_tiles.at(x, y);
        const uint32_t s = filled_subtiles.at(x, y);
        const uint32_t u = filled_microtiles.at(x, y);

        image(x, y) = max(max(c, t), max(s, u));
    }
}

__device__
void Renderable2D::copyDepthToImage()
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    const unsigned size = image.size_px;
    if (x < size && y < size) {
        const uint32_t c = image(x, y);
        const uint32_t t = filled_tiles.at(x, y);
        const uint32_t s = filled_subtiles.at(x, y);

        image(x, y) = (c || t || s) ? (image.size_px - 1) : 0;
    }
}

__global__
void Renderable3D_copyDepthToImage(Renderable3D* r)
{
    r->copyDepthToImage();
}

__global__
void Renderable2D_copyDepthToImage(Renderable2D* r)
{
    r->copyDepthToImage();
}

__device__
void Renderable3D::copyDepthToSurface(cudaSurfaceObject_t surf,
                                      uint32_t texture_size,
                                      bool append)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size && y < texture_size) {
        uint32_t px = x * image.size_px / texture_size;
        uint32_t py = y * image.size_px / texture_size;
        const auto h = image(px, image.size_px - py - 1);
        if (h) {
            surf2Dwrite(0x00FFFFFF | (((h * 255) / image.size_px) << 24),
                        surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__device__
void Renderable3D::copySSAOToSurface(cudaSurfaceObject_t surf,
                                     uint32_t texture_size,
                                     bool append)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size && y < texture_size) {
        uint32_t px = x * image.size_px / texture_size;
        uint32_t py = y * image.size_px / texture_size;
        const auto h = image(px, image.size_px - py - 1);
        if (h) {
            const uint8_t o = ssao(px, image.size_px - py - 1);
            surf2Dwrite((0xFF << 24) | (o << 16) | (o << 8) | (o << 0),
                        surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__device__
void Renderable3D::shade()
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < image.size_px && y < image.size_px) {
        const auto h = image(x, y);
        if (h) {
            const uint8_t s = ssao(x, y);

            // Get normal from image
            const auto n = norm(x, y);
            float dx = (float)(n & 0xFF) - 128.0f;
            float dy = (float)((n >> 8) & 0xFF) - 128.0f;
            float dz = (float)((n >> 16) & 0xFF) - 128.0f;
            Eigen::Vector3f normal = Eigen::Vector3f{dx, dy, dz}.normalized();

            // Apply a single light
            const float3 pos_f3 = image.voxelPos(make_uint3(x, y, h));
            const Eigen::Vector3f pos { pos_f3.x, pos_f3.y, pos_f3.z };

            const Eigen::Vector3f light_pos { 5, 5, 10 };
            const Eigen::Vector3f light_dir = (light_pos - pos).normalized();

            // Apply light
            float light = fmaxf(0.0f, light_dir.dot(normal)) * 0.8f;

            // SSAO dimming
            light *= s / 255.0f;

            // Ambient
            light += 0.2f;

            // Clamp
            if (light < 0.0f) {
                light = 0.0f;
            } else if (light > 1.0f) {
                light = 1.0f;
            }

            uint8_t color = light * 255.0f;

            temp(x, y) = (0xFF << 24) | (color << 16) | (color << 8) | (color << 0);
        }
    }
}

__device__
void Renderable3D::copyShadedToSurface(cudaSurfaceObject_t surf,
                                       uint32_t texture_size,
                                       bool append)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size && y < texture_size) {
        uint32_t px = x * image.size_px / texture_size;
        uint32_t py = y * image.size_px / texture_size;
        const auto h = image(px, image.size_px - py - 1);
        if (h) {
            const uint32_t c = temp(px, image.size_px - py - 1);
            surf2Dwrite(c, surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__device__
void Renderable3D::drawSSAO(float radius)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < image.size_px && y < image.size_px) {
        const auto h = image(x, y);
        const float3 pos = image.voxelPos(make_uint3(x, y, h));
        if (h) {
            // Based on http://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
            uint32_t n = norm(x, y);

            // Get normal from image
            float dx = (float)(n & 0xFF) - 128.0f;
            float dy = (float)((n >> 8) & 0xFF) - 128.0f;
            float dz = (float)((n >> 16) & 0xFF) - 128.0f;
            Eigen::Vector3f normal = Eigen::Vector3f{dx, dy, dz}.normalized();

            Eigen::Vector3f rvec = ssao_rvecs.row((threadIdx.x % 16) * 16 + (threadIdx.y % 16));
            Eigen::Vector3f tangent = (rvec - normal * rvec.dot(normal)).normalized();
            Eigen::Vector3f bitangent = normal.cross(tangent);
            Eigen::Matrix3f tbn;
            tbn.col(0) = tangent;
            tbn.col(1) = bitangent;
            tbn.col(2) = normal;

            float occlusion = 0.0f;
            for (unsigned i=0; i < ssao_kernel.rows(); ++i) {
                Eigen::Vector3f sample_pos =
                    tbn * ssao_kernel.row(i).transpose() * radius +
                    Eigen::Vector3f{pos.x, pos.y, pos.z};

                const unsigned px = (sample_pos.x() / 2.0f + 0.5f) * image.size_px;
                const unsigned py = (sample_pos.y() / 2.0f + 0.5f) * image.size_px;
                const unsigned actual_h =
                    (px < image.size_px && py < image.size_px)
                    ? image(px, py)
                    : 0;
                const float actual_z = 2.0f * ((actual_h + 0.5f) / image.size_px - 0.5f);

                const auto dz = fabsf(sample_pos.z() - actual_z);
                if (dz < radius) {
                    occlusion += sample_pos.z() <= actual_z;
                } else if (dz < radius * 2.0f) {
                    if (sample_pos.z() <= actual_z) {
                        occlusion += powf((radius - (dz - radius)) / radius, 2.0f);
                    }
                }
            }
            occlusion = 1.0 - (occlusion / ssao_kernel.rows());
            const uint8_t o = occlusion * 255;
            ssao(x, y) = o;
        }
    }
}

__device__
void Renderable3D::blurSSAO(void)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    const int BLUR_RADIUS = 2;

    float best = 1000000.0f;
    float value = 0.0f;
    auto run = [x, y, this, &best, &value](int xmin, int ymin) {
        float sum = 0.0f;
        float count = 0.0f;
        for (int i=0; i <= BLUR_RADIUS; ++i) {
            for (int j=0; j <= BLUR_RADIUS; ++j) {
                const int tx = x + xmin + i;
                const int ty = y + ymin + j;
                if (tx >= 0 && tx < image.size_px &&
                    ty >= 0 && ty < image.size_px)
                {
                    if (image(tx, ty)) {
                        sum += ssao(tx, ty);
                        count++;
                    }
                }
            }
        }
        const float mean = sum / count;
        float stdev = 0.0f;
        for (int i=0; i <= BLUR_RADIUS; ++i) {
            for (int j=0; j <= BLUR_RADIUS; ++j) {
                const int tx = xmin + i;
                const int ty = ymin + j;
                if (tx >= 0 && tx < image.size_px &&
                    ty >= 0 && ty < image.size_px)
                {
                    if (image(tx, ty)) {
                        const float d = (mean - ssao(tx, ty));
                        stdev += d * d;
                    }
                }
            }
        }
        stdev /= count - 1.0f;
        stdev = sqrtf(stdev);
        if (stdev < best) {
            best = stdev;
            value = mean;
        }
    };

    for (unsigned i=0; i < 4; ++i) {
        run((i & 1) ? 0 : -BLUR_RADIUS,
            (i & 2) ? 0 : -BLUR_RADIUS);
    }
    temp(x, y) = value;
}

__device__
void Renderable3D::copyNormalToSurface(cudaSurfaceObject_t surf,
                                       uint32_t texture_size,
                                       bool append)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size && y < texture_size) {
        uint32_t px = x * image.size_px / texture_size;
        uint32_t py = y * image.size_px / texture_size;
        const auto h = image(px, image.size_px - py - 1);
        if (h) {
            surf2Dwrite(norm(px, image.size_px - py - 1), surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__device__
void Renderable2D::copyToSurface(cudaSurfaceObject_t surf,
                                 uint32_t texture_size, bool append)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size && y < texture_size) {
        const uint32_t px = x * image.size_px / texture_size;
        const uint32_t py = y * image.size_px / texture_size;
        const auto h = image(px, image.size_px - py - 1);
        if (h) {
            surf2Dwrite(0xFFFFFFFF, surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__global__
void Renderable3D_copyDepthToSurface(Renderable3D* r, cudaSurfaceObject_t surf,
                                     uint32_t texture_size, bool append)
{
    r->copyDepthToSurface(surf, texture_size, append);
}

__global__
void Renderable3D_copyNormalToSurface(Renderable3D* r,
                                      cudaSurfaceObject_t surf,
                                      uint32_t texture_size, bool append)
{
    r->copyNormalToSurface(surf, texture_size, append);
}

__global__
void Renderable3D_copySSAOToSurface(Renderable3D* r,
                                      cudaSurfaceObject_t surf,
                                      uint32_t texture_size, bool append)
{
    r->copySSAOToSurface(surf, texture_size, append);
}

__global__
void Renderable3D_copyShadedToSurface(Renderable3D* r,
                                      cudaSurfaceObject_t surf,
                                      uint32_t texture_size, bool append)
{
    r->copyShadedToSurface(surf, texture_size, append);
}

__global__
void Renderable2D_copyToSurface(Renderable2D* r, cudaSurfaceObject_t surf,
                                uint32_t texture_size, bool append)
{
    r->copyToSurface(surf, texture_size, append);
}

////////////////////////////////////////////////////////////////////////////////

void Renderable::Deleter::operator()(Renderable* r)
{
    r->~Renderable();
    CUDA_FREE(r);
}

Renderable::~Renderable()
{
    for (unsigned i=0; i < LIBFIVE_CUDA_NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

Renderable::Handle Renderable::build(libfive::Tree tree, uint32_t image_size_px, uint8_t dimension)
{
    Renderable* out;
    if (dimension == 2) {
        out = CUDA_MALLOC(Renderable2D, 1);
        new (out) Renderable2D(tree, image_size_px);
    } else if (dimension == 3) {
        out = CUDA_MALLOC(Renderable3D, 1);
        new (out) Renderable3D(tree, image_size_px);
    }
    cudaDeviceSynchronize();
    return Handle(out);
}

void Renderable::printStats() const {
    std::cout << "choices: " << tape.num_csg_choices << "\n";
    std::cout << "regs: " << tape.num_regs << "\n";
    std::cout << "clauses: " << tape.num_clauses << "\n";
    std::cout << "constants: " << tape.num_constants << "\n";
}

Renderable::Renderable(libfive::Tree tree, uint32_t image_size_px)
    : image(image_size_px),
      tape(std::move(Tape::build(tree)))
{
    for (unsigned i=0; i < LIBFIVE_CUDA_NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
}

Renderable3D::Renderable3D(libfive::Tree tree, uint32_t image_size_px)
    : Renderable(tree, image_size_px),
      norm(image_size_px),
      ssao(image_size_px),
      temp(image_size_px),

      filled_tiles(image_size_px),
      filled_subtiles(image_size_px),
      filled_microtiles(image_size_px),

      tile_renderer(tape, subtapes, image),
      subtile_renderer(tape, subtapes, image, tile_renderer.tiles),
      microtile_renderer(tape, subtapes, image, subtile_renderer.subtiles),

      pixel_renderer(tape, subtapes, image, microtile_renderer.subtiles),
      normal_renderer(tape, subtapes, norm)
{
    // Based on http://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
    for (unsigned i = 0; i < ssao_kernel.rows(); ++i) {
        ssao_kernel.row(i) = Eigen::RowVector3f{
            2.0f * ((float)(rand()) / (float)(RAND_MAX) - 0.5f),
            2.0f * ((float)(rand()) / (float)(RAND_MAX) - 0.5f),
            (float)(rand()) / (float)(RAND_MAX) };
        ssao_kernel.row(i) /= ssao_kernel.row(i).norm();

        // Scale to keep most samples near the center
        float scale = float(i) / float(ssao_kernel.rows() - 1);
        scale = (scale * scale) * 0.9f + 0.1f;
        ssao_kernel.row(i) *= scale;
    }
    for (unsigned i = 0; i < ssao_rvecs.rows(); ++i) {
        ssao_rvecs.row(i) = Eigen::RowVector3f{
            2.0f * ((float)(rand()) / (float)(RAND_MAX) - 0.5f),
            2.0f * ((float)(rand()) / (float)(RAND_MAX) - 0.5f),
            0.0f };
        ssao_rvecs.row(i) /= ssao_rvecs.row(i).norm();
    }
}

Renderable2D::Renderable2D(libfive::Tree tree, uint32_t image_size_px)
    : Renderable(tree, image_size_px),

      filled_tiles(image_size_px),
      filled_subtiles(image_size_px),

      tile_renderer(tape, subtapes, image),
      subtile_renderer(tape, subtapes, image, tile_renderer.tiles),

      pixel_renderer(tape, subtapes, image, subtile_renderer.subtiles)
{
    // Nothing to do here
}

void Renderable3D::run(const View& view, Renderable::Mode mode)
{
    // Reset everything in preparation for a render
    subtapes.reset();
    image.reset();
    norm.reset();
    ssao.reset();
    tile_renderer.tiles.reset();
    subtile_renderer.subtiles.reset();
    microtile_renderer.subtiles.reset();

    filled_tiles.reset();
    filled_subtiles.reset();
    filled_microtiles.reset();

    // Record this local variable because otherwise it looks up memory
    // that has been loaned to the GPU and not synchronized.
    auto tile_renderer = &this->tile_renderer;
    auto subtile_renderer = &this->subtile_renderer;
    auto microtile_renderer = &this->microtile_renderer;
    auto pixel_renderer = &this->pixel_renderer;

    cudaStream_t streams[LIBFIVE_CUDA_NUM_STREAMS];
    for (unsigned i=0; i < LIBFIVE_CUDA_NUM_STREAMS; ++i) {
        streams[i] = this->streams[i];
    }

    {   // Do per-tile evaluation to get filled / ambiguous tiles
        const uint32_t stride = LIBFIVE_CUDA_TILE_THREADS *
                                LIBFIVE_CUDA_TILE_BLOCKS;

        const uint32_t total_tiles = pow(
                image.size_px / decltype(tile_renderer->tiles)::sizePx(),
                decltype(tile_renderer->tiles)::dimension());
        tile_renderer->tiles.resizeToFit(total_tiles);
        tile_renderer->tiles.setDefaultPositions();

        auto queue_out = &this->queue_ping;
        queue_out->resizeToFit(total_tiles);

        auto filled_out = &this->filled_tiles;

        for (unsigned i=0; i < total_tiles; i += stride) {
            TileRenderer_check<<<
                LIBFIVE_CUDA_TILE_BLOCKS,
                LIBFIVE_CUDA_TILE_THREADS,
                0,
                streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    tile_renderer, queue_out, filled_out, i, view);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    {   // Refine ambiguous tiles from their subtapes
        const uint32_t stride = LIBFIVE_CUDA_REFINE_BLOCKS *
                                LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK;
        auto queue_in  = &this->queue_ping;
        const uint32_t active = queue_in->count;

        auto queue_out = &this->queue_pong;
        queue_out->resizeToFit(active * subtile_renderer->subtilesPerTile());

        auto filled_in  = &this->filled_tiles;
        auto filled_out = &this->filled_subtiles;

        subtile_renderer->subtiles.resizeToFit(
                active * subtile_renderer->subtilesPerTile());

        for (unsigned i=0; i < active; i += stride) {
            SubtileRenderer_check<<<
                LIBFIVE_CUDA_REFINE_BLOCKS,
                subtile_renderer->subtilesPerTile() *
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK,
                0,
                streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    subtile_renderer,
                    queue_in, queue_out,
                    filled_in, filled_out,
                    i, view);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    {   // Refine ambiguous tiles from their subtapes
        const uint32_t stride = LIBFIVE_CUDA_REFINE_BLOCKS *
                                LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK;

        auto queue_in  = &this->queue_pong;
        const uint32_t active = queue_in->count;

        auto filled_in  = &this->filled_subtiles;
        auto filled_out = &this->filled_microtiles;

        auto queue_out = &this->queue_ping;
        queue_out->resizeToFit(active * microtile_renderer->subtilesPerTile());

        microtile_renderer->subtiles.resizeToFit(
                active * microtile_renderer->subtilesPerTile());

        for (unsigned i=0; i < active; i += stride) {
            SubtileRenderer_check<<<
                LIBFIVE_CUDA_REFINE_BLOCKS,
                microtile_renderer->subtilesPerTile() *
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK,
                0,
                streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    microtile_renderer,
                    queue_in, queue_out,
                    filled_in, filled_out,
                    i, view);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    {   // Do pixel-by-pixel rendering for active subtiles
        const uint32_t stride = LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS *
                                LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK;

        auto queue_in  = &this->queue_ping;
        const uint32_t active = queue_in->count;

        auto filled_in = &this->filled_microtiles;

        for (unsigned i=0; i < active; i += stride) {
            PixelRenderer_draw<<<
                LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS,
                pixel_renderer->pixelsPerSubtile() *
                    LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK,
                0,
                streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    pixel_renderer, queue_in, filled_in, i, view);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const unsigned u = (image.size_px + 15) / 16;
    Renderable3D_copyDepthToImage<<<dim3(u, u), dim3(16, 16)>>>(this);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (mode >= MODE_NORMALS) {
        const uint32_t active = pow(image.size_px / 8, 2);
        const uint32_t stride = LIBFIVE_CUDA_NORMAL_RENDER_BLOCKS *
                                LIBFIVE_CUDA_NORMAL_RENDER_TILES_PER_BLOCK;
        for (unsigned i=0; i < active; i += stride) {
            Renderable3D_drawNormals<<<
                LIBFIVE_CUDA_NORMAL_RENDER_BLOCKS,
                pow(8, 2) * LIBFIVE_CUDA_NORMAL_RENDER_TILES_PER_BLOCK,
                0, streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    this, i, view);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (mode >= MODE_SSAO) {
        const float radius = 0.1f;
        Renderable3D_drawSSAO<<<dim3(u, u), dim3(16, 16)>>>(this, radius);
        Renderable3D_blurSSAO<<<dim3(u, u), dim3(16, 16)>>>(this);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaMemcpy(ssao.data, temp.data,
                   sizeof(uint32_t) * image.size_px * image.size_px,
                   cudaMemcpyDeviceToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (mode >= MODE_SHADED) {
        Renderable3D_shade<<<dim3(u, u), dim3(16, 16)>>>(this);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void Renderable2D::run(const View& view, Renderable::Mode mode)
{
    // 2D rendering only uses heightmap
    (void)mode;

    // Reset everything in preparation for a render
    subtapes.reset();
    image.reset();
    tile_renderer.tiles.reset();
    subtile_renderer.subtiles.reset();

    filled_tiles.reset();
    filled_subtiles.reset();

    // Record this local variable because otherwise it looks up memory
    // that has been loaned to the GPU and not synchronized.
    auto tile_renderer = &this->tile_renderer;
    auto subtile_renderer = &this->subtile_renderer;
    auto pixel_renderer = &this->pixel_renderer;

    cudaStream_t streams[LIBFIVE_CUDA_NUM_STREAMS];
    for (unsigned i=0; i < LIBFIVE_CUDA_NUM_STREAMS; ++i) {
        streams[i] = this->streams[i];
    }

    {   // Do per-tile evaluation to get filled / ambiguous tiles
        const uint32_t stride = LIBFIVE_CUDA_TILE_THREADS *
                                LIBFIVE_CUDA_TILE_BLOCKS;

        const uint32_t total_tiles = pow(
                image.size_px / decltype(tile_renderer->tiles)::sizePx(),
                decltype(tile_renderer->tiles)::dimension());
        tile_renderer->tiles.resizeToFit(total_tiles);
        tile_renderer->tiles.setDefaultPositions();

        auto queue_out = &this->queue_ping;
        queue_out->resizeToFit(total_tiles);

        auto filled_out = &this->filled_tiles;

        for (unsigned i=0; i < total_tiles; i += stride) {
            TileRenderer_check<<<
                LIBFIVE_CUDA_TILE_BLOCKS,
                LIBFIVE_CUDA_TILE_THREADS,
                0,
                streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    tile_renderer, queue_out, filled_out, i, view);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    {   // Refine ambiguous tiles from their subtapes
        const uint32_t stride = LIBFIVE_CUDA_REFINE_BLOCKS *
                                LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK;

        auto queue_in  = &this->queue_ping;
        const uint32_t active = queue_in->count;

        auto queue_out = &this->queue_pong;
        queue_out->resizeToFit(active * subtile_renderer->subtilesPerTile());

        auto filled_in  = &this->filled_tiles;
        auto filled_out = &this->filled_subtiles;

        subtile_renderer->subtiles.resizeToFit(
                active * subtile_renderer->subtilesPerTile());

        for (unsigned i=0; i < active; i += stride) {
            SubtileRenderer_check<<<
                LIBFIVE_CUDA_REFINE_BLOCKS,
                subtile_renderer->subtilesPerTile() *
                    LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK,
                0,
                streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    subtile_renderer,
                    queue_in, queue_out,
                    filled_in, filled_out,
                    i, view);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    {   // Do pixel-by-pixel rendering for active subtiles
        const uint32_t stride = LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS *
                                LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK;
        auto queue_in  = &this->queue_pong;
        auto filled_in = &this->filled_subtiles;
        const uint32_t active = queue_in->count;
        for (unsigned i=0; i < active; i += stride) {
            PixelRenderer_draw<<<
                LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS,
                pixel_renderer->pixelsPerSubtile() *
                    LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK,
                0,
                streams[(i / stride) % LIBFIVE_CUDA_NUM_STREAMS]>>>(
                    pixel_renderer, queue_in, filled_in, i, view);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const unsigned u = (image.size_px + 15) / 16;
    Renderable2D_copyDepthToImage<<<dim3(u, u), dim3(16, 16)>>>(this);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Renderable2D::runBrute(const View& view)
{
    // Reset everything in preparation for a render
    image.reset();

    const unsigned bs = (image.size_px + 15) / 16;
    PixelRenderer_drawBrute<<<dim3(bs, bs), dim3(16, 16)>>>(
            &this->pixel_renderer, view);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Renderable2D::runBruteKernel(const View& v)
{
    // Reset everything in preparation for a render
    image.reset();

    const unsigned bs = (image.size_px + 15) / 16;
    evalRawTape<<<dim3(bs, bs), dim3(16, 16)>>>(&this->image, v);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

cudaGraphicsResource* Renderable::registerTexture(GLuint t)
{
    cudaGraphicsResource* gl_tex;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&gl_tex, t, GL_TEXTURE_2D,
                                      cudaGraphicsMapFlagsWriteDiscard));
    return gl_tex;
}

void Renderable2D::copyToTexture(cudaGraphicsResource* gl_tex,
                                 uint32_t texture_size,
                                 bool append, Renderable::Mode mode)
{
    (void)mode; // (unused in 2D)
    const unsigned u = (texture_size + 15) / 16;

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
    Renderable2D_copyToSurface<<<dim3(u, u), dim3(16, 16)>>>(
            this, surf, texture_size, append);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDestroySurfaceObject(surf));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &gl_tex));
}

void Renderable3D::copyToTexture(cudaGraphicsResource* gl_tex,
                                 uint32_t texture_size,
                                 bool append, Mode mode)
{
    const unsigned u = (texture_size + 15) / 16;

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
    if (mode == MODE_HEIGHTMAP) {
        Renderable3D_copyDepthToSurface<<<dim3(u, u), dim3(16, 16)>>>(
                this, surf, texture_size, append);
    } else if (mode == MODE_NORMALS) {
        Renderable3D_copyNormalToSurface<<<dim3(u, u), dim3(16, 16)>>>(
                this, surf, texture_size, append);
    } else if (mode == MODE_SSAO) {
        Renderable3D_copySSAOToSurface<<<dim3(u, u), dim3(16, 16)>>>(
                this, surf, texture_size, append);
    } else if (mode == MODE_SHADED) {
        Renderable3D_copyShadedToSurface<<<dim3(u, u), dim3(16, 16)>>>(
                this, surf, texture_size, append);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDestroySurfaceObject(surf));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &gl_tex));
}
