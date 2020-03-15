#pragma once
#include <cstdint>

#include "util.hpp"

namespace libfive {
namespace cuda {

// Forward declaration
struct Tape;

struct TileNode {
    int32_t position;
    int32_t tape;
    int32_t next;
};

struct Tiles {
    /* 2D array of filled Z values (or 0) */
    Ptr<int32_t> filled;

    /*  1D list of active tiles */
    Ptr<TileNode> tiles;
    size_t tile_array_size;
};

struct Context {
    Context(int32_t image_size_px);

    const int32_t image_size_px;

    Ptr<uint64_t> tape_data;    // original tape is copied to index 0
    Ptr<int32_t> tape_index;    // single value

    Tiles stages[4];        // 64^3, 16^3, 4^3, voxels

    Ptr<int32_t> num_active_tiles;  // GPU-allocated count of active tiles

    Ptr<void> values; // Used to pass data around
    size_t values_size;

    Ptr<uint32_t> normals;
};

} // cuda
} // libfive
