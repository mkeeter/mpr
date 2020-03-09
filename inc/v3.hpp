#pragma once
#include <cstdint>
#include <Eigen/Eigen>
#include <cuda_runtime.h>

#include "libfive/tree/tree.hpp"

struct v3_tile_node_t {
    int32_t position;
    int32_t tape;
    int32_t next;
};

struct v3_tiles_t {
    /* 2D array of filled Z values (or 0) */
    int32_t* filled;

    /*  1D list of active tiles */
    v3_tile_node_t* tiles;
    size_t tile_array_size;
};

struct v3_blob_t {
    int32_t image_size_px;

    uint64_t* tape_data;    // original tape is at index 0
    int32_t* tape_index;    // single value
    int32_t tape_length;    // used to reset tape index

    v3_tiles_t stages[4];   // 64^3, 16^3, 4^3, voxels

    int32_t* num_active_tiles;  // GPU-allocated count of active tiles

    void* values; // Used to pass data around
    int32_t values_size;

    uint32_t* normals;
};

////////////////////////////////////////////////////////////////////////////////

v3_blob_t build_v3_blob(libfive::Tree tree, const int32_t image_size_px);
void render_v3_blob(v3_blob_t& blob, Eigen::Matrix4f mat);
void free_v3_blob(v3_blob_t& blob);
