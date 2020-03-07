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

struct v3_tape_push_data_t {
    uint32_t choices[256];
    int32_t choice_index;
    int32_t tape_end;
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
    int32_t* image;

    uint64_t* tape_data;    // original tape is at index 0
    int32_t* tape_index;    // single value
    int32_t tape_length;    // used to reset tape index

    v3_tiles_t stages[3];   // 64^3, 16^3, 4^3

    int32_t* num_active_tiles;  // GPU-allocated count of active tiles

    v3_tape_push_data_t* push_data; // Data useful when pushing the tape
    int32_t* push_target_buffer;    // Array of values which need tape push
    int32_t* push_target_count;     // Single counter value allocated on GPU

    void* values; // Used to pass data around
};

////////////////////////////////////////////////////////////////////////////////

v3_blob_t build_v3_blob(libfive::Tree tree, const int32_t image_size_px);
void render_v3_blob(v3_blob_t& blob, Eigen::Matrix4f mat);
void free_v3_blob(v3_blob_t& blob);
