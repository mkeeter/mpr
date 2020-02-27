#pragma once
#include <Eigen/Eigen>

#include "libfive/tree/tree.hpp"
#include "gpu_interval.hpp"

/* The clause is implemented as a struct
 * packed into a single 64-bit value
    struct clause_t {
        uint8_t opcode;
        uint8_t flags;
        uint8_t out;
        uint8_t lhs;
        union {
            uint8_t rhs;
            float imm;
        }
    };
*/

struct in_tile_t {
    uint32_t position;
    uint32_t tape;
    Interval X, Y, Z;
};

struct out_tile_t {
    uint32_t position;
    uint32_t tape;
};

struct stage_t {
    in_tile_t* input;
    out_tile_t* output;
    uint32_t* index; // single value
    uint32_t input_array_size;
    uint32_t output_array_size;
};

struct v2_blob_t {
    uint32_t* filled_tiles;
    uint32_t* filled_subtiles;
    uint32_t* filled_microtiles;

    uint32_t image_size_px;
    uint32_t* image;

    uint64_t* tape_data; // original tape is at index 0
    uint32_t* tape_index; // single value

    stage_t tiles;
    stage_t subtiles;
    stage_t microtiles;

    float* values; // buffer used when assigning values for float evaluation
};

v2_blob_t build_v2_blob(libfive::Tree tree, const uint32_t image_size_px);
void render_v2_blob(v2_blob_t blob, Eigen::Matrix4f mat);
