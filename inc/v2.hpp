#pragma once
#include <Eigen/Eigen>

#include "libfive/tree/tree.hpp"

/* The clause is implemented as a struct
 * packed into a single 64-bit value
    struct clause_t {
        uint8_t opcode;
        uint8_t out;
        uint8_t lhs;
        uint8_t rhs;
        union {
            uint8_t rhs;
            float imm;
        }
    };
*/

// Forward declarations
struct in_tile_t;
struct out_tile_t;

struct stage_t {
    void resize_to_fit(size_t count);

    uint32_t* filled;

    in_tile_t* input;
    uint32_t input_array_size;

    // The output array is the same size as the input array
    // TODO: re-use the input array instead?  Proposal:
    //      To mark a tile as filled, leave its position and set tape = -1
    //      To discard an empty tile, set its position to -1
    //      Otherwise (needs recursion), leave its position unchanged
    //          and update the tape value based on pushing
    out_tile_t* output;
    uint32_t* output_index;
};

struct v2_blob_t {
    uint32_t image_size_px;
    uint32_t* image;

    uint64_t* tape_data;    // original tape is at index 0
    uint32_t* tape_index;   // single value
    uint32_t tape_length;   // used to reset tape index

    stage_t tiles;
    stage_t subtiles;
    stage_t microtiles;
};

v2_blob_t build_v2_blob(libfive::Tree tree, const uint32_t image_size_px);
void render_v2_blob(v2_blob_t& blob, Eigen::Matrix4f mat);
void free_v2_blob(v2_blob_t& blob);
