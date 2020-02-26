#pragma once
#include <Eigen/Eigen>
#include "tape.hpp"

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

struct v2_blob_t {
    uint32_t* filled_tiles;
    uint32_t* filled_subtiles;
    uint32_t* filled_microtiles;

    uint32_t image_size_px;
    uint32_t* image;

    uint64_t* tape; // original tape

    uint64_t* subtapes; // big array of subtapes
    uint32_t* subtape_index; // single value

    uint32_t* ping_index; // single value
    uint32_t* ping_queue; // array
    uint32_t  ping_queue_len; // size of allocated array

    uint32_t* pong_index; // single value
    uint32_t* pong_queue; // array
    uint32_t  pong_queue_len; // size of allocated array

    void* values; // buffer used when assigning values
};

v2_blob_t build_v2_blob(const Tape& tape, const uint32_t image_size_px);

uint64_t* build_v2_tape(const Tape& tape, const uint32_t size);
void eval_v2_tape(const uint64_t* data, uint32_t* image, uint32_t size);

void render_v2_blob(v2_blob_t blob, Eigen::Matrix4f mat);
