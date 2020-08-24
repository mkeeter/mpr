/*
Reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace mpr {

enum Opcode {
    GPU_OP_INVALID = 0,
    GPU_OP_JUMP, // meta-op to jump to a new point in the tape

    GPU_OP_SQUARE_LHS,
    GPU_OP_SQRT_LHS,
    GPU_OP_NEG_LHS,
    GPU_OP_SIN_LHS,
    GPU_OP_COS_LHS,
    GPU_OP_ASIN_LHS,
    GPU_OP_ACOS_LHS,
    GPU_OP_ATAN_LHS,
    GPU_OP_EXP_LHS,
    GPU_OP_ABS_LHS,
    GPU_OP_LOG_LHS,

    // Commutative opcodes
    GPU_OP_ADD_LHS_IMM,
    GPU_OP_ADD_LHS_RHS,
    GPU_OP_MUL_LHS_IMM,
    GPU_OP_MUL_LHS_RHS,
    GPU_OP_MIN_LHS_IMM,
    GPU_OP_MIN_LHS_RHS,
    GPU_OP_MAX_LHS_IMM,
    GPU_OP_MAX_LHS_RHS,
    GPU_OP_RAD_LHS_IMM,
    GPU_OP_RAD_LHS_RHS,

    // Non-commutative opcodes
    GPU_OP_SUB_LHS_IMM,
    GPU_OP_SUB_IMM_RHS,
    GPU_OP_SUB_LHS_RHS,
    GPU_OP_DIV_LHS_IMM,
    GPU_OP_DIV_IMM_RHS,
    GPU_OP_DIV_LHS_RHS,

    // Copy-only opcodes (used after pushing)
    GPU_OP_COPY_IMM,
    GPU_OP_COPY_LHS,
    GPU_OP_COPY_RHS,
};

__host__ __device__
const char* gpu_op_str(uint8_t op);

}   // namespace mpr
