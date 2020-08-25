/*
Reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include "gpu_opcode.hpp"

namespace mpr {

__host__ __device__
const char* gpu_op_str(uint8_t op) {
    switch (op) {
        case GPU_OP_INVALID: return "INVALID";
        case GPU_OP_JUMP: return "JUMP";

        case GPU_OP_SQUARE_LHS: return "SQUARE_LHS";
        case GPU_OP_SQRT_LHS: return "SQRT_LHS";
        case GPU_OP_NEG_LHS: return "NEG_LHS";
        case GPU_OP_SIN_LHS: return "SIN_LHS";
        case GPU_OP_COS_LHS: return "COS_LHS";
        case GPU_OP_ASIN_LHS: return "ASIN_LHS";
        case GPU_OP_ACOS_LHS: return "ACOS_LHS";
        case GPU_OP_ATAN_LHS: return "ATAN_LHS";
        case GPU_OP_EXP_LHS: return "EXP_LHS";
        case GPU_OP_ABS_LHS: return "ABS_LHS";
        case GPU_OP_LOG_LHS: return "LOG_LHS";

        // Commutative opcodes
        case GPU_OP_ADD_LHS_IMM: return "ADD_LHS_IMM";
        case GPU_OP_ADD_LHS_RHS: return "ADD_LHS_RHS";
        case GPU_OP_MUL_LHS_IMM: return "MUL_LHS_IMM";
        case GPU_OP_MUL_LHS_RHS: return "MUL_LHS_RHS";
        case GPU_OP_MIN_LHS_IMM: return "MIN_LHS_IMM";
        case GPU_OP_MIN_LHS_RHS: return "MIN_LHS_RHS";
        case GPU_OP_MAX_LHS_IMM: return "MAX_LHS_IMM";
        case GPU_OP_MAX_LHS_RHS: return "MAX_LHS_RHS";
        case GPU_OP_RAD_LHS_RHS: return "RAD_LHS_RHS";

        // Non-commutative opcodes
        case GPU_OP_SUB_LHS_IMM: return "SUB_LHS_IMM";
        case GPU_OP_SUB_IMM_RHS: return "SUB_IMM_RHS";
        case GPU_OP_SUB_LHS_RHS: return "SUB_LHS_RHS";
        case GPU_OP_DIV_LHS_IMM: return "DIV_LHS_IMM";
        case GPU_OP_DIV_IMM_RHS: return "DIV_IMM_RHS";
        case GPU_OP_DIV_LHS_RHS: return "DIV_LHS_RHS";

        // Copy-only opcodes (used after pushing)
        case GPU_OP_COPY_IMM: return "COPY_IMM";
        case GPU_OP_COPY_LHS: return "COPY_LHS";
        case GPU_OP_COPY_RHS: return "COPY_RHS";
        default: return "UNKNOWN";
    }
}

}   // namespace mpr
