#include "gpu_opcode.hpp"

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
