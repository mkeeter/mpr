#pragma once

enum GPUOp {
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
