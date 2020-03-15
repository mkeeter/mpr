#pragma once

// A clause is defined as a 64-bit value.
//
// These macros let us address individual parts of that value, in a way
// which almost definitely is Undefined Behavior.
#define OP(d) (((uint8_t*)d)[0])
#define I_OUT(d) (((uint8_t*)d)[1])
#define I_LHS(d) (((uint8_t*)d)[2])
#define I_RHS(d) (((uint8_t*)d)[3])
#define IMM(d) (((float*)d)[1])
#define JUMP_TARGET(d) (((int32_t*)d)[1])
