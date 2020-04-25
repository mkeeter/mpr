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

// A clause is defined as a 64-bit value.
//
// These macros let us address individual parts of that value, in a way
// which almost definitely is Undefined Behavior.
#define OP(d) (((uint8_t*)(d))[0])
#define I_OUT(d) (((uint8_t*)(d))[1])
#define I_LHS(d) (((uint8_t*)(d))[2])
#define I_RHS(d) (((uint8_t*)(d))[3])
#define IMM(d) (((float*)(d))[1])
#define JUMP_TARGET(d) (((int32_t*)(d))[1])
