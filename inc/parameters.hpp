/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#pragma once

#define NUM_TILES (4)
#define NUM_THREADS (64 * NUM_TILES)
#define SUBTAPE_CHUNK_SIZE 64

#ifdef BIG_SERVER
#define NUM_SUBTAPES 6400000
#else
#define NUM_SUBTAPES 640000
#endif
