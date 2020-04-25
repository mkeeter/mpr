/*
Reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>

#include "stats.hpp"

double get_stats(std::function<void()> f, int warmup, int count) {
    // Warm up
    for (int i=0; i < warmup; ++i) {
        f();
    }
    std::vector<double> times_ms;
    for (int i=0; i < count; ++i) {
        using std::chrono::steady_clock;
        using std::chrono::duration_cast;
        using std::chrono::nanoseconds;
        auto start_gpu = steady_clock::now();
        f();
        auto end_gpu = steady_clock::now();
        times_ms.push_back(
                duration_cast<nanoseconds>(end_gpu - start_gpu).count() / 1e6);
    }
    double mean = 0;
    for (auto& b : times_ms) {
        mean += b;
    }
    mean /= times_ms.size();
    double stdev = 0;
    for (auto& b : times_ms) {
        stdev += std::pow(b - mean, 2);
    }
    stdev = sqrt(stdev / (times_ms.size() - 1));
    std::cout << mean << " " << stdev << "\n";
    return mean;
}
