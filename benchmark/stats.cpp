#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>

#include "stats.hpp"

void get_stats(std::function<void()> f, int warmup, int count) {
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
}
