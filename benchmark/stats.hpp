#include <functional>

void get_stats(std::function<void()> f, int warmup=20, int count=100);
