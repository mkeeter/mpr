#pragma once

#include <math_constants.h>

struct Deriv {
    __device__ inline Deriv() : v(make_float2(0.0f, 0.0f)) {}
    __device__ inline Deriv(float f) : v(make_float2(f, 0.0f)) {}
    __device__ inline Deriv(float v, float d) : v(make_float2(v, d)) {}
    __device__ inline float value() const { return v.x; }
    __device__ inline float deriv() const { return v.y; }
    float2 v;
};

#ifdef __CUDACC__
__device__ inline float value(const Deriv& x) {
    return x.value();
}

__device__ inline float value(const float& x) {
    return x;
}

__device__ inline float deriv(const Deriv& x) {
    return x.deriv();
}

__device__ inline float deriv(const float& x) {
    return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Deriv operator-(const Deriv& x) {
    return {-x.value(), -x.deriv()};
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Deriv operator+(const Deriv& x, const Deriv& y) {
    return {x.value() + y.value(), x.deriv() + y.deriv()};
}

__device__ inline Deriv operator+(const Deriv& x, const float& y) {
    return {x.value() + y, x.deriv()};
}

__device__ inline Deriv operator+(const float& y, const Deriv& x) {
    return x + y;
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Deriv operator*(const Deriv& x, const Deriv& y) {
    return {x.value() * y.value(),
            x.deriv() * y.value() + y.deriv() * x.value()}; // produce rule
}

__device__ inline Deriv operator*(const Deriv& x, const float& y) {
    return {x.value() * y, x.deriv() * y};
}

__device__ inline Deriv operator*(const float& x, const Deriv& y) {
    return y * x;
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Deriv operator/(const Deriv& x, const Deriv& y) {
    return {x.value() / y.value(),
            (y.value() * x.deriv() - x.value() * y.deriv()) /
                powf(y.value(), 2)};
}

__device__ inline Deriv operator/(const Deriv& x, const float& y) {
    return {x.value() / y, x.deriv() / y};
}


////////////////////////////////////////////////////////////////////////////////

__device__ inline Deriv min(const Deriv& x, const Deriv& y) {
    return (x.value() < y.value()) ? x : y;
}

__device__ inline Deriv min(const Deriv& x, const float& y) {
    return (x.value() < y) ? x : Deriv(y);
}

__device__ inline Deriv min(const float& x, const Deriv& y) {
    return min(y, x);
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Deriv max(const Deriv& x, const Deriv& y) {
    return (x.value() >= y.value()) ? x : y;
}

__device__ inline Deriv max(const Deriv& x, const float& y) {
    return (x.value() >= y) ? x : Deriv(y);
}

__device__ inline Deriv max(const float& x, const Deriv& y) {
    return max(y, x);
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Deriv square(const Deriv& x) {
    return {x.value() * x.value(), x.deriv() * x.value() * 2};
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Deriv operator-(const Deriv& x, const Deriv& y) {
    return {x.value() - y.value(), x.deriv() - y.deriv()};
}

__device__ inline Deriv operator-(const Deriv& x, const float& y) {
    return {x.value() - y, x.deriv()};
}

__device__ inline Deriv operator-(const float& x, const Deriv& y) {
    return {x - y.value(), -y.deriv()};
}

__device__ inline Deriv sqrt(const Deriv& x) {
    return {sqrt(x.value()), x.deriv() / (2 * sqrt(x.value()))};
}
#endif
