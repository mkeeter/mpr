#pragma once

#include <math_constants.h>
#include "gpu_interval.hpp"

// Sources:
// Fast Reliable Interrogation of Procedurally Defined Implicit Surfaces Using Extended Revised Affine Arithmetic
// Self 􏰓Validated Numerical Methds and Applications
// https://www.shadertoy.com/view/4sV3zm

struct Affine {
    __device__ inline Affine() {}

    __device__ inline Affine(float f)
        : v0(f), d1(0.0f), d2(0.0f), d3(0.0f), err(0.0f)
    {}

    __device__ inline Affine(float v0, float d1, float d2, float d3, float err)
        : v0(v0), d1(d1), d2(d2), d3(d3), err(err)
    {}

    __device__ inline explicit Affine(const Interval& i)
        : v0(i.mid()),
          d1(0.0f), d2(0.0f), d3(0.0f),
          err(i.rad())
    {}

    __device__ inline static Affine X(const Interval& x)
    { return Affine(x.mid(), x.rad(), 0.0f, 0.0f, 0.0f); }

    __device__ inline static Affine Y(const Interval& y)
    { return Affine(y.mid(), 0.0f, y.rad(), 0.0f, 0.0f); }

    __device__ inline static Affine Z(const Interval& z)
    { return Affine(z.mid(), 0.0f, 0.0f, z.rad(), 0.0f); }

#ifdef __CUDACC__
    __device__ Interval as_interval() const {
        float lower = v0;
        float upper = v0;
#define CHECK(d) do {  \
            const float a = fabsf(d);   \
            lower = __fsub_rd(lower, a);    \
            upper = __fadd_ru(upper, a);    \
        } while(0)

        CHECK(d1);
        CHECK(d2);
        CHECK(d3);
        CHECK(err);
#undef CHECK
        return {lower, upper};
    }

    __device__ float upper() const {
        return as_interval().upper();
    }

    __device__ float lower() const {
        return as_interval().lower();
    }
#endif

    float v0 = 0.0f;
    float d1 = 0.0f; // X component of error
    float d2 = 0.0f; // Y component of error
    float d3 = 0.0f; // Z component of error
    float err = 0.0f; // non-linear error
};

#ifdef __CUDACC__
__device__ inline Affine operator-(const Affine& a) {
    return {-a.v0, -a.d1, -a.d2, -a.d3, a.err};
}

__device__ inline float lower(const Affine& a) {
    return a.lower();
}

__device__ inline float upper(const Affine& a) {
    return a.upper();
}

// General form of affine binary operations,
// where the function approximation is of the form
//      zeta + alpha * X + beta * Y +/- delta
__device__ inline Affine binary_op(const Affine& x, const Affine& y,
        const float alpha, const float beta, const float zeta,
        const float delta)
{
    return {alpha * x.v0 + beta * y.v0 + zeta,
            alpha * x.d1 + beta * y.d1,
            alpha * x.d2 + beta * y.d2,
            alpha * x.d3 + beta * y.d3,
            fabsf(alpha) * x.err + fabsf(beta) * y.err + delta};
}

// General form of affine unary operations,
// where the function approximation is of the form
//      zeta + alpha * X +/- delta
__device__ inline Affine unary_op(const Affine& x,
        const float alpha, const float zeta,
        const float delta)
{
    return {alpha * x.v0 + zeta,
            alpha * x.d1,
            alpha * x.d2,
            alpha * x.d3,
            fabsf(alpha) * x.err + delta};
}

// Min-range reciprocal
__device__ inline Affine reciprocal(const Affine& input) {
    // Stolfi, p. 70
    const Interval i = input.as_interval();
    const float a = fminf(fabsf(i.lower()), fabsf(i.upper()));
    const float b = fmaxf(fabsf(i.lower()), fabsf(i.upper()));
    const float alpha = -__fdiv_rd(1.0f, __fmul_ru(b, b));
    const float d_max = __fsub_ru(__fdiv_ru(1.0f, a), __fmul_ru(alpha, a));
    const float d_min = __fsub_rd(__fdiv_rd(1.0f, b), __fmul_rd(alpha, b));
    Interval d { d_min, d_max};
    float zeta = d.mid();
    if (i.lower() < 0.0f) {
        zeta = -zeta;
    }
    const float delta = d.rad();
    return unary_op(input, alpha, zeta, delta);
}

// Optimal sqrt
__device__ inline Affine sqrt(const Affine& input) {
    // Fryazinov, p. 9
    Interval i = input.as_interval();
    if (i.upper() < 0.0f) {
        return {CUDART_NAN_F, 0.0f, 0.0f, 0.0f, 0.0f};
    }
    // Clamp lower input to 0
    if (i.lower() < 0.0f) {
        i.v.x = 0.0f;
    }
    const float sq1 = sqrtf(i.lower());
    const float sq2 = sqrtf(i.upper());
    const float alpha = 1.0f / (sq1 + sq2);
    const float zeta = (sq1 + sq2) / 8.0f + 0.5f * sq1 * sq2 / (sq1 + sq2);
    double delta = (sq2 - sq1) * (sq2 - sq1) / (8.0f * (sq1 + sq2));
    return unary_op(input, alpha, zeta, delta);
}

__device__ inline Affine abs(const Affine& input) {
    const Interval i = input.as_interval();
    if (i.upper() <= 0.0f) {
        return -input;
    } else if (i.lower() >= 0.0f) {
        return input;
    }
    // TODO: optimize this more, e.g. from https://www.shadertoy.com/view/4sV3zm
    return Affine(abs(i));
}

__device__ inline Affine operator+(const Affine& a, const Affine& b) {
    return {a.v0 + b.v0,
            a.d1 + b.d1,
            a.d2 + b.d2,
            a.d3 + b.d3,
            a.err + b.err};
}

__device__ inline Affine operator+(const Affine& a, const float& b) {
    return {a.v0 + b,
            a.d1,
            a.d2,
            a.d3,
            a.err};
}

__device__ inline Affine operator+(const float& a, const Affine& b) {
    return b + a;
}

__device__ inline Affine operator*(const Affine& a, const Affine& b) {
    const float u = fabsf(a.d1) + fabsf(a.d2) + fabsf(a.d3);
    const float v = fabsf(b.d1) + fabsf(b.d2) + fabsf(b.d3);
    const float e_xy = a.err * b.err +
        b.err * (fabsf(a.v0) + u) +
        a.err * (fabsf(b.v0) + v)
        + u * v - 0.5f * (fabsf(a.d1 * b.d1) +
                          fabsf(a.d2 * b.d2) +
                          fabsf(a.d3 * b.d3));
    return {a.v0 * b.v0 + 0.5f * (a.d1 * b.d1 + a.d2 * b.d2 + a.d3 * b.d3),
            (a.v0 * b.d1 + a.d1 * b.v0),
            (a.v0 * b.d2 + a.d2 * b.v0),
            (a.v0 * b.d3 + a.d3 * b.v0),
            e_xy};
}

__device__ inline Affine square(const Affine& a) {
    return Affine(square(a.as_interval()));
    const float u = fabsf(a.d1) + fabsf(a.d2) + fabsf(a.d3);
    const float w = (a.d1 * a.d1) +
                    (a.d2 * a.d2) +
                    (a.d3 * a.d3) +
                    (a.err * a.err);
    const float e_xx = a.err * (a.err + 2.0f * (fabsf(a.v0) + u))
        - 0.5f * w;
    return {a.v0 * a.v0 + 0.5f * w,
            2.0f * a.v0 * a.d1,
            2.0f * a.v0 * a.d2,
            2.0f * a.v0 * a.d3,
            e_xx};
}

__device__ inline Affine operator*(const Affine& a, const float& b) {
    return {a.v0 * b, a.d1 * b, a.d2 * b, a.d3 * b, a.err * fabsf(b)};
}

__device__ inline Affine operator*(const float& a, const Affine& b) {
    return b * a;
}

__device__ inline Affine operator-(const Affine& a, const Affine& b) {
    return {a.v0 - b.v0,
            a.d1 - b.d1,
            a.d2 - b.d2,
            a.d3 - b.d3,
            a.err + b.err};
}

__device__ inline Affine operator-(const Affine& a, const float& b) {
    return {a.v0 - b,
            a.d1,
            a.d2,
            a.d3,
            a.err};
}

__device__ inline Affine operator-(const float& b, const Affine& a) {
    return {b - a.v0,
            -a.d1,
            -a.d2,
            -a.d3,
            a.err};
}

__device__ inline Affine operator/(const Affine& a, const Affine& b) {
    return a * reciprocal(b);
}

__device__ inline Affine operator/(const Affine& a, const float& b) {
    return a * Affine(1.0f/b);
}

__device__ inline Affine min(const Affine& a, const Affine& b) {
    const Interval ia = a.as_interval();
    const Interval ib = b.as_interval();
    if (ia.upper() <= ib.lower()) {
        return a;
    } else if (ib.upper() <= ia.lower()) {
        return b;
    }
    // Do fancier checking!
    bool a_always_below = true;
    bool b_always_below = true;
    for (unsigned i=0; i < 8 && (a_always_below || b_always_below); ++i) {
        // These are values before uncorrelated error terms are applied
        const float ai = a.v0 + ((i & 1) ? a.d1 : (-a.d1)) +
                                ((i & 2) ? a.d2 : (-a.d2)) +
                                ((i & 4) ? a.d3 : (-a.d3));
        const float bi = b.v0 + ((i & 1) ? b.d1 : (-b.d1)) +
                                ((i & 2) ? b.d2 : (-b.d2)) +
                                ((i & 4) ? b.d3 : (-b.d3));
        a_always_below &= (__fadd_ru(ai, a.err) < __fsub_rd(bi, b.err));
        b_always_below &= (__fadd_ru(bi, b.err) < __fsub_rd(ai, a.err));
    }
    if (a_always_below) {
        return a;
    } else if (b_always_below) {
        return b;
    }
    return Affine(min(ia, ib));
}

__device__ inline Affine max(const Affine& a, const Affine& b) {
    const Interval ia = a.as_interval();
    const Interval ib = b.as_interval();
    if (ia.lower() >= ib.upper()) {
        return a;
    } else if (ib.lower() >= ia.upper()) {
        return b;
    }
    // Do fancier checking!
    bool a_always_below = true;
    bool b_always_below = true;
    for (unsigned i=0; i < 8 && (a_always_below || b_always_below); ++i) {
        // These are values before uncorrelated error terms are applied
        const float ai = a.v0 + ((i & 1) ? a.d1 : (-a.d1)) +
                                ((i & 2) ? a.d2 : (-a.d2)) +
                                ((i & 4) ? a.d3 : (-a.d3));
        const float bi = b.v0 + ((i & 1) ? b.d1 : (-b.d1)) +
                                ((i & 2) ? b.d2 : (-b.d2)) +
                                ((i & 4) ? b.d3 : (-b.d3));
        a_always_below &= (__fadd_ru(ai, a.err) < __fsub_rd(bi, b.err));
        b_always_below &= (__fadd_ru(bi, b.err) < __fsub_rd(ai, a.err));
    }
    if (a_always_below) {
        return b;
    } else if (b_always_below) {
        return a;
    }
    return Affine(max(ia, ib));
}

// TODO
__device__ inline Affine asin(const Affine& a) {
    return Affine();
}
__device__ inline Affine acos(const Affine& a) {
    return Affine();
}
__device__ inline Affine atan(const Affine& a) {
    return Affine();
}
__device__ inline Affine sin(const Affine& a) {
    return Affine();
}
__device__ inline Affine cos(const Affine& a) {
    return Affine();
}
__device__ inline Affine log(const Affine& a) {
    return Affine();
}
__device__ inline Affine exp(const Affine& a) {
    return Affine();
}

#endif
