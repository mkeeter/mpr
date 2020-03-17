#pragma once

#include <math_constants.h>

namespace libfive {
namespace cuda {

struct Interval {
    __device__ inline Interval() { /* YOLO */ }
    __device__ inline explicit Interval(float f) : v(make_float2(f, f)) {}
    __device__ inline Interval(float a, float b) : v(make_float2(a, b)) {}
    __device__ inline float upper() const { return v.y; }
    __device__ inline float lower() const { return v.x; }

    __device__ inline static Interval X(const Interval& x) { return x; }
    __device__ inline static Interval Y(const Interval& y) { return y; }
    __device__ inline static Interval Z(const Interval& z) { return z; }

#ifdef __CUDACC__
    __device__ inline float mid() const {
        return __fdiv_ru(lower(), 2.0f) + __fdiv_rd(upper(), 2.0f);
    }
    __device__ inline float rad() const {
        const float m = mid();
        return fmaxf(__fsub_ru(m, lower()), __fsub_ru(upper(), m));
    }
    __device__ inline float width() const {
        return __fsub_ru(upper(), lower());
    }
#endif

    float2 v;
};

#ifdef __CUDACC__

__device__ inline float upper(const Interval& x) {
    return x.upper();
}

__device__ inline float upper(const float& x) {
    return x;
}

__device__ inline float lower(const Interval& x) {
    return x.lower();
}

__device__ inline float lower(const float& x) {
    return x;
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Interval operator-(const Interval& x) {
    return {-x.upper(), -x.lower()};
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Interval operator+(const Interval& x, const Interval& y) {
    return {__fadd_rd(x.lower(), y.lower()), __fadd_ru(x.upper(), y.upper())};
}

__device__ inline Interval operator+(const Interval& x, const float& y) {
    return {__fadd_rd(x.lower(), y), __fadd_ru(x.upper(), y)};
}

__device__ inline Interval operator+(const float& y, const Interval& x) {
    return x + y;
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Interval operator*(const Interval& x, const Interval& y) {
    if (x.lower() < 0.0f) {
        if (x.upper() > 0.0f) {
            if (y.lower() < 0.0f) {
                if (y.upper() > 0.0f) { // M * M
                    return {fminf(__fmul_rd(x.lower(), y.upper()),
                                  __fmul_rd(x.upper(), y.lower())),
                            fmaxf(__fmul_ru(x.lower(), y.lower()),
                                  __fmul_ru(x.upper(), y.upper()))};
                } else { // M * N
                    return {__fmul_rd(x.upper(), y.lower()),
                            __fmul_ru(x.lower(), y.lower())};
                }
            } else {
                if (y.upper() > 0.0f) { // M * P
                    return {__fmul_rd(x.lower(), y.upper()),
                            __fmul_ru(x.upper(), y.upper())};
                } else { // M * Z
                    return {0.0f, 0.0f};
                }
            }
        } else {
            if (y.lower() < 0.0f) {
                if (y.upper() > 0.0f) { // N * M
                    return {__fmul_rd(x.lower(), y.upper()),
                            __fmul_ru(x.lower(), y.lower())};
                } else { // N * N
                    return {__fmul_rd(x.upper(), y.upper()),
                            __fmul_ru(x.lower(), y.lower())};
                }
            } else {
                if (y.upper() > 0.0f) { // N * P
                    return {__fmul_rd(x.lower(), y.upper()),
                            __fmul_ru(x.upper(), y.lower())};
                } else { // N * Z
                    return {0.0f, 0.0f};
                }
            }
        }
    } else {
        if (x.upper() > 0.0f) {
            if (y.lower() < 0.0f) {
                if (y.upper() > 0.0f) { // P * M
                    return {__fmul_rd(x.upper(), y.lower()),
                            __fmul_ru(x.upper(), y.upper())};
                } else {// P * N
                    return {__fmul_rd(x.upper(), y.lower()),
                            __fmul_ru(x.lower(), y.upper())};
                }
            } else {
                if (y.upper() > 0.0f) { // P * P
                    return {__fmul_rd(x.lower(), y.lower()),
                            __fmul_ru(x.upper(), y.upper())};
                } else {// P * Z
                    return {0.0f, 0.0f};
                }
            }
        } else { // Z * ?
            return {0.0f, 0.0f};
        }
    }
}

__device__ inline Interval operator*(const Interval& x, const float& y) {
    if (y < 0.0f) {
        return {__fmul_rd(x.upper(), y), __fmul_ru(x.lower(), y)};
    } else {
        return {__fmul_rd(x.lower(), y), __fmul_ru(x.upper(), y)};
    }
}

__device__ inline Interval operator*(const float& x, const Interval& y) {
    return y * x;
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Interval operator/(const Interval& x, const Interval& y) {
    if (y.lower() <= 0.0f && y.upper() >= 0.0f) {
        return {-CUDART_INF_F, CUDART_INF_F};
    } else if (x.upper() < 0.0f) {
        if (y.upper() < 0.0f) {
            return { __fdiv_rd(x.upper(), y.lower()),
                     __fdiv_ru(x.lower(), y.upper()) };
        } else {
            return { __fdiv_rd(x.lower(), y.lower()),
                     __fdiv_ru(x.upper(), y.upper()) };
        }
    } else if (x.lower() < 0.0f) {
        if (y.upper() < 0.0f) {
            return { __fdiv_rd(x.upper(), y.upper()),
                     __fdiv_ru(x.lower(), y.upper()) };
        } else {
            return { __fdiv_rd(x.lower(), y.lower()),
                     __fdiv_ru(x.upper(), y.lower()) };
        }
    } else {
        if (y.upper() < 0.0f) {
            return { __fdiv_rd(x.upper(), y.upper()),
                     __fdiv_ru(x.lower(), y.lower()) };
        } else {
            return { __fdiv_rd(x.lower(), y.upper()),
                     __fdiv_ru(x.upper(), y.lower()) };
        }
    }
}

__device__ inline Interval operator/(const Interval& x, const float& y) {
    if (y < 0.0f) {
        return { __fdiv_rd(x.upper(), y), __fdiv_ru(x.lower(), y) };
    } else if (y > 0.0f) {
        return { __fdiv_rd(x.lower(), y), __fdiv_ru(x.upper(), y) };
    } else {
        return {-CUDART_INF_F, CUDART_INF_F};
    }
}

__device__ inline Interval operator/(const float& x, const Interval& y) {
    return Interval(x) / y;
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Interval min(const Interval& x, const Interval& y, int& choice) {
    if (x.upper() < y.lower()) {
        choice = 1;
        return x;
    } else if (y.upper() < x.lower()) {
        choice = 2;
        return y;
    }
    return {fminf(x.lower(), y.lower()), fminf(x.upper(), y.upper())};
}

__device__ inline Interval min(const Interval& x, const float& y, int& choice) {
    if (x.upper() < y) {
        choice = 1;
        return x;
    } else if (y < x.lower()) {
        choice = 2;
        return Interval(y);
    }
    return {fminf(x.lower(), y), fminf(x.upper(), y)};
}

__device__ inline Interval min(const float& x, const Interval& y, int& choice) {
    if (x < y.lower()) {
        choice = 1;
        return Interval(x);
    } else if (y.upper() < x) {
        choice = 2;
        return y;
    }
    return {fminf(x, y.lower()), fminf(x, y.upper())};
}

__device__ inline Interval min(const float& x, const float& y, int& choice) {
    if (x < y) {
        choice = 1;
        return Interval(x);
    } else if (y < x) {
        choice = 2;
        return Interval(y);
    }
    return Interval(x);
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Interval max(const Interval& x, const Interval& y, int& choice) {
    if (x.lower() > y.upper()) {
        choice = 1;
        return x;
    } else if (y.lower() > x.upper()) {
        choice = 2;
        return y;
    }
    return {fmaxf(x.lower(), y.lower()), fmaxf(x.upper(), y.upper())};
}

__device__ inline Interval max(const Interval& x, const float& y, int& choice) {
    if (x.lower() > y) {
        choice = 1;
        return x;
    } else if (y > x.upper()) {
        choice = 2;
        return Interval(y);
    }
    return {fmaxf(x.lower(), y), fmaxf(x.upper(), y)};
}

__device__ inline Interval max(const float& x, const Interval& y, int& choice) {
    if (x > y.upper()) {
        choice = 1;
        return Interval(x);
    } else if (y.lower() > x) {
        choice = 2;
        return y;
    }
    return {fmaxf(x, y.lower()), fmaxf(x, y.upper())};
}

__device__ inline Interval max(const float& x, const float& y, int& choice) {
    if (x > y) {
        choice = 1;
        return Interval(x);
    } else if (y > x) {
        choice = 2;
        return Interval(y);
    }
    return Interval(x);
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Interval square(const Interval& x) {
    if (x.upper() < 0.0f) {
        return {__fmul_rd(x.upper(), x.upper()), __fmul_ru(x.lower(), x.lower())};
    } else if (x.lower() > 0.0f) {
        return {__fmul_rd(x.lower(), x.lower()), __fmul_ru(x.upper(), x.upper())};
    } else if (-x.lower() > x.upper()) {
        return {0.0f, __fmul_ru(x.lower(), x.lower())};
    } else {
        return {0.0f, __fmul_ru(x.upper(), x.upper())};
    }
}

__device__ inline Interval abs(const Interval& x) {
    if (x.lower() >= 0.0f) {
        return x;
    } else if (x.upper() < 0.0f) {
        return -x;
    } else {
        return {0.0f, fmaxf(-x.lower(), x.upper())};
    }
}

__device__ inline float square(const float& x) {
    return x * x;
}

////////////////////////////////////////////////////////////////////////////////

__device__ inline Interval operator-(const Interval& x, const Interval& y) {
    return {__fsub_rd(x.lower(), y.upper()), __fsub_ru(x.upper(), y.lower())};
}

__device__ inline Interval operator-(const Interval& x, const float& y) {
    return {__fsub_rd(x.lower(), y), __fsub_ru(x.upper(), y)};
}

__device__ inline Interval operator-(const float& x, const Interval& y) {
    return {__fsub_rd(x, y.upper()), __fsub_ru(x, y.lower())};
}

__device__ inline Interval sqrt(const Interval& x) {
    if (x.upper() < 0.0f) {
        return {CUDART_NAN_F, CUDART_NAN_F};
    } else if (x.lower() <= 0.0f) {
        return {0.0f, __fsqrt_ru(x.upper())};
    } else {
        return {__fsqrt_rd(x.lower()), __fsqrt_ru(x.upper())};
    }
}

__device__ inline Interval acos(const Interval& x) {
    if (x.upper() < -1.0f || x.lower() > 1.0f) {
        return {CUDART_NAN_F, CUDART_NAN_F};
    } else {
        // Use double precision, since there aren't _ru / _rd primitives
        return {__double2float_rd(::acos(x.upper())),
                __double2float_ru(::acos(x.lower()))};
    }
}

__device__ inline Interval asin(const Interval& x) {
    if (x.upper() < -1.0f || x.lower() > 1.0f) {
        return {CUDART_NAN_F, CUDART_NAN_F};
    } else {
        // Use double precision, since there aren't _ru / _rd primitives
        return {__double2float_rd(::asin(x.lower())),
                __double2float_ru(::asin(x.upper()))};
    }
}

__device__ inline Interval atan(const Interval& x) {
    // Use double precision, since there aren't _ru / _rd primitives
    return {__double2float_rd(::atan(x.lower())),
            __double2float_ru(::atan(x.upper()))};
}

__device__ inline Interval exp(const Interval& x) {
    // Use double precision, since there aren't _ru / _rd primitives
    return {__double2float_rd(::exp(x.lower())),
            __double2float_ru(::exp(x.upper()))};
}

__device__ inline Interval fmod(const Interval& x, const Interval& y) {
    // Caveats from the Boost Interval library also apply here:
    //  This is only useful for clamping trig functions
    const float yb = x.lower() < 0.0f ? y.lower() : y.upper();
    const float n = floorf(__fdiv_rd(x.lower(), yb));
    return x - n * y;
}

__device__ inline Interval cos(const Interval& x) {
    static const float pi_f_l = 13176794.0f/(1<<22);
    static const float pi_f_u = 13176795.0f/(1<<22);

    const Interval pi{pi_f_l, pi_f_u};
    const Interval pi2 = pi * 2.0f;

    return Interval{-1.0f, 1.0f};

    Interval tmp = fmod(x, pi2);

    // We are covering a full period!
    if (tmp.width() >= pi2.lower()) {
        return Interval{-1.0f, 1.0f};
    }

    if (tmp.lower() >= pi.upper()) {
        return -cos(tmp - pi);
    }

    // Use double precision, since there aren't _ru / _rd primitives
    const float l = tmp.lower();
    const float u = tmp.lower();
    if (u <= pi.lower()) {
        return {__double2float_rd(::cos(u)), __double2float_ru(::cos(l))};
    } else if (u <= pi2.lower()) {
        return {-1.0f, __double2float_ru(::cos(fminf(__fsub_rd(pi2.lower(), u), l)))};
    } else {
        return {-1.0f, 1.0f};
    }
}

__device__ inline Interval sin(const Interval& x) {
    return cos(x - Interval{M_PI, M_PI} / 2.0f);
}

__device__ inline Interval log(const Interval& x) {
    if (x.upper() < 0.0f) {
        return {CUDART_NAN_F, CUDART_NAN_F};
    } else if (x.lower() <= 0.0f) {
        return {0.0f, __double2float_ru(::log(x.upper()))};
    } else {
        return {__double2float_rd(::log(x.lower())),
                __double2float_ru(::log(x.upper()))};
    }
}
#endif

}   // namespace cuda
}   // namespace libfive
