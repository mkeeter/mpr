#pragma once

#include <math_constants.h>

struct Interval {
    float lower;
    float upper;

    __device__ inline Interval operator+(const Interval& other) {
        return {__fadd_rd(lower, other.lower), __fadd_ru(upper, other.upper)};
    }

    __device__ inline Interval operator*(const Interval& other) {
        if (lower < 0.0f) {
            if (upper > 0.0f) {
                if (other.lower < 0.0f) {
                    if (other.upper > 0.0f) { // M * M
                        return {fminf(__fmul_rd(lower, other.upper),
                                      __fmul_rd(upper, other.lower)),
                                fmaxf(__fmul_ru(lower, other.lower),
                                      __fmul_ru(upper, other.upper))};
                    } else { // M * N
                        return {__fmul_rd(upper, other.lower),
                                __fmul_ru(lower, other.lower)};
                    }
                } else {
                    if (other.upper > 0.0f) { // M * P
                        return {__fmul_rd(lower, other.upper),
                                __fmul_ru(upper, other.upper)};
                    } else { // M * Z
                        return {0.0f, 0.0f};
                    }
                }
            } else {
                if (other.lower < 0.0f) {
                    if (other.upper > 0.0f) { // N * M
                        return {__fmul_rd(lower, other.upper),
                                __fmul_ru(lower, other.lower)};
                    } else { // N * N
                        return {__fmul_rd(upper, other.upper),
                                __fmul_ru(lower, other.lower)};
                    }
                } else {
                    if (other.upper > 0.0f) { // N * P
                        return {__fmul_rd(lower, other.upper),
                                __fmul_ru(upper, other.lower)};
                    } else { // N * Z
                        return {0.0f, 0.0f};
                    }
                }
            }
        } else {
            if (upper > 0.0f) {
                if (other.lower < 0.0f) {
                    if (other.upper > 0.0f) { // P * M
                        return {__fmul_rd(upper, other.lower),
                                __fmul_ru(upper, other.upper)};
                    } else {// P * N
                        return {__fmul_rd(upper, other.lower),
                                __fmul_ru(lower, other.upper)};
                    }
                } else {
                    if (other.upper > 0.0f) { // P * P
                        return {__fmul_rd(lower, other.lower),
                                __fmul_ru(upper, other.upper)};
                    } else {// P * Z
                        return {0.0f, 0.0f};
                    }
                }
            } else { // Z * ?
                return {0.0f, 0.0f};
            }
        }
    }

    __device__ inline Interval operator/(const Interval& other) {
        if (upper < 0.0f) {
            if (other.upper < 0.0f) {
                return { __fdiv_rd(upper, other.lower),
                         __fdiv_ru(lower, other.upper) };
            } else {
                return { __fdiv_rd(lower, other.lower),
                         __fdiv_ru(upper, other.upper) };
            }
        } else if (lower < 0.0f) {
            if (other.upper < 0.0f) {
                return { __fdiv_rd(upper, other.upper),
                         __fdiv_ru(lower, other.upper) };
            } else {
                return { __fdiv_rd(lower, other.lower),
                         __fdiv_ru(upper, other.lower) };
            }
        } else {
            if (other.upper < 0.0f) {
                return { __fdiv_rd(upper, other.upper),
                         __fdiv_ru(lower, other.lower) };
            } else {
                return { __fdiv_rd(lower, other.upper),
                         __fdiv_ru(upper, other.lower) };
            }
        }
    }

    __device__ inline Interval min(const Interval& other) {
        return {fminf(lower, other.lower), fminf(upper, other.upper)};
    }

    __device__ inline Interval max(const Interval& other) {
        return {fmaxf(lower, other.lower), fmaxf(upper, other.upper)};
    }

    __device__ inline Interval operator-(const Interval& other) {
        return {__fsub_rd(lower, other.upper), __fsub_ru(upper, other.lower)};
    }

    __device__ inline Interval square() const {
        if (upper < 0.0f) {
            return {__fmul_rd(upper, upper), __fmul_ru(lower, lower)};
        } else if (lower > 0.0f) {
            return {__fmul_rd(lower, lower), __fmul_ru(upper, upper)};
        } else if (-lower > upper) {
            return {0.0f, __fmul_ru(lower, lower)};
        } else {
            return {0.0f, __fmul_ru(upper, upper)};
        }
    }

    __device__ inline Interval sqrt() const {
        if (upper < 0.0f) {
            return {CUDART_NAN_F, CUDART_NAN_F};
        } else if (lower <= 0.0f) {
            return {0.0f, __fsqrt_ru(upper)};
        } else {
            return {__fsqrt_rd(lower), __fsqrt_ru(upper)};
        }
    }

    __device__ inline Interval operator-() const {
        return {-upper, -lower};
    }
};

