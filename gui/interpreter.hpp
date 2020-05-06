/*
Simple GUI to demonstrate the reference implementation of
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#pragma once
#include "libfive-guile.h"

struct Range {
    int start_row;
    int end_row;
    int start_col;
    int end_col;
};

struct Interpreter {
    Interpreter();
    void eval(std::string script);

    /*  Lots of miscellaneous Scheme objects, constructed once
     *  during init() so that we don't need to build them over
     *  and over again at runtime */
    SCM scm_eval_sandboxed;
    SCM scm_port_eof_p;
    SCM scm_valid_sym;
    SCM scm_syntax_error_sym;
    SCM scm_numerical_overflow_sym;

    SCM scm_syntax_error_fmt;
    SCM scm_numerical_overflow_fmt;
    SCM scm_other_error_fmt;
    SCM scm_result_fmt;
    SCM scm_in_function_fmt;

    bool result_valid = false;
    std::string result_str;
    std::string result_err_str;
    std::string result_err_stack;
    Range result_err_range;

    std::map<libfive::Tree::Id, libfive::Tree> shapes;
};
