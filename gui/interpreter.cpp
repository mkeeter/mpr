/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include "interpreter.hpp"

Interpreter::Interpreter() {
    // Initialize libfive-guile bindings
    scm_init_guile();
    scm_init_libfive_modules();
    scm_c_use_module("libfive kernel");

    scm_eval_sandboxed = scm_c_eval_string(R"(
(use-modules (libfive sandbox))
eval-sandboxed
)");
    scm_syntax_error_sym = scm_from_utf8_symbol("syntax-error");
    scm_numerical_overflow_sym = scm_from_utf8_symbol("numerical-overflow");
    scm_valid_sym = scm_from_utf8_symbol("valid");
    scm_result_fmt = scm_from_locale_string("~S");
    scm_other_error_fmt = scm_from_locale_string("~A: ~A");
    scm_in_function_fmt = scm_from_locale_string("In function ~A:\n~A");
    scm_syntax_error_fmt = scm_from_locale_string("~A: ~A in form ~A");
    scm_numerical_overflow_fmt = scm_from_locale_string("~A: ~A in ~A");

    // Protect all of our interpreter vars from garbage collection
    for (auto s : {scm_eval_sandboxed, scm_valid_sym,
                   scm_syntax_error_sym, scm_numerical_overflow_sym,
                   scm_result_fmt, scm_syntax_error_fmt,
                   scm_numerical_overflow_fmt, scm_other_error_fmt,
                   scm_in_function_fmt})
    {
        scm_permanent_object(s);
    }
}

void Interpreter::eval(std::string script)
{
    auto result = scm_call_1(scm_eval_sandboxed,
            scm_from_locale_string(script.c_str()));

    //  Loop through the whole result list, looking for an invalid clause
    result_valid = true;
    for (auto r = result; !scm_is_null(r) && result_valid; r = scm_cdr(r)) {
        result_valid &= scm_is_eq(scm_caar(r), scm_valid_sym);
    }

    // If there is at least one result, then we'll convert the last one
    // into a string (with special cases for various error forms)
    auto last = scm_is_null(result) ? nullptr
                                    : scm_cdr(scm_car(scm_last_pair(result)));
    if (!result_valid) {
        /* last = '(before after key params) */
        auto before = scm_car(last);
        auto after = scm_cadr(last);
        auto key = scm_caddr(last);
        auto params = scm_cadddr(last);

        auto _stack = scm_car(scm_cddddr(last));
        SCM _str = nullptr;

        if (scm_is_eq(key, scm_syntax_error_sym)) {
            _str = scm_simple_format(SCM_BOOL_F, scm_syntax_error_fmt,
                   scm_list_3(key, scm_cadr(params), scm_cadddr(params)));
        } else if (scm_is_eq(key, scm_numerical_overflow_sym)) {
            _str = scm_simple_format(SCM_BOOL_F, scm_numerical_overflow_fmt,
                   scm_list_3(key, scm_cadr(params), scm_car(params)));
        } else {
            _str = scm_simple_format(SCM_BOOL_F, scm_other_error_fmt,
                   scm_list_2(key, scm_simple_format(
                        SCM_BOOL_F, scm_cadr(params), scm_caddr(params))));
        }
        if (!scm_is_false(scm_car(params))) {
            _str = scm_simple_format(SCM_BOOL_F, scm_in_function_fmt,
                                     scm_list_2(scm_car(params), _str));
        }
        auto str = scm_to_locale_string(_str);
        auto stack = scm_to_locale_string(_stack);

        result_err_str = std::string(str);
        result_err_stack = std::string(stack);
        result_err_range = {scm_to_int(scm_car(before)),
                            scm_to_int(scm_car(after)),
                            scm_to_int(scm_cdr(before)),
                            scm_to_int(scm_cdr(after))};

        free(str);
        free(stack);
    } else if (last) {
        char* str = nullptr;
        if (scm_to_int64(scm_length(last)) == 1) {
            auto str = scm_to_locale_string(
                    scm_simple_format(SCM_BOOL_F, scm_result_fmt,
                                      scm_list_1(scm_car(last))));

            result_str = std::string(str);
        } else {
            auto str = scm_to_locale_string(
                    scm_simple_format(SCM_BOOL_F, scm_result_fmt,
                                      scm_list_1(last)));

            result_str = std::string("(values") + str + ")";
        }
        free(str);
    } else {
        result_str = "#<eof>";
    }

    // Then iterate over the results, picking out shapes
    if (result_valid) {
        // Initialize variables and their textual positions
        std::map<libfive::Tree::Id, float> vars;
        std::map<libfive::Tree::Id, Range> var_pos;

        {   // Walk through the global variable map
            auto vs = scm_c_eval_string(R"(
                (use-modules (libfive sandbox))
                (hash-map->list (lambda (k v) v) vars) )");

            for (auto v = vs; !scm_is_null(v); v = scm_cdr(v))
            {
                auto data = scm_cdar(v);
                auto id = static_cast<libfive::Tree::Id>(
                        libfive_tree_id(scm_get_tree(scm_car(data))));
                auto value = scm_to_double(scm_cadr(data));
                vars[id] = value;

                auto vp = scm_caddr(data);
                var_pos[id] = {scm_to_int(scm_car(vp)), 0,
                               scm_to_int(scm_cadr(vp)),
                               scm_to_int(scm_caddr(vp))};
            }
        }

        // Then walk through the result list, picking out trees
        shapes.clear();
        while (!scm_is_null(result)) {
            for (auto r = scm_cdar(result); !scm_is_null(r); r = scm_cdr(r)) {
                if (scm_is_shape(scm_car(r))) {
                    auto t = *scm_get_tree(scm_car(r));
                    shapes.insert({t.id(), t});
                }
            }
            result = scm_cdr(result);
        }

        // Do something with shapes
        // Do something with vars
    }
}
