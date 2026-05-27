// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::c_void;
use std::fmt::Display;
use std::fmt::Formatter;
use std::ptr;

use crate::cpp;
use crate::cpp::duckdb_vx_expr_class;
use crate::duckdb::DDBString;
use crate::duckdb::ScalarFunction;
use crate::duckdb::ScalarFunctionRef;
use crate::duckdb::Value;
use crate::duckdb::ValueRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(Expression, cpp::duckdb_vx_expr, cpp::duckdb_vx_destroy_expr);

impl Display for ExpressionRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { cpp::duckdb_vx_expr_to_string(self.as_ptr()) };
        let cstr = unsafe { CStr::from_ptr(ptr) };
        let result = write!(f, "{}", cstr.to_string_lossy());
        unsafe { cpp::duckdb_free(ptr.cast_mut().cast()) };
        result
    }
}

impl ExpressionRef {
    pub fn as_class_id(&self) -> duckdb_vx_expr_class {
        unsafe { cpp::duckdb_vx_expr_get_class(self.as_ptr()) }
    }

    /// Match the subclass of the expression.
    pub fn as_class(&self) -> Option<ExpressionClass<'_>> {
        Some(
            match unsafe { cpp::duckdb_vx_expr_get_class(self.as_ptr()) } {
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF => {
                    let name = unsafe {
                        let ptr = cpp::duckdb_vx_expr_get_bound_column_ref_get_name(self.as_ptr());
                        DDBString::own(ptr.cast_mut())
                    };

                    ExpressionClass::BoundColumnRef(BoundColumnRef { name })
                }
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_CONSTANT => {
                    let value = unsafe {
                        Value::borrow(cpp::duckdb_vx_expr_bound_constant_get_value(self.as_ptr()))
                    };
                    ExpressionClass::BoundConstant(BoundConstant { value })
                }
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_CONJUNCTION => {
                    let mut out = cpp::duckdb_vx_expr_bound_conjunction {
                        children: ptr::null_mut(),
                        children_count: 0,
                        type_: cpp::DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                    };
                    unsafe {
                        cpp::duckdb_vx_expr_get_bound_conjunction(self.as_ptr(), &raw mut out)
                    };

                    let children =
                        unsafe { std::slice::from_raw_parts(out.children, out.children_count) };

                    ExpressionClass::BoundConjunction(BoundConjunction {
                        children,
                        op: out.type_,
                    })
                }
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COMPARISON => {
                    let mut out = cpp::duckdb_vx_expr_bound_comparison {
                        left: ptr::null_mut(),
                        right: ptr::null_mut(),
                        type_: cpp::DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                    };
                    unsafe {
                        cpp::duckdb_vx_expr_get_bound_comparison(self.as_ptr(), &raw mut out)
                    };

                    ExpressionClass::BoundComparison(BoundComparison {
                        left: unsafe { Expression::borrow(out.left) },
                        right: unsafe { Expression::borrow(out.right) },
                        op: out.type_,
                    })
                }
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_BETWEEN => {
                    let mut out = cpp::duckdb_vx_expr_bound_between {
                        input: ptr::null_mut(),
                        lower: ptr::null_mut(),
                        upper: ptr::null_mut(),
                        lower_inclusive: false,
                        upper_inclusive: false,
                    };
                    unsafe {
                        cpp::duckdb_vx_expr_get_bound_between(self.as_ptr(), &raw mut out);
                    }

                    ExpressionClass::BoundBetween(BoundBetween {
                        input: unsafe { Expression::borrow(out.input) },
                        lower: unsafe { Expression::borrow(out.lower) },
                        upper: unsafe { Expression::borrow(out.upper) },
                        lower_inclusive: out.lower_inclusive,
                        upper_inclusive: out.upper_inclusive,
                    })
                }
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_OPERATOR => {
                    let mut out = cpp::duckdb_vx_expr_bound_operator {
                        children: ptr::null_mut(),
                        children_count: 0,
                        type_: cpp::DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                    };
                    unsafe { cpp::duckdb_vx_expr_get_bound_operator(self.as_ptr(), &raw mut out) };

                    let children =
                        unsafe { std::slice::from_raw_parts(out.children, out.children_count) };

                    ExpressionClass::BoundOperator(BoundOperator {
                        children,
                        op: out.type_,
                    })
                }
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_FUNCTION => {
                    let mut out = cpp::duckdb_vx_expr_bound_function {
                        children: ptr::null_mut(),
                        children_count: 0,
                        scalar_function: ptr::null_mut(),
                        bind_info: ptr::null_mut(),
                    };
                    unsafe { cpp::duckdb_vx_expr_get_bound_function(self.as_ptr(), &raw mut out) };

                    let children =
                        unsafe { std::slice::from_raw_parts(out.children, out.children_count) };

                    ExpressionClass::BoundFunction(BoundFunction {
                        children,
                        scalar_function: unsafe { ScalarFunction::borrow(out.scalar_function) },
                        bind_info: out.bind_info,
                    })
                }
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_REF => {
                    ExpressionClass::BoundRef
                }
                _ => {
                    return None;
                }
            },
        )
    }
}

pub enum ExpressionClass<'a> {
    BoundColumnRef(BoundColumnRef),
    BoundConstant(BoundConstant<'a>),
    BoundComparison(BoundComparison<'a>),
    BoundConjunction(BoundConjunction<'a>),
    BoundBetween(BoundBetween<'a>),
    BoundOperator(BoundOperator<'a>),
    BoundFunction(BoundFunction<'a>),
    /// Column inside ExpressionFilter for expression pushed down to Vortex.
    BoundRef,
}

pub struct BoundColumnRef {
    pub name: DDBString,
}

pub struct BoundConstant<'a> {
    pub value: &'a ValueRef,
}

pub struct BoundComparison<'a> {
    pub left: &'a ExpressionRef,
    pub right: &'a ExpressionRef,
    pub op: cpp::DUCKDB_VX_EXPR_TYPE,
}

pub struct BoundBetween<'a> {
    pub input: &'a ExpressionRef,
    pub lower: &'a ExpressionRef,
    pub upper: &'a ExpressionRef,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

pub struct BoundConjunction<'a> {
    children: &'a [cpp::duckdb_vx_expr],
    pub op: cpp::DUCKDB_VX_EXPR_TYPE,
}

impl<'a> BoundConjunction<'a> {
    /// Returns the children expressions of the bound conjunction.
    pub fn children(&self) -> impl Iterator<Item = &'a ExpressionRef> + 'a {
        self.children
            .iter()
            .map(|&child| unsafe { Expression::borrow(child) })
    }
}

pub struct BoundOperator<'a> {
    children: &'a [cpp::duckdb_vx_expr],
    pub op: cpp::DUCKDB_VX_EXPR_TYPE,
}

impl<'a> BoundOperator<'a> {
    /// Returns the children expressions of the bound operator.
    pub fn children(&self) -> impl Iterator<Item = &'a ExpressionRef> + 'a {
        self.children
            .iter()
            .map(|&child| unsafe { Expression::borrow(child) })
    }
}

pub struct BoundFunction<'a> {
    children: &'a [cpp::duckdb_vx_expr],
    pub scalar_function: &'a ScalarFunctionRef,
    pub bind_info: *const c_void,
}

impl<'a> BoundFunction<'a> {
    /// Returns the children expressions of the bound function.
    pub fn children(&self) -> impl Iterator<Item = &'a ExpressionRef> + 'a {
        self.children
            .iter()
            .map(|&child| unsafe { Expression::borrow(child) })
    }
}
