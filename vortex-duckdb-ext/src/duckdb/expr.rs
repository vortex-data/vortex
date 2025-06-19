use std::ffi::{CStr, c_void};
use std::fmt::{Display, Formatter};
use std::ptr;

use crate::cpp::duckdb_vx_expr_class;
use crate::duckdb::{ScalarFunction, Value};
use crate::{cpp, wrapper};

wrapper!(Expression, cpp::duckdb_vx_expr, cpp::duckdb_vx_destroy_expr);

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { cpp::duckdb_vx_expr_to_string(self.as_ptr()) };
        let cstr = unsafe { CStr::from_ptr(ptr) };
        let result = write!(f, "{}", cstr.to_string_lossy());
        unsafe { cpp::duckdb_free(ptr.cast_mut().cast()) };
        result
    }
}

impl Expression {
    pub fn as_class_id(&self) -> duckdb_vx_expr_class {
        unsafe { cpp::duckdb_vx_expr_get_class(self.as_ptr()) }
    }

    /// Match the subclass of the expression.
    pub fn as_class(&self) -> Option<ExpressionClass> {
        Some(
            match unsafe { cpp::duckdb_vx_expr_get_class(self.as_ptr()) } {
                cpp::DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF => {
                    let ptr =
                        unsafe { cpp::duckdb_vx_expr_get_bound_column_ref_get_name(self.as_ptr()) };
                    ExpressionClass::BoundColumnRef(BoundColumnRef {
                        name: unsafe { CStr::from_ptr(ptr) },
                    })
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
                    unsafe { cpp::duckdb_vx_expr_get_bound_conjunction(self.as_ptr(), &mut out) };

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
                    unsafe { cpp::duckdb_vx_expr_get_bound_comparison(self.as_ptr(), &mut out) };

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
                        cpp::duckdb_vx_expr_get_bound_between(self.as_ptr(), &mut out);
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
                    unsafe { cpp::duckdb_vx_expr_get_bound_operator(self.as_ptr(), &mut out) };

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
                    unsafe { cpp::duckdb_vx_expr_get_bound_function(self.as_ptr(), &mut out) };

                    let children =
                        unsafe { std::slice::from_raw_parts(out.children, out.children_count) };

                    ExpressionClass::BoundFunction(BoundFunction {
                        children,
                        scalar_function: unsafe { ScalarFunction::borrow(out.scalar_function) },
                        bind_info: out.bind_info,
                    })
                }
                _ => {
                    return None;
                }
            },
        )
    }
}

pub enum ExpressionClass<'a> {
    BoundColumnRef(BoundColumnRef<'a>),
    BoundConstant(BoundConstant),
    BoundComparison(BoundComparison),
    BoundConjunction(BoundConjunction<'a>),
    BoundBetween(BoundBetween),
    BoundOperator(BoundOperator<'a>),
    BoundFunction(BoundFunction<'a>),
}

pub struct BoundColumnRef<'a> {
    pub name: &'a CStr,
}

pub struct BoundConstant {
    pub value: Value,
}

pub struct BoundComparison {
    pub left: Expression,
    pub right: Expression,
    pub op: cpp::DUCKDB_VX_EXPR_TYPE,
}

pub struct BoundBetween {
    pub input: Expression,
    pub lower: Expression,
    pub upper: Expression,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

pub struct BoundConjunction<'a> {
    children: &'a [cpp::duckdb_vx_expr],
    pub op: cpp::DUCKDB_VX_EXPR_TYPE,
}

impl BoundConjunction<'_> {
    /// Returns the children expressions of the bound operator.
    pub fn children(&self) -> impl Iterator<Item = Expression> {
        self.children
            .iter()
            .map(|&child| unsafe { Expression::borrow(child) })
    }
}

pub struct BoundOperator<'a> {
    children: &'a [cpp::duckdb_vx_expr],
    pub op: cpp::DUCKDB_VX_EXPR_TYPE,
}

impl BoundOperator<'_> {
    /// Returns the children expressions of the bound operator.
    pub fn children(&self) -> impl Iterator<Item = Expression> {
        self.children
            .iter()
            .map(|&child| unsafe { Expression::borrow(child) })
    }
}

pub struct BoundFunction<'a> {
    children: &'a [cpp::duckdb_vx_expr],
    pub scalar_function: ScalarFunction,
    pub bind_info: *const c_void,
}

impl BoundFunction<'_> {
    /// Returns the children expressions of the bound function.
    pub fn children(&self) -> impl Iterator<Item = Expression> {
        self.children
            .iter()
            .map(|&child| unsafe { Expression::borrow(child) })
    }
}
