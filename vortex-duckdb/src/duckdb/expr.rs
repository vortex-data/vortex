// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{CStr, CString, c_void};
use std::fmt::{Display, Formatter};
use std::ptr;

use vortex::error::{VortexResult, vortex_bail, vortex_err};

use crate::cpp::*;
use crate::duckdb::{ScalarFunction, Value};
use crate::{cpp, duckdb, wrapper};

wrapper!(Expression, duckdb_vx_expr, duckdb_vx_destroy_expr);

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { duckdb_vx_expr_to_string(self.as_ptr()) };
        let cstr = unsafe { CStr::from_ptr(ptr) };
        let result = write!(f, "{}", cstr.to_string_lossy());
        unsafe { duckdb_free(ptr.cast_mut().cast()) };
        result
    }
}

impl Expression {
    pub fn as_class_id(&self) -> duckdb_vx_expr_class {
        unsafe { duckdb_vx_expr_get_class(self.as_ptr()) }
    }

    /// Match the subclass of the expression.
    pub fn as_class(&self) -> Option<ExpressionClass<'_>> {
        Some(
            match unsafe { duckdb_vx_expr_get_class(self.as_ptr()) } {
                DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF => {
                    let ptr =
                        unsafe { duckdb_vx_expr_get_bound_column_ref_get_name(self.as_ptr()) };

                    ExpressionClass::BoundColumnRef(BoundColumnRef {
                        name: duckdb::string::String::from_ptr(ptr),
                    })
                }
                DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_CONSTANT => {
                    let value = unsafe {
                        Value::borrow(duckdb_vx_expr_bound_constant_get_value(self.as_ptr()))
                    };
                    ExpressionClass::BoundConstant(BoundConstant { value })
                }
                DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_CONJUNCTION => {
                    let mut out = duckdb_vx_expr_bound_conjunction {
                        children: ptr::null_mut(),
                        children_count: 0,
                        type_: DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                    };
                    unsafe {
                        duckdb_vx_expr_get_bound_conjunction(self.as_ptr(), &raw mut out)
                    };

                    let children =
                        unsafe { std::slice::from_raw_parts(out.children, out.children_count) };

                    ExpressionClass::BoundConjunction(BoundConjunction {
                        children,
                        op: out.type_,
                    })
                }
                DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COMPARISON => {
                    let mut out = duckdb_vx_expr_bound_comparison {
                        left: ptr::null_mut(),
                        right: ptr::null_mut(),
                        type_: DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                    };
                    unsafe {
                        duckdb_vx_expr_get_bound_comparison(self.as_ptr(), &raw mut out)
                    };

                    ExpressionClass::BoundComparison(BoundComparison {
                        left: unsafe { Expression::borrow(out.left) },
                        right: unsafe { Expression::borrow(out.right) },
                        op: out.type_,
                    })
                }
                DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_BETWEEN => {
                    let mut out = duckdb_vx_expr_bound_between {
                        input: ptr::null_mut(),
                        lower: ptr::null_mut(),
                        upper: ptr::null_mut(),
                        lower_inclusive: false,
                        upper_inclusive: false,
                    };
                    unsafe {
                        duckdb_vx_expr_get_bound_between(self.as_ptr(), &raw mut out);
                    }

                    ExpressionClass::BoundBetween(BoundBetween {
                        input: unsafe { Expression::borrow(out.input) },
                        lower: unsafe { Expression::borrow(out.lower) },
                        upper: unsafe { Expression::borrow(out.upper) },
                        lower_inclusive: out.lower_inclusive,
                        upper_inclusive: out.upper_inclusive,
                    })
                }
                DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_OPERATOR => {
                    let mut out = duckdb_vx_expr_bound_operator {
                        children: ptr::null_mut(),
                        children_count: 0,
                        type_: DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                    };
                    unsafe { duckdb_vx_expr_get_bound_operator(self.as_ptr(), &raw mut out) };

                    let children =
                        unsafe { std::slice::from_raw_parts(out.children, out.children_count) };

                    ExpressionClass::BoundOperator(BoundOperator {
                        children,
                        op: out.type_,
                    })
                }
                DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_FUNCTION => {
                    let mut out = duckdb_vx_expr_bound_function {
                        children: ptr::null_mut(),
                        children_count: 0,
                        scalar_function: ptr::null_mut(),
                        bind_info: ptr::null_mut(),
                    };
                    unsafe { duckdb_vx_expr_get_bound_function(self.as_ptr(), &raw mut out) };

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
    BoundColumnRef(BoundColumnRef),
    BoundConstant(BoundConstant),
    BoundComparison(BoundComparison),
    BoundConjunction(BoundConjunction<'a>),
    BoundBetween(BoundBetween),
    BoundOperator(BoundOperator<'a>),
    BoundFunction(BoundFunction<'a>),
}

pub struct BoundColumnRef {
    pub name: duckdb::string::String,
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

// ==============================================
// Logical Plan Expression Support
// ==============================================

/// Represents the type of an expression in DuckDB's logical plan (simplified enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LogicalExpressionType {
    BoundColumnRef = DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_BOUND_COLUMN_REF,
    BoundFunction = DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_BOUND_FUNCTION,
    Constant = DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_CONSTANT,
    Unknown = DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_EXPRESSION_UNKNOWN,
}

/// Column binding information for logical plans
#[derive(Debug, Clone, Copy)]
pub struct ColumnBinding {
    pub table_index: u64,
    pub column_index: u64,
}

impl From<ColumnBinding> for duckdb_vx_column_binding {
    fn from(binding: ColumnBinding) -> Self {
        duckdb_vx_column_binding {
            table_index: binding.table_index,
            column_index: binding.column_index,
        }
    }
}

impl From<duckdb_vx_column_binding> for ColumnBinding {
    fn from(binding: duckdb_vx_column_binding) -> Self {
        ColumnBinding {
            table_index: binding.table_index,
            column_index: binding.column_index,
        }
    }
}

// Add logical plan methods to the unified Expression struct
impl Expression {
    /// Get the logical plan type of this expression (simplified enum)
    pub fn logical_expression_type(&self) -> LogicalExpressionType {
        let expr_type = unsafe { duckdb_vx_get_expression_type(self.as_ptr()) };
        match expr_type {
            DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_BOUND_COLUMN_REF => LogicalExpressionType::BoundColumnRef,
            DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_BOUND_FUNCTION => LogicalExpressionType::BoundFunction,
            DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_CONSTANT => LogicalExpressionType::Constant,
            _ => LogicalExpressionType::Unknown,
        }
    }

    /// Get string representation using legacy function name
    pub fn to_string_legacy(&self) -> VortexResult<String> {
        unsafe {
            let str_ptr = duckdb_vx_expression_to_string(self.as_ptr());
            if str_ptr.is_null() {
                vortex_bail!("Failed to convert expression to string");
            }
            let result = CStr::from_ptr(str_ptr).to_string_lossy().into_owned();
            duckdb_vx_free_string(str_ptr);
            Ok(result)
        }
    }

    /// Get function name if this is a function expression  
    pub fn function_name(&self) -> VortexResult<Option<String>> {
        unsafe {
            let name_ptr = duckdb_vx_get_function_name_from_expr(self.as_ptr());
            if name_ptr.is_null() {
                Ok(None)
            } else {
                let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
                duckdb_vx_free_string(name_ptr);
                Ok(Some(name))
            }
        }
    }

    /// Get function argument count if this is a function expression
    pub fn function_arg_count(&self) -> usize {
        unsafe { duckdb_vx_get_function_arg_count(self.as_ptr()) as usize }
    }

    /// Get function argument by index if this is a function expression
    pub fn get_function_arg(&self, index: usize) -> Option<Expression> {
        unsafe {
            let arg_ptr = duckdb_vx_get_function_arg(self.as_ptr(), index as u64);
            if arg_ptr.is_null() {
                None
            } else {
                Some(Expression::borrow(arg_ptr))
            }
        }
    }

    /// Get column alias if this is a column reference
    pub fn column_alias(&self) -> VortexResult<Option<String>> {
        unsafe {
            let alias_ptr = duckdb_vx_get_column_alias(self.as_ptr());
            if alias_ptr.is_null() {
                Ok(None)
            } else {
                let alias = CStr::from_ptr(alias_ptr).to_string_lossy().into_owned();
                duckdb_vx_free_string(alias_ptr);
                Ok(Some(alias))
            }
        }
    }

    /// Get column binding if this is a column reference
    pub fn column_binding(&self) -> ColumnBinding {
        let binding = unsafe { duckdb_vx_get_column_binding(self.as_ptr()) };
        binding.into()
    }

    /// Update column binding if this is a column reference
    pub fn update_column_binding(&self, binding: ColumnBinding) {
        unsafe {
            duckdb_vx_update_column_binding(self.as_ptr(), binding.into());
        }
    }

    /// Create a new column reference expression
    pub fn create_column_ref(name: &str, binding: ColumnBinding, depth: u64) -> VortexResult<Self> {
        let c_name = CString::new(name).map_err(|e| {
            vortex_err!("Invalid column name: {}", e)
        })?;
        unsafe {
            let expr_ptr = duckdb_vx_create_column_ref(c_name.as_ptr(), binding.into(), depth);
            if expr_ptr.is_null() {
                vortex_bail!("Failed to create column reference expression")
            } else {
                Ok(Self::own(expr_ptr))
            }
        }
    }
}
