// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{CString, c_void};
use std::fmt::{Debug, Display, Formatter};
use std::ptr;

use vortex::error::{VortexResult, vortex_bail, vortex_err};

use crate::cpp::{duckdb_vx_expr_class, *};
use crate::duckdb::string::{VxString, c_string_to_rust_string};
use crate::duckdb::{ScalarFunction, Value, ValueRef};
use crate::{cpp, duckdb, wrapper};

// TODO(joe): replace with lifetime_wrapper!
wrapper!(Expression, duckdb_vx_expr, duckdb_vx_destroy_expr);

impl Expression {
    pub fn as_class_id(&self) -> duckdb_vx_expr_class {
        unsafe { duckdb_vx_expr_get_class(self.as_ptr()) }
    }

    /// Get the expression depth if this is a BoundColumnRef expression.
    /// Returns None for other expression types.
    ///
    /// Expression depth represents how many query levels deep a column reference is.
    /// Depth 0 = current query level, depth 1 = parent query (correlated), etc.
    pub fn get_expression_depth(&self) -> Option<u64> {
        (self.as_class_id() == DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF)
            .then(|| unsafe { duckdb_vx_expr_get_bound_column_ref_depth(self.as_ptr()) })
    }

    /// Match the subclass of the expression.
    pub fn as_class(&self) -> Option<ExpressionClass<'_>> {
        Some(match unsafe { duckdb_vx_expr_get_class(self.as_ptr()) } {
            DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF => {
                let ptr = unsafe { duckdb_vx_expr_get_bound_column_ref_get_name(self.as_ptr()) };
                let bind_ptr = unsafe { duckdb_vx_get_column_binding(self.as_ptr()) };

                ExpressionClass::BoundColumnRef(BoundColumnRef {
                    expr: self,
                    name: duckdb::string::String::from_ptr(ptr),
                    column_binding: bind_ptr.into(),
                })
            }
            DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_CONSTANT => {
                let value = unsafe {
                    ValueRef::borrow(duckdb_vx_expr_bound_constant_get_value(self.as_ptr()))
                };
                ExpressionClass::BoundConstant(BoundConstant { expr: self, value })
            }
            DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_CONJUNCTION => {
                let mut out = duckdb_vx_expr_bound_conjunction {
                    children: ptr::null_mut(),
                    children_count: 0,
                    type_: DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                };
                unsafe { duckdb_vx_expr_get_bound_conjunction(self.as_ptr(), &raw mut out) };

                let children =
                    unsafe { std::slice::from_raw_parts(out.children, out.children_count) };

                ExpressionClass::BoundConjunction(BoundConjunction {
                    expr: self,
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
                unsafe { duckdb_vx_expr_get_bound_comparison(self.as_ptr(), &raw mut out) };

                ExpressionClass::BoundComparison(BoundComparison {
                    expr: self,
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
                    expr: self,
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
                    expr: self,
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
                    expr: self,
                    children,
                    scalar_function: unsafe { ScalarFunction::borrow(out.scalar_function) },
                    bind_info: out.bind_info,
                })
            }
            _ => {
                return None;
            }
        })
    }
}

pub enum ExpressionClass<'a> {
    BoundColumnRef(BoundColumnRef<'a>),
    BoundConstant(BoundConstant<'a>),
    BoundComparison(BoundComparison<'a>),
    BoundConjunction(BoundConjunction<'a>),
    BoundBetween(BoundBetween<'a>),
    BoundOperator(BoundOperator<'a>),
    BoundFunction(BoundFunction<'a>),
}

pub struct BoundColumnRef<'a> {
    expr: &'a Expression,
    pub name: duckdb::string::String,
    pub column_binding: ColumnBinding,
}

impl BoundColumnRef<'_> {
    /// Get the expression depth for this BoundColumnRef.
    ///
    /// Expression depth in DuckDB represents how many query levels deep this column reference is.
    /// A depth of 0 means the column is from the current query level,
    /// depth 1 means it's from a parent query (correlated subquery), etc.
    /// This is important for query optimization and determining if a subquery is correlated.
    pub fn expression_depth(&self) -> u64 {
        unsafe { duckdb_vx_expr_get_bound_column_ref_depth(self.expr.as_ptr()) }
    }
}

pub struct BoundConstant<'a> {
    expr: &'a Expression,
    pub value: ValueRef<'a>,
}

impl BoundConstant<'_> {
    // Specific methods for BoundConstant can be added here
}

pub struct BoundComparison<'a> {
    expr: &'a Expression,
    pub left: Expression,
    pub right: Expression,
    pub op: DUCKDB_VX_EXPR_TYPE,
}

pub struct BoundBetween<'a> {
    expr: &'a Expression,
    pub input: Expression,
    pub lower: Expression,
    pub upper: Expression,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

pub struct BoundConjunction<'a> {
    expr: &'a Expression,
    children: &'a [duckdb_vx_expr],
    pub op: DUCKDB_VX_EXPR_TYPE,
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
    expr: &'a Expression,
    children: &'a [duckdb_vx_expr],
    pub op: DUCKDB_VX_EXPR_TYPE,
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
    expr: &'a Expression,
    children: &'a [duckdb_vx_expr],
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

    pub fn function_name(&self) -> Option<String> {
        unsafe {
            let name_ptr = duckdb_vx_get_function_name_from_expr(self.expr.as_ptr());
            c_string_to_rust_string(name_ptr)
        }
    }

    /// Get function argument count - only works if this is a BoundFunction expression
    pub fn function_arg_count(&self) -> usize {
        unsafe {
            duckdb_vx_get_function_arg_count(self.expr.as_ptr())
                .try_into()
                .unwrap_or(0)
        }
        // } else {
        //     0
        // }
    }

    /// Get function argument by index - only works if this is a BoundFunction expression
    pub fn get_function_arg(&self, index: usize) -> Option<Expression> {
        // Check if this is a BoundFunction using the class ID directly to avoid borrowing issues
        unsafe {
            let arg_ptr = duckdb_vx_get_function_arg(self.expr.as_ptr(), index as u64);
            if arg_ptr.is_null() {
                None
            } else {
                Some(Expression::borrow(arg_ptr))
            }
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
            DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_BOUND_COLUMN_REF => {
                LogicalExpressionType::BoundColumnRef
            }
            DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_BOUND_FUNCTION => {
                LogicalExpressionType::BoundFunction
            }
            DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_CONSTANT => LogicalExpressionType::Constant,
            _ => LogicalExpressionType::Unknown,
        }
    }

    /// Get string representation using legacy function name
    pub fn to_string_legacy(&self) -> VortexResult<String> {
        unsafe {
            let vx_string_ptr = duckdb_vx_expression_to_string(self.as_ptr());
            match VxString::from_raw(vx_string_ptr) {
                Some(vx_string) => Ok(vx_string.to_string()),
                None => vortex_bail!("Failed to convert expression to string"),
            }
        }
    }

    /// Get column alias - only works if this is a BoundColumnRef expression
    pub fn column_alias(&self) -> VortexResult<Option<String>> {
        // Check if this is a BoundColumnRef using the class ID directly to avoid borrowing issues
        // if unsafe { duckdb_vx_expr_get_class(self.as_ptr()) }
        //     == DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF
        // {
        unsafe {
            let alias_ptr = duckdb_vx_get_column_alias(self.as_ptr());
            Ok(c_string_to_rust_string(alias_ptr))
        }
        // } else {
        //     Ok(None)
        // }
    }

    /// Get column binding - only works if this is a BoundColumnRef expression
    pub fn column_binding(&self) -> Option<ColumnBinding> {
        // Check if this is a BoundColumnRef using the class ID directly to avoid borrowing issues
        // if unsafe { duckdb_vx_expr_get_class(self.as_ptr()) }
        //     == DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF
        // {
        let binding = unsafe { duckdb_vx_get_column_binding(self.as_ptr()) };
        Some(binding.into())
        // } else {
        //     None
        // }
    }

    /// Update column binding - only works if this is a BoundColumnRef expression
    pub fn update_column_binding(&self, binding: ColumnBinding) {
        // Check if this is a BoundColumnRef using the class ID directly to avoid borrowing issues
        // if unsafe { duckdb_vx_expr_get_class(self.as_ptr()) }
        //     == DUCKDB_VX_EXPR_CLASS::DUCKDB_VX_EXPR_CLASS_BOUND_COLUMN_REF
        // {
        unsafe {
            duckdb_vx_update_column_binding(self.as_ptr(), binding.into());
        }
        // } else {
        //     false
        // }
    }

    /// Create a new column reference expression
    pub fn create_column_ref(name: &str, binding: ColumnBinding, depth: u64) -> VortexResult<Self> {
        let c_name = CString::new(name).map_err(|e| vortex_err!("Invalid column name: {}", e))?;
        unsafe {
            let expr_ptr = duckdb_vx_create_column_ref(c_name.as_ptr(), binding.into(), depth);
            if expr_ptr.is_null() {
                vortex_bail!("Failed to create column reference expression")
            } else {
                Ok(Self::own(expr_ptr))
            }
        }
    }

    /// Get detailed debug string representation of this expression
    pub fn to_debug_string(&self) -> VortexResult<String> {
        unsafe {
            let vx_string_ptr = duckdb_vx_expr_to_debug_string(self.as_ptr());
            match VxString::from_raw(vx_string_ptr) {
                Some(vx_string) => Ok(vx_string.to_string()),
                None => vortex_bail!("Failed to convert expression to debug string"),
            }
        }
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.to_string_legacy() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<Expression>"),
        }
    }
}

impl Debug for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.to_debug_string() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<Expression>"),
        }
    }
}
