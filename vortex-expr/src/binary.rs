use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::compute::{and_kleene, compare, or_kleene, Operator as ArrayOperator};
use vortex_array::ArrayData;
use vortex_dtype::field::Field;
use vortex_error::VortexResult;

use crate::{unbox_any, ExprRef, Operator, VortexExpr};

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    lhs: ExprRef,
    operator: Operator,
    rhs: ExprRef,
}

impl BinaryExpr {
    pub fn new_expr(lhs: ExprRef, operator: Operator, rhs: ExprRef) -> ExprRef {
        Arc::new(Self { lhs, operator, rhs })
    }

    pub fn lhs(&self) -> &ExprRef {
        &self.lhs
    }

    pub fn rhs(&self) -> &ExprRef {
        &self.rhs
    }

    pub fn op(&self) -> Operator {
        self.operator
    }
}

impl Display for BinaryExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {} {})", self.lhs, self.operator, self.rhs)
    }
}

impl VortexExpr for BinaryExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let lhs = self.lhs.evaluate(batch)?;
        let rhs = self.rhs.evaluate(batch)?;

        match self.operator {
            Operator::Eq => compare(lhs, rhs, ArrayOperator::Eq),
            Operator::NotEq => compare(lhs, rhs, ArrayOperator::NotEq),
            Operator::Lt => compare(lhs, rhs, ArrayOperator::Lt),
            Operator::Lte => compare(lhs, rhs, ArrayOperator::Lte),
            Operator::Gt => compare(lhs, rhs, ArrayOperator::Gt),
            Operator::Gte => compare(lhs, rhs, ArrayOperator::Gte),
            Operator::And => and_kleene(lhs, rhs),
            Operator::Or => or_kleene(lhs, rhs),
        }
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        self.lhs.collect_references(references);
        self.rhs.collect_references(references);
    }
}

impl PartialEq<dyn Any> for BinaryExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x.operator == self.operator && x.lhs.eq(&self.lhs) && x.rhs.eq(&self.rhs))
            .unwrap_or(false)
    }
}

/// Create a new `BinaryExpr` using the `Eq` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArrayData, IntoArrayVariant};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{eq, ident, lit};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable).into_array();
/// let result = eq(ident(), lit(3)).evaluate(&xs).unwrap();
///
/// assert_eq!(
///     result.into_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).boolean_buffer(),
/// );
/// ```
pub fn eq(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::Eq, rhs)
}

/// Create a new `BinaryExpr` using the `NotEq` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArrayData, IntoArrayVariant};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{ident, lit, not_eq};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable).into_array();
/// let result = not_eq(ident(), lit(3)).evaluate(&xs).unwrap();
///
/// assert_eq!(
///     result.into_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).boolean_buffer(),
/// );
/// ```
pub fn not_eq(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::NotEq, rhs)
}

/// Create a new `BinaryExpr` using the `Gte` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArrayData, IntoArrayVariant};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{gt_eq, ident, lit};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable).into_array();
/// let result = gt_eq(ident(), lit(3)).evaluate(&xs).unwrap();
///
/// assert_eq!(
///     result.into_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).boolean_buffer(),
/// );
/// ```
pub fn gt_eq(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::Gte, rhs)
}

/// Create a new `BinaryExpr` using the `Gt` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArrayData, IntoArrayVariant};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{gt, ident, lit};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable).into_array();
/// let result = gt(ident(), lit(2)).evaluate(&xs).unwrap();
///
/// assert_eq!(
///     result.into_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).boolean_buffer(),
/// );
/// ```
pub fn gt(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::Gt, rhs)
}

/// Create a new `BinaryExpr` using the `Lte` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArrayData, IntoArrayVariant};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{ident, lit, lt_eq};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable).into_array();
/// let result = lt_eq(ident(), lit(2)).evaluate(&xs).unwrap();
///
/// assert_eq!(
///     result.into_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).boolean_buffer(),
/// );
/// ```
pub fn lt_eq(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::Lte, rhs)
}

/// Create a new `BinaryExpr` using the `Lt` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArrayData, IntoArrayVariant};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{ident, lit, lt};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable).into_array();
/// let result = lt(ident(), lit(3)).evaluate(&xs).unwrap();
///
/// assert_eq!(
///     result.into_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).boolean_buffer(),
/// );
/// ```
pub fn lt(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::Lt, rhs)
}

/// Create a new `BinaryExpr` using the `Or` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::BoolArray;
/// use vortex_array::{IntoArrayData, IntoArrayVariant};
/// use vortex_expr::{ ident, lit, or};
///
/// let xs = BoolArray::from_iter(vec![true, false, true]).into_array();
/// let result = or(ident(), lit(false)).evaluate(&xs).unwrap();
///
/// assert_eq!(
///     result.into_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).boolean_buffer(),
/// );
/// ```
pub fn or(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::Or, rhs)
}

/// Create a new `BinaryExpr` using the `And` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::BoolArray;
/// use vortex_array::{IntoArrayData, IntoArrayVariant};
/// use vortex_expr::{and, ident, lit};
///
/// let xs = BoolArray::from_iter(vec![true, false, true]).into_array();
/// let result = and(ident(), lit(true)).evaluate(&xs).unwrap();
///
/// assert_eq!(
///     result.into_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).boolean_buffer(),
/// );
/// ```
pub fn and(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::And, rhs)
}
