use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::compute::{and_kleene, compare, or_kleene, Operator as ArrayOperator};
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ExprRef, Operator, VortexExpr};

#[derive(Debug, Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
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

    fn unchecked_evaluate(&self, batch: &Array) -> VortexResult<Array> {
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

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.lhs, &self.rhs]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 2);
        BinaryExpr::new_expr(children[0].clone(), self.operator, children[1].clone())
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        let lhs = self.lhs.return_dtype(scope_dtype)?;
        let rhs = self.rhs.return_dtype(scope_dtype)?;
        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }
}

impl PartialEq for BinaryExpr {
    fn eq(&self, other: &BinaryExpr) -> bool {
        other.operator == self.operator && other.lhs.eq(&self.lhs) && other.rhs.eq(&self.rhs)
    }
}

/// Create a new `BinaryExpr` using the `Eq` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArray, IntoArrayVariant};
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
/// use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArray, IntoArrayVariant};
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
/// use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArray, IntoArrayVariant};
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
/// use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArray, IntoArrayVariant};
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
/// use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArray, IntoArrayVariant};
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
/// use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// use vortex_array::{IntoArray, IntoArrayVariant};
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
/// use vortex_array::arrays::BoolArray;
/// use vortex_array::{IntoArray, IntoArrayVariant};
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
/// use vortex_array::arrays::BoolArray;
/// use vortex_array::{IntoArray, IntoArrayVariant};
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability};

    use crate::{and, col, eq, gt, gt_eq, lt, lt_eq, not_eq, or, test_harness, VortexExpr};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        let bool1: Arc<dyn VortexExpr> = col("bool1");
        let bool2: Arc<dyn VortexExpr> = col("bool2");
        assert_eq!(
            and(bool1.clone(), bool2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
        assert_eq!(
            or(bool1.clone(), bool2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");

        assert_eq!(
            eq(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            not_eq(col1.clone(), col2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            gt(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            gt_eq(col1.clone(), col2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            lt(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            lt_eq(col1.clone(), col2.clone())
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );

        assert_eq!(
            or(
                lt(col1.clone(), col2.clone()),
                not_eq(col1.clone(), col2.clone())
            )
            .return_dtype(&dtype)
            .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }
}
