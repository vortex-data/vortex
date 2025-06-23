use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::compute::{Operator as ArrayOperator, add, and_kleene, compare, or_kleene};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{AnalysisExpr, ExprRef, Operator, Scope, ScopeDType, StatsCatalog, VortexExpr};

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

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind::Kind;

    use crate::{BinaryExpr, ExprDeserialize, ExprRef, ExprSerializable, Id};

    pub(crate) struct BinarySerde;

    impl Id for BinarySerde {
        fn id(&self) -> &'static str {
            "binary"
        }
    }

    impl ExprDeserialize for BinarySerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::BinaryOp(op) = kind else {
                vortex_bail!("wrong kind {:?}, binary", kind)
            };

            Ok(BinaryExpr::new_expr(
                children[0].clone(),
                (*op).try_into()?,
                children[1].clone(),
            ))
        }
    }

    impl ExprSerializable for BinaryExpr {
        fn id(&self) -> &'static str {
            BinarySerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::BinaryOp(self.operator.into()))
        }
    }
}

impl AnalysisExpr for BinaryExpr {
    fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        match self.operator {
            Operator::Eq => {
                let min_lhs = self.lhs.min(catalog);
                let max_lhs = self.lhs.max(catalog);

                let min_rhs = self.rhs.min(catalog);
                let max_rhs = self.rhs.max(catalog);

                let left = min_lhs.zip(max_rhs).map(|(a, b)| gt(a, b));
                let right = min_rhs.zip(max_lhs).map(|(a, b)| gt(a, b));
                left.into_iter().chain(right).reduce(or)
            }
            Operator::NotEq => {
                let min_lhs = self.lhs.min(catalog)?;
                let max_lhs = self.lhs.max(catalog)?;

                let min_rhs = self.rhs.min(catalog)?;
                let max_rhs = self.rhs.max(catalog)?;

                Some(and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs)))
            }
            Operator::Gt => Some(lt_eq(self.lhs.max(catalog)?, self.rhs.min(catalog)?)),
            Operator::Gte => Some(lt(self.lhs.max(catalog)?, self.rhs.min(catalog)?)),
            Operator::Lt => Some(gt_eq(self.lhs.min(catalog)?, self.rhs.max(catalog)?)),
            Operator::Lte => Some(gt(self.lhs.min(catalog)?, self.rhs.max(catalog)?)),
            Operator::And => self
                .lhs
                .stat_falsification(catalog)
                .into_iter()
                .chain(self.rhs.stat_falsification(catalog))
                .reduce(or),
            Operator::Or => Some(and(
                self.lhs.stat_falsification(catalog)?,
                self.rhs.stat_falsification(catalog)?,
            )),
            Operator::CheckedAdd => None,
        }
    }
}

impl VortexExpr for BinaryExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, scope: &Scope) -> VortexResult<ArrayRef> {
        let lhs = self.lhs.unchecked_evaluate(scope)?;
        let rhs = self.rhs.unchecked_evaluate(scope)?;

        match self.operator {
            Operator::Eq => compare(&lhs, &rhs, ArrayOperator::Eq),
            Operator::NotEq => compare(&lhs, &rhs, ArrayOperator::NotEq),
            Operator::Lt => compare(&lhs, &rhs, ArrayOperator::Lt),
            Operator::Lte => compare(&lhs, &rhs, ArrayOperator::Lte),
            Operator::Gt => compare(&lhs, &rhs, ArrayOperator::Gt),
            Operator::Gte => compare(&lhs, &rhs, ArrayOperator::Gte),
            Operator::And => and_kleene(&lhs, &rhs),
            Operator::Or => or_kleene(&lhs, &rhs),
            Operator::CheckedAdd => add(&lhs, &rhs),
        }
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.lhs, &self.rhs]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 2);
        BinaryExpr::new_expr(children[0].clone(), self.operator, children[1].clone())
    }

    fn return_dtype(&self, ctx: &ScopeDType) -> VortexResult<DType> {
        let lhs = self.lhs.return_dtype(ctx)?;
        let rhs = self.rhs.return_dtype(ctx)?;

        if self.operator == Operator::CheckedAdd {
            if lhs.is_primitive() && lhs.eq_ignore_nullability(&rhs) {
                return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
            }
            vortex_bail!("incompatible types for checked add: {} {}", lhs, rhs);
        }

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
/// use vortex_array::{Array, IntoArray, ToCanonical};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{eq, root, lit, Scope};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = eq(root(), lit(3)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().unwrap().boolean_buffer(),
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
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{root, lit, not_eq, Scope};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = not_eq(root(), lit(3)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().unwrap().boolean_buffer(),
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
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{gt_eq, root, lit, Scope};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = gt_eq(root(), lit(3)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().unwrap().boolean_buffer(),
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
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{gt, root, lit, Scope};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = gt(root(), lit(2)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().unwrap().boolean_buffer(),
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
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{root, lit, lt_eq, Scope};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = lt_eq(root(), lit(2)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().unwrap().boolean_buffer(),
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
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_expr::{root, lit, lt, Scope};
///
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = lt(root(), lit(3)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().unwrap().boolean_buffer(),
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
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_expr::{root, lit, or, Scope};
///
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = or(root(), lit(false)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).boolean_buffer(),
/// );
/// ```
pub fn or(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::Or, rhs)
}

/// Collects a list of `or`ed values into a single vortex, expr
/// [x, y, z] => x or (y or z)
pub fn or_collect<I>(iter: I) -> Option<ExprRef>
where
    I: IntoIterator<Item = ExprRef>,
    I::IntoIter: DoubleEndedIterator<Item = ExprRef>,
{
    let mut iter = iter.into_iter();
    let first = iter.next_back()?;
    Some(iter.rfold(first, |acc, elem| or(elem, acc)))
}

/// Create a new `BinaryExpr` using the `And` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::arrays::BoolArray;
/// use vortex_array::{IntoArray, ToCanonical};
/// use vortex_expr::{and, root, lit, Scope};
///
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = and(root(), lit(true)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).boolean_buffer(),
/// );
/// ```
pub fn and(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::And, rhs)
}

/// Collects a list of `and`ed values into a single vortex, expr
/// [x, y, z] => x and (y and z)
pub fn and_collect<I>(iter: I) -> Option<ExprRef>
where
    I: IntoIterator<Item = ExprRef>,
    I::IntoIter: DoubleEndedIterator<Item = ExprRef>,
{
    let mut iter = iter.into_iter();
    let first = iter.next_back()?;
    Some(iter.rfold(first, |acc, elem| and(elem, acc)))
}

/// Collects a list of `and`ed values into a single vortex, expr
/// [x, y, z] => x and (y and z)
pub fn and_collect_right<I>(iter: I) -> Option<ExprRef>
where
    I: IntoIterator<Item = ExprRef>,
{
    let iter = iter.into_iter();
    iter.reduce(and)
}

/// Create a new `BinaryExpr` using the `CheckedAdd` operator.
///
/// ## Example usage
///
/// ```
/// use vortex_array::IntoArray;
/// use vortex_array::arrow::IntoArrowArray as _;
/// use vortex_buffer::buffer;
/// use vortex_expr::{Scope, checked_add, lit, root};
///
/// let xs = buffer![1, 2, 3].into_array();
/// let result = checked_add(root(), lit(5))
///     .evaluate(&Scope::new(xs.to_array()))
///     .unwrap();
///
/// assert_eq!(
///     &result.into_arrow_preferred().unwrap(),
///     &buffer![6, 7, 8]
///         .into_array()
///         .into_arrow_preferred()
///         .unwrap()
/// );
/// ```
pub fn checked_add(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new_expr(lhs, Operator::CheckedAdd, rhs)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability};

    use crate::{
        ScopeDType, VortexExpr, and, and_collect, and_collect_right, col, eq, gt, gt_eq, lit, lt,
        lt_eq, not_eq, or, test_harness,
    };

    #[test]
    fn and_collect_left_assoc() {
        let values = vec![lit(1), lit(2), lit(3)];
        assert_eq!(
            Some(and(lit(1), and(lit(2), lit(3)))),
            and_collect(values.into_iter())
        );
    }

    #[test]
    fn and_collect_right_assoc() {
        let values = vec![lit(1), lit(2), lit(3)];
        assert_eq!(
            Some(and(and(lit(1), lit(2)), lit(3))),
            and_collect_right(values.into_iter())
        );
    }

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        let bool1: Arc<dyn VortexExpr> = col("bool1");
        let bool2: Arc<dyn VortexExpr> = col("bool2");
        assert_eq!(
            and(bool1.clone(), bool2.clone())
                .return_dtype(&ScopeDType::new(dtype.clone()))
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
        assert_eq!(
            or(bool1.clone(), bool2.clone())
                .return_dtype(&ScopeDType::new(dtype.clone()))
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");

        assert_eq!(
            eq(col1.clone(), col2.clone())
                .return_dtype(&ScopeDType::new(dtype.clone()))
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            not_eq(col1.clone(), col2.clone())
                .return_dtype(&ScopeDType::new(dtype.clone()))
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            gt(col1.clone(), col2.clone())
                .return_dtype(&ScopeDType::new(dtype.clone()))
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            gt_eq(col1.clone(), col2.clone())
                .return_dtype(&ScopeDType::new(dtype.clone()))
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            lt(col1.clone(), col2.clone())
                .return_dtype(&ScopeDType::new(dtype.clone()))
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            lt_eq(col1.clone(), col2.clone())
                .return_dtype(&ScopeDType::new(dtype.clone()))
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );

        assert_eq!(
            or(
                lt(col1.clone(), col2.clone()),
                not_eq(col1.clone(), col2.clone())
            )
            .return_dtype(&ScopeDType::new(dtype))
            .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }
}
