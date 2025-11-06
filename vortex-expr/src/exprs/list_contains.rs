// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::compute::list_contains as compute_list_contains;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::exprs::binary::{and, gt, lt, or};
use crate::exprs::literal::{Literal, lit};
use crate::{ChildName, ExprId, Expression, ExpressionView, StatsCatalog, VTable, VTableExt};

pub struct ListContains;

impl VTable for ListContains {
    type Instance = ();

    fn id(&self) -> ExprId {
        ExprId::from("vortex.list.contains")
    }

    fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        Ok(Some(()))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 2 {
            vortex_bail!(
                "ListContains expression requires exactly 2 children, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            1 => ChildName::from("needle"),
            _ => unreachable!(
                "Invalid child index {} for ListContains expression",
                child_idx
            ),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "contains(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let list_dtype = expr.child(0).return_dtype(scope)?;
        let value_dtype = expr.child(1).return_dtype(scope)?;

        let nullability = match list_dtype {
            DType::List(_, list_nullability) => list_nullability,
            _ => {
                vortex_bail!(
                    "First argument to ListContains must be a List, got {:?}",
                    list_dtype
                );
            }
        } | value_dtype.nullability();

        Ok(DType::Bool(nullability))
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let list_array = expr.child(0).evaluate(scope)?;
        let value_array = expr.child(1).evaluate(scope)?;
        compute_list_contains(list_array.as_ref(), value_array.as_ref())
    }

    fn stat_falsification(
        &self,
        expr: &ExpressionView<Self>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        // falsification(contains([1,2,5], x)) =>
        //   falsification(x != 1) and falsification(x != 2) and falsification(x != 5)
        let min = expr.list().stat_min(catalog)?;
        let max = expr.list().stat_max(catalog)?;
        // If the list is constant when we can compare each element to the value
        if min == max {
            let list_ = min
                .as_opt::<Literal>()
                .and_then(|l| l.data().as_list_opt())
                .and_then(|l| l.elements())?;
            if list_.is_empty() {
                // contains([], x) is always false.
                return Some(lit(true));
            }
            let value_max = expr.needle().stat_max(catalog)?;
            let value_min = expr.needle().stat_min(catalog)?;

            return list_
                .iter()
                .map(move |v| {
                    or(
                        lt(value_max.clone(), lit(v.clone())),
                        gt(value_min.clone(), lit(v.clone())),
                    )
                })
                .reduce(and);
        }

        None
    }
}

/// Creates an expression that checks if a value is contained in a list.
///
/// Returns a boolean array indicating whether the value appears in each list.
///
/// ```rust
/// # use vortex_expr::{list_contains, lit, root};
/// let expr = list_contains(root(), lit(42));
/// ```
pub fn list_contains(list: Expression, value: Expression) -> Expression {
    ListContains.new_expr((), [list, value])
}

impl ExpressionView<'_, ListContains> {
    pub fn list(&self) -> &Expression {
        &self.children()[0]
    }

    pub fn needle(&self) -> &Expression {
        &self.children()[1]
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BoolArray, ListArray, PrimitiveArray};
    use vortex_array::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_buffer::BitBuffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Field, FieldPath, FieldPathSet, Nullability, StructFields};
    use vortex_scalar::Scalar;
    use vortex_utils::aliases::hash_map::HashMap;

    use super::list_contains;
    use crate::exprs::binary::{and, gt, lt, or};
    use crate::exprs::get_item::{col, get_item};
    use crate::exprs::literal::lit;
    use crate::exprs::root::root;
    use crate::pruning::checked_pruning_expr;
    use crate::{Arc, HashSet};

    fn test_array() -> ArrayRef {
        ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2, 2, 2, 3, 3, 3]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 10]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array()
    }

    #[test]
    pub fn test_one() {
        let arr = test_array();

        let expr = list_contains(root(), lit(1));
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(item.scalar_at(0), Scalar::bool(true, Nullability::Nullable));
        assert_eq!(
            item.scalar_at(1),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_all() {
        let arr = test_array();

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(item.scalar_at(0), Scalar::bool(true, Nullability::Nullable));
        assert_eq!(item.scalar_at(1), Scalar::bool(true, Nullability::Nullable));
    }

    #[test]
    pub fn test_none() {
        let arr = test_array();

        let expr = list_contains(root(), lit(4));
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(
            item.scalar_at(0),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_empty() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(item.scalar_at(0), Scalar::bool(true, Nullability::Nullable));
        assert_eq!(
            item.scalar_at(1),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_nullable() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::Array(BoolArray::from(BitBuffer::from(vec![true, false])).into_array()),
        )
        .unwrap()
        .into_array();

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(item.scalar_at(0), Scalar::bool(true, Nullability::Nullable));
        assert!(!item.is_valid(1));
    }

    #[test]
    pub fn test_return_type() {
        let scope = DType::Struct(
            StructFields::new(
                ["array"].into(),
                vec![DType::List(
                    Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                    Nullability::Nullable,
                )],
            ),
            Nullability::NonNullable,
        );

        let expr = list_contains(get_item("array", root()), lit(2));

        // Expect nullable, although scope is non-nullable
        assert_eq!(
            expr.return_dtype(&scope).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }

    #[test]
    pub fn list_falsification() {
        let expr = list_contains(
            lit(Scalar::list(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                vec![1.into(), 2.into(), 3.into()],
                Nullability::NonNullable,
            )),
            col("a"),
        );

        let (expr, st) = checked_pruning_expr(
            &expr,
            &FieldPathSet::from_iter([
                FieldPath::from_iter([Field::Name("a".into()), Field::Name("max".into())]),
                FieldPath::from_iter([Field::Name("a".into()), Field::Name("min".into())]),
            ]),
        )
        .unwrap();

        assert_eq!(
            &expr,
            &and(
                and(
                    or(lt(col("a_max"), lit(1i32)), gt(col("a_min"), lit(1i32)),),
                    or(lt(col("a_max"), lit(2i32)), gt(col("a_min"), lit(2i32)),)
                ),
                or(lt(col("a_max"), lit(3i32)), gt(col("a_min"), lit(3i32)),)
            )
        );

        assert_eq!(
            st.map(),
            &HashMap::from_iter([(
                FieldPath::from_name("a"),
                HashSet::from([Stat::Min, Stat::Max])
            )])
        );
    }

    #[test]
    pub fn test_display() {
        let expr = list_contains(get_item("tags", root()), lit("urgent"));
        assert_eq!(expr.to_string(), "contains($.tags, \"urgent\")");

        let expr2 = list_contains(root(), lit(42));
        assert_eq!(expr2.to_string(), "contains($, 42i32)");
    }
}
