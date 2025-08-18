// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use vortex_array::compute::list_contains as compute_list_contains;
use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, LiteralVTable, Scope, StatsCatalog,
    VTable, and, gt, lit, lt, or, vtable,
};

vtable!(ListContains);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, Hash, Eq)]
pub struct ListContainsExpr {
    list: ExprRef,
    value: ExprRef,
}

impl PartialEq for ListContainsExpr {
    fn eq(&self, other: &Self) -> bool {
        self.list.eq(&other.list) && self.value.eq(&other.value)
    }
}

pub struct ListContainsExprEncoding;

impl VTable for ListContainsVTable {
    type Expr = ListContainsExpr;
    type Encoding = ListContainsExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("list_contains")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(ListContainsExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.list, &expr.value]
    }

    fn with_children(_expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(ListContainsExpr::new(
            children[0].clone(),
            children[1].clone(),
        ))
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if children.len() != 2 {
            vortex_bail!(
                "ListContains expression must have exactly 2 children, got {}",
                children.len()
            );
        }
        Ok(ListContainsExpr::new(
            children[0].clone(),
            children[1].clone(),
        ))
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        compute_list_contains(
            expr.list.evaluate(scope)?.as_ref(),
            expr.value.evaluate(scope)?.as_ref(),
        )
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        Ok(DType::Bool(
            expr.list.return_dtype(scope)?.nullability()
                | expr.value.return_dtype(scope)?.nullability(),
        ))
    }
}

impl ListContainsExpr {
    pub fn new(list: ExprRef, value: ExprRef) -> Self {
        Self { list, value }
    }

    pub fn new_expr(list: ExprRef, value: ExprRef) -> ExprRef {
        Self::new(list, value).into_expr()
    }

    pub fn value(&self) -> &ExprRef {
        &self.value
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
pub fn list_contains(list: ExprRef, value: ExprRef) -> ExprRef {
    ListContainsExpr::new(list, value).into_expr()
}

impl Display for ListContainsExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "contains({}, {})", &self.list, &self.value)
    }
}

impl AnalysisExpr for ListContainsExpr {
    // falsification(contains([1,2,5], x)) =>
    //   falsification(x != 1) and falsification(x != 2) and falsification(x != 5)

    fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        let min = self.list.min(catalog)?;
        let max = self.list.max(catalog)?;
        // If the list is constant when we can compare each element to the value
        if min == max {
            let list_ = min
                .as_opt::<LiteralVTable>()
                .and_then(|l| l.value().as_list_opt())
                .and_then(|l| l.elements())?;
            if list_.is_empty() {
                // contains([], x) is always false.
                return Some(lit(true));
            }
            let value_max = self.value.max(catalog)?;
            let value_min = self.value.min(catalog)?;

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

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BoolArray, BooleanBuffer, ListArray, PrimitiveArray};
    use vortex_array::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Field, FieldPath, FieldPathSet, Nullability, StructFields};
    use vortex_scalar::Scalar;
    use vortex_utils::aliases::hash_map::HashMap;

    use crate::list_contains::list_contains;
    use crate::pruning::checked_pruning_expr;
    use crate::{Arc, HashSet, Scope, and, col, get_item, gt, lit, lt, or, root};

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
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_all() {
        let arr = test_array();

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_none() {
        let arr = test_array();

        let expr = list_contains(root(), lit(4));
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1).unwrap(),
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
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_nullable() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::Array(BoolArray::from(BooleanBuffer::from(vec![true, false])).into_array()),
        )
        .unwrap()
        .into_array();

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert!(!item.is_valid(1).unwrap());
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
}
