// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_array::arrays::{BoolArray, ConstantArray};
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, StatsCatalog, VTable, eq, lit,
    vtable,
};

vtable!(IsNull);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Debug, Hash, Eq)]
pub struct IsNullExpr {
    child: ExprRef,
}

impl PartialEq for IsNullExpr {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child)
    }
}

pub struct IsNullExprEncoding;

impl VTable for IsNullVTable {
    type Expr = IsNullExpr;
    type Encoding = IsNullExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("is_null")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(IsNullExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.child]
    }

    fn with_children(_expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(IsNullExpr::new(children[0].clone()))
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if children.len() != 1 {
            vortex_bail!("IsNull expects exactly one child, got {}", children.len());
        }
        Ok(IsNullExpr::new(children[0].clone()))
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let array = expr.child.unchecked_evaluate(scope)?;
        match array.validity_mask() {
            Mask::AllTrue(len) => Ok(ConstantArray::new(false, len).into_array()),
            Mask::AllFalse(len) => Ok(ConstantArray::new(true, len).into_array()),
            Mask::Values(mask) => Ok(BoolArray::from(mask.bit_buffer().not()).into_array()),
        }
    }

    fn return_dtype(_expr: &Self::Expr, _scope: &DType) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }
}

impl IsNullExpr {
    pub fn new(child: ExprRef) -> Self {
        Self { child }
    }

    pub fn new_expr(child: ExprRef) -> ExprRef {
        Self::new(child).into_expr()
    }
}

impl DisplayAs for IsNullExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "is_null({})", self.child)
            }
            DisplayFormat::Tree => {
                write!(f, "IsNull")
            }
        }
    }
}

impl AnalysisExpr for IsNullExpr {
    fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        let field_path = self.child.field_path()?;
        let null_count_expr = catalog.stats_ref(&field_path, Stat::NullCount)?;
        Some(eq(null_count_expr, lit(0u64)))
    }
}

/// Creates an expression that checks for null values.
///
/// Returns a boolean array indicating which positions contain null values.
///
/// ```rust
/// # use vortex_expr::{is_null, root};
/// let expr = is_null(root());
/// ```
pub fn is_null(child: ExprRef) -> ExprRef {
    IsNullExpr::new(child).into_expr()
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::stats::Stat;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Field, FieldPath, FieldPathSet, Nullability};
    use vortex_scalar::Scalar;
    use vortex_utils::aliases::hash_map::HashMap;

    use crate::is_null::is_null;
    use crate::pruning::checked_pruning_expr;
    use crate::{HashSet, Scope, col, eq, get_item, lit, root, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            is_null(root()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = is_null(root());
        let _ = expr.with_children(vec![root()]);
    }

    #[test]
    fn evaluate_mask() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array();
        let expected = [false, true, false, true, false];

        let result = is_null(root())
            .evaluate(&Scope::new(test_array.clone()))
            .unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result.scalar_at(i),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_false() {
        let test_array = buffer![1, 2, 3, 4, 5].into_array();

        let result = is_null(root())
            .evaluate(&Scope::new(test_array.clone()))
            .unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(
            result.as_constant().unwrap(),
            Scalar::bool(false, Nullability::NonNullable)
        );
    }

    #[test]
    fn evaluate_all_true() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![None::<i32>, None, None, None, None])
                .into_array();

        let result = is_null(root())
            .evaluate(&Scope::new(test_array.clone()))
            .unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(
            result.as_constant().unwrap(),
            Scalar::bool(true, Nullability::NonNullable)
        );
    }

    #[test]
    fn evaluate_struct() {
        let test_array = StructArray::from_fields(&[(
            "a",
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array(),
        )])
        .unwrap()
        .into_array();
        let expected = [false, true, false, true, false];

        let result = is_null(get_item("a", root()))
            .evaluate(&Scope::new(test_array.clone()))
            .unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result.scalar_at(i),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn test_display() {
        let expr = is_null(get_item("name", root()));
        assert_eq!(expr.to_string(), "is_null($.name)");

        let expr2 = is_null(root());
        assert_eq!(expr2.to_string(), "is_null($)");
    }

    #[test]
    fn test_is_null_falsification() {
        let expr = is_null(col("a"));

        let (pruning_expr, st) = checked_pruning_expr(
            &expr,
            &FieldPathSet::from_iter([FieldPath::from_iter([
                Field::Name("a".into()),
                Field::Name("null_count".into()),
            ])]),
        )
        .unwrap();

        assert_eq!(&pruning_expr, &eq(col("a_null_count"), lit(0u64)));
        assert_eq!(
            st.map(),
            &HashMap::from_iter([(FieldPath::from_name("a"), HashSet::from([Stat::NullCount]))])
        );
    }
}
