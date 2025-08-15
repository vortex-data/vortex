// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::ops::Not;

use vortex_array::arrays::{BoolArray, ConstantArray};
use vortex_array::{Array, ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable};

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
        match array.validity_mask()? {
            Mask::AllTrue(len) => Ok(ConstantArray::new(false, len).into_array()),
            Mask::AllFalse(len) => Ok(ConstantArray::new(true, len).into_array()),
            Mask::Values(mask) => Ok(BoolArray::from(mask.boolean_buffer().not()).into_array()),
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

impl Display for IsNullExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "is_null({})", self.child)
    }
}

impl AnalysisExpr for IsNullExpr {}

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
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::is_null::is_null;
    use crate::{Scope, get_item, root, test_harness};

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
                result.scalar_at(i).unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_false() {
        let test_array = PrimitiveArray::from_iter(vec![1, 2, 3, 4, 5]).into_array();

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
                result.scalar_at(i).unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }
}
