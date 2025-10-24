// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata, ToCanonical};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, Scope, VTable, vtable};

vtable!(Mask);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Debug, Hash, Eq)]
pub struct MaskExpr {
    /// The target array to mask
    pub target: ExprRef,
    /// An expression that yields a boolean array for the masking operation.
    /// True values will be set to null, false values will be set to non-null
    pub mask: ExprRef,
}

impl PartialEq for MaskExpr {
    fn eq(&self, other: &Self) -> bool {
        self.target.eq(&other.target) && self.mask.eq(&other.mask)
    }
}

impl MaskExpr {
    /// Create a new `MaskExpr` against the provided `target` and `mask`.
    ///
    /// The `target` is an expression that evaluates to any array type.
    ///
    /// `mask` must evaluate to a non-nullable `Bool` array that is the same length as the `target`.
    /// All `true` values will set the result to `null`, and `false` values will preserve the
    /// corresponding value from `target`.
    pub fn new(target: ExprRef, mask: ExprRef) -> Self {
        Self { target, mask }
    }

    pub fn target(&self) -> &ExprRef {
        &self.target
    }

    pub fn mask(&self) -> &ExprRef {
        &self.mask
    }
}

impl DisplayAs for MaskExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "mask({}, {})", self.target, self.mask)
            }
            DisplayFormat::Tree => {
                write!(f, "Mask")
            }
        }
    }

    fn child_names(&self) -> Option<Vec<String>> {
        Some(vec!["target".to_string(), "mask".to_string()])
    }
}

pub struct MaskExprEncoding;

impl VTable for MaskVTable {
    type Expr = MaskExpr;
    type Encoding = MaskExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("vortex.mask")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(MaskExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.target, &expr.target]
    }

    fn with_children(_expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        vortex_ensure!(
            children.len() == 2,
            "cannot build MaskExpr: expected children [target, mask], received {} children",
            children.len()
        );

        let target = children[0].clone();
        let mask = children[1].clone();

        Ok(MaskExpr { target, mask })
    }

    fn build(
        _encoding: &Self::Encoding,
        _: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        vortex_ensure!(
            children.len() == 2,
            "MaskExpr expected children [target, mask], received {} children",
            children.len()
        );

        let target = children[0].clone();
        let mask = children[1].clone();

        Ok(MaskExpr { target, mask })
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let target = expr.target.evaluate(scope)?;
        let mask = expr.mask.evaluate(scope)?;
        vortex_array::compute::mask(
            target.as_ref(),
            &Mask::from_buffer(mask.to_bool().bit_buffer().clone()),
        )
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        // Mask operation always returns a nullable result
        Ok(expr.target.return_dtype(scope)?.as_nullable())
    }
}

impl AnalysisExpr for MaskExpr {}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BoolArray, PrimitiveArray, StructArray};
    use vortex_array::{ArrayEq, IntoArray, Precision};

    use crate::exprs::mask::MaskExpr;
    use crate::{Scope, get_item, is_null, root};

    #[test]
    fn test_mask_primitive() {
        let root_array =
            PrimitiveArray::from_option_iter([Some(1), Some(2), None, Some(4)]).into_array();
        let scope = Scope::new(root_array.clone());

        // Mask(root, IsNull(root)) should match root
        let expr = MaskExpr::new(root(), is_null(root()));
        let result = expr.evaluate(&scope).unwrap();
        assert!(result.array_eq(&root_array, Precision::Value));
    }

    #[test]
    fn test_mask_struct() {
        // Perform a mask operation onto a nested result using the struct array instead.
        let a = PrimitiveArray::from_option_iter([Some(1), Some(2), None, Some(4)]).into_array();
        let b = BoolArray::from_iter([false, true, false, true]).into_array();

        let root_array = StructArray::from_fields(&[("a", a), ("b", b)])
            .unwrap()
            .into_array();

        let scope = Scope::new(root_array.clone());

        // mask a using the b array.
        let expr = MaskExpr::new(get_item("a", root()), get_item("b", root()));

        let result = expr.evaluate(&scope).unwrap();

        assert_eq!(result.scalar_at(0), Some(1i32).into());
        assert_eq!(result.scalar_at(1), Option::<i32>::None.into());
        assert_eq!(result.scalar_at(2), Option::<i32>::None.into());
        assert_eq!(result.scalar_at(3), Option::<i32>::None.into());
    }
}
