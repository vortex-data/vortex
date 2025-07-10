// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::hash::Hash;

use vortex_array::compute::invert;
use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable};

vtable!(Not);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Debug, Hash)]
pub struct NotExpr {
    child: ExprRef,
}

impl PartialEq for NotExpr {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child)
    }
}

pub struct NotExprEncoding;

impl VTable for NotVTable {
    type Expr = NotExpr;
    type Encoding = NotExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("not")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(NotExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.child]
    }

    fn with_children(_expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(NotExpr::new(children[0].clone()))
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if children.len() != 1 {
            vortex_bail!(
                "Not expression expects exactly one child, got {}",
                children.len()
            );
        }
        Ok(NotExpr::new(children[0].clone()))
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let child_result = expr.child.unchecked_evaluate(scope)?;
        invert(&child_result)
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let child = expr.child.return_dtype(scope)?;
        if !matches!(child, DType::Bool(_)) {
            vortex_bail!("Not expression expects a boolean child, got: {}", child);
        }
        Ok(child)
    }
}

impl NotExpr {
    pub fn new(child: ExprRef) -> Self {
        Self { child }
    }

    pub fn new_expr(child: ExprRef) -> ExprRef {
        Self::new(child).into_expr()
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }
}

impl Display for NotExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!{}", self.child)
    }
}

impl AnalysisExpr for NotExpr {}

pub fn not(operand: ExprRef) -> ExprRef {
    NotExpr::new(operand).into_expr()
}

#[cfg(test)]
mod tests {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_dtype::{DType, Nullability};

    use crate::{Scope, col, not, root, test_harness};

    #[test]
    fn invert_booleans() {
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            not_expr
                .evaluate(&Scope::new(bools.to_array()))
                .unwrap()
                .to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }

    #[test]
    fn dtype() {
        let not_expr = not(root());
        let dtype = DType::Bool(Nullability::NonNullable);
        assert_eq!(
            not_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let dtype = test_harness::struct_dtype();
        assert_eq!(
            not(col("bool1")).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }
}
