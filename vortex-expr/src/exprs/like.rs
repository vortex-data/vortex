// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_array::compute::{LikeOptions, like};
use vortex_array::{ArrayRef, DeserializeMetadata, ProstMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_proto::expr as pb;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable};

vtable!(Like);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Debug, Hash, Eq)]
pub struct LikeExpr {
    child: ExprRef,
    pattern: ExprRef,
    negated: bool,
    case_insensitive: bool,
}

impl PartialEq for LikeExpr {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child)
            && self.pattern.eq(&other.pattern)
            && self.negated == other.negated
            && self.case_insensitive == other.case_insensitive
    }
}

pub struct LikeExprEncoding;

impl VTable for LikeVTable {
    type Expr = LikeExpr;
    type Encoding = LikeExprEncoding;
    type Metadata = ProstMetadata<pb::LikeOpts>;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("like")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(LikeExprEncoding.as_ref())
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(ProstMetadata(pb::LikeOpts {
            negated: expr.negated,
            case_insensitive: expr.case_insensitive,
        }))
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.child, &expr.pattern]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(LikeExpr::new(
            children[0].clone(),
            children[1].clone(),
            expr.negated,
            expr.case_insensitive,
        ))
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if children.len() != 2 {
            vortex_bail!(
                "Like expression must have exactly 2 children, got {}",
                children.len()
            );
        }

        Ok(LikeExpr::new(
            children[0].clone(),
            children[1].clone(),
            metadata.negated,
            metadata.case_insensitive,
        ))
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let child = expr.child().unchecked_evaluate(scope)?;
        let pattern = expr.pattern().unchecked_evaluate(scope)?;
        like(
            &child,
            &pattern,
            LikeOptions {
                negated: expr.negated,
                case_insensitive: expr.case_insensitive,
            },
        )
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let input = expr.child().return_dtype(scope)?;
        let pattern = expr.pattern().return_dtype(scope)?;
        Ok(DType::Bool(
            (input.is_nullable() || pattern.is_nullable()).into(),
        ))
    }
}

impl LikeExpr {
    pub fn new(child: ExprRef, pattern: ExprRef, negated: bool, case_insensitive: bool) -> Self {
        Self {
            child,
            pattern,
            negated,
            case_insensitive,
        }
    }

    pub fn new_expr(
        child: ExprRef,
        pattern: ExprRef,
        negated: bool,
        case_insensitive: bool,
    ) -> ExprRef {
        Self::new(child, pattern, negated, case_insensitive).into_expr()
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }

    pub fn pattern(&self) -> &ExprRef {
        &self.pattern
    }

    pub fn negated(&self) -> bool {
        self.negated
    }

    pub fn case_insensitive(&self) -> bool {
        self.case_insensitive
    }
}

impl DisplayAs for LikeExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "{} LIKE {}", self.child(), self.pattern())
            }
            DisplayFormat::Tree => {
                write!(f, "Like")
            }
        }
    }

    fn child_names(&self) -> Option<Vec<String>> {
        Some(vec!["child".to_string(), "pattern".to_string()])
    }
}

impl AnalysisExpr for LikeExpr {}

#[cfg(test)]
mod tests {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_dtype::{DType, Nullability};

    use crate::{LikeExpr, Scope, get_item, lit, not, root};

    #[test]
    fn invert_booleans() {
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            not_expr
                .evaluate(&Scope::new(bools.to_array()))
                .unwrap()
                .to_bool()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }

    #[test]
    fn dtype() {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let like_expr = LikeExpr::new(root(), lit("%test%"), false, false);
        assert_eq!(
            like_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn test_display() {
        let expr = LikeExpr::new(get_item("name", root()), lit("%john%"), false, false);
        assert_eq!(expr.to_string(), "$.name LIKE \"%john%\"");

        let expr2 = LikeExpr::new(root(), lit("test*"), true, true);
        assert_eq!(expr2.to_string(), "$ LIKE \"test*\"");
    }
}
