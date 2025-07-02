use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;

use vortex_array::compute::invert;
use vortex_array::{ArrayRef, DeserializeMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, ScopeDType, VTable,
    VortexExpr, vtable,
};

vtable!(Not);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NotExpr {
    child: ExprRef,
}

pub struct NotExprEncoding;

impl VTable for NotVTable {
    type Expr = NotExpr;
    type Encoding = NotExprEncoding;
    type Metadata = ();

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("not")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(&NotExprEncoding)
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(())
    }

    fn children(expr: &Self::Expr) -> Vec<ExprRef> {
        vec![expr.child.clone()]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        if children.len() != 1 {
            vortex_bail!(
                "Not expression expects exactly one child, got {}",
                children.len()
            );
        }
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

    fn return_dtype(expr: &Self::Expr, scope: &ScopeDType) -> VortexResult<DType> {
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

    pub fn child(&self) -> &ExprRef {
        &self.child
    }
}

impl Display for NotExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!{}", self.child)
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use expr::kind;
    use vortex_error::VortexResult;
    use vortex_proto::expr;
    use vortex_proto::expr::kind::Kind;

    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id, NotExpr};

    pub struct NotSerde;

    impl Id for NotSerde {
        fn id(&self) -> &'static str {
            "not"
        }
    }

    impl ExprDeserialize for NotSerde {
        fn deserialize(&self, _expr: &Kind, mut children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            Ok(NotExpr::new(children.remove(0)))
        }
    }

    impl ExprSerializable for NotExpr {
        fn id(&self) -> &'static str {
            NotSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::Not(kind::Not {}))
        }
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

    use crate::{Scope, ScopeDType, col, not, root, test_harness};

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
            not_expr.return_dtype(&ScopeDType::new(dtype)).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let dtype = test_harness::struct_dtype();
        assert_eq!(
            not(col("bool1"))
                .return_dtype(&ScopeDType::new(dtype))
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }
}
