use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ExprRef, Identifier, Scope, ScopeDType, StatsPrunable, VortexExpr};

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Eq, Hash)]
/// Let expressions are of the form `let var = bind in expr`,
/// see `Scope`.
pub struct Let {
    var: Identifier,
    bind: ExprRef,
    expr: ExprRef,
}

impl Let {
    pub fn new_expr(var: Identifier, bind: ExprRef, expr: ExprRef) -> ExprRef {
        Arc::new(Self { var, bind, expr })
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;

    use crate::let_::Let;
    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id};

    pub(crate) struct LetSerde;

    impl Id for LetSerde {
        fn id(&self) -> &'static str {
            "let"
        }
    }

    impl ExprDeserialize for LetSerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::Let(op) = kind else {
                vortex_bail!("wrong kind {:?}, wanted let", kind)
            };

            Ok(Let::new_expr(
                op.var.clone().parse()?,
                children[0].clone(),
                children[1].clone(),
            ))
        }
    }

    impl ExprSerializable for Let {
        fn id(&self) -> &'static str {
            LetSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::Identity(kind::Identity {}))
        }
    }
}

impl Display for Let {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "let {} = {} in {}", self.var, self.bind, self.expr)
    }
}

impl StatsPrunable for Let {}

impl VortexExpr for Let {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, scope: &Scope) -> VortexResult<ArrayRef> {
        let v = self.bind.unchecked_evaluate(scope)?;
        let ctx_p = scope.copy_with_array(self.var.clone(), v);
        self.expr.unchecked_evaluate(&ctx_p)
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.bind, &self.expr]
    }

    fn replacing_children(self: Arc<Self>, mut children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 2);
        let expr = children.remove(1);
        let bind = children.remove(0);
        Let::new_expr(self.var.clone(), bind, expr)
    }

    fn return_dtype(&self, scope: &ScopeDType) -> VortexResult<DType> {
        let v = self.bind.return_dtype(scope)?;
        let ctx_p = scope.copy_with_dtype(self.var.clone(), v);
        self.expr.return_dtype(&ctx_p)
    }
}

impl PartialEq for Let {
    fn eq(&self, other: &Let) -> bool {
        self.var == other.var && self.bind.eq(&other.bind) && self.expr.eq(&other.expr)
    }
}

pub fn let_(ident: Identifier, bind: ExprRef, expr: ExprRef) -> ExprRef {
    Let::new_expr(ident, bind, expr)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::{Scope, eq, get_item_scope, let_, var};

    #[test]
    fn test_two_vars() {
        let a1 = PrimitiveArray::new(buffer![5, 4, 3, 2, 1, 0], Validity::AllValid).to_array();
        let a2 = PrimitiveArray::from_iter(1..=6).to_array();

        let struct_arr = StructArray::from_fields(&[("a1", a1), ("a2", a2)])
            .unwrap()
            .to_array();

        let expr = let_(
            "x".parse().unwrap(),
            get_item_scope("a1"),
            let_(
                "y".parse().unwrap(),
                get_item_scope("a2"),
                eq(var("x"), var("y")),
            ),
        );
        let res = expr.evaluate(&Scope::new(struct_arr)).unwrap();
        let res = res.to_bool().unwrap().boolean_buffer().iter().collect_vec();

        assert_eq!(res, vec![false, false, true, false, false, false])
    }
}
