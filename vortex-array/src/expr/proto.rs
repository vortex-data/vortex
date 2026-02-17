// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::session::ExprSessionExt;

pub trait ExprSerializeProtoExt {
    /// Serialize the expression to its protobuf representation.
    fn serialize_proto(&self) -> VortexResult<pb::Expr>;
}

impl ExprSerializeProtoExt for Expression {
    fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        let children = self
            .children()
            .iter()
            .map(|child| child.serialize_proto())
            .try_collect()?;

        let metadata = self.options().serialize()?.ok_or_else(|| {
            vortex_err!("Expression '{}' is not serializable: {}", self.id(), self)
        })?;

        Ok(pb::Expr {
            id: self.id().to_string(),
            children,
            metadata: Some(metadata),
        })
    }
}

impl Expression {
    pub fn from_proto(expr: &pb::Expr, session: &VortexSession) -> VortexResult<Expression> {
        let expr_id = ExprId::new_arc(Arc::from(expr.id.to_string()));
        let vtable = session
            .expressions()
            .registry()
            .find(&expr_id)
            .ok_or_else(|| vortex_err!("unknown expression id: {}", expr_id))?;

        let children = expr
            .children
            .iter()
            .map(|e| Expression::from_proto(e, session))
            .collect::<VortexResult<Vec<_>>>()?;

        Expression::try_new(vtable.deserialize(expr.metadata(), session)?, children)
    }
}

/// Deserialize a [`Expression`] from the protobuf representation.
#[deprecated(note = "Use Expression::from_proto instead")]
pub fn deserialize_expr_proto(
    expr: &pb::Expr,
    session: &VortexSession,
) -> VortexResult<Expression> {
    Expression::from_proto(expr, session)
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_proto::expr as pb;

    use super::ExprSerializeProtoExt;
    use crate::LEGACY_SESSION;
    use crate::expr::BetweenOptions;
    use crate::expr::Expression;
    use crate::expr::StrictComparison;
    use crate::expr::exprs::between::between;
    use crate::expr::exprs::binary::and;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::binary::or;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;

    #[test]
    fn expression_serde() {
        let expr: Expression = or(
            and(
                between(
                    lit(1),
                    root(),
                    get_item("a", root()),
                    BetweenOptions {
                        lower_strict: StrictComparison::Strict,
                        upper_strict: StrictComparison::Strict,
                    },
                ),
                lit(1),
            ),
            eq(lit(1), root()),
        );

        let s_expr = expr.serialize_proto().unwrap();
        let buf = s_expr.encode_to_vec();
        let s_expr = pb::Expr::decode(buf.as_slice()).unwrap();
        let deser_expr = Expression::from_proto(&s_expr, &LEGACY_SESSION).unwrap();

        assert_eq!(&deser_expr, &expr);
    }
}
