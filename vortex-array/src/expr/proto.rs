// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::expr::Expression;
use crate::scalar_fn::ForeignScalarFnVTable;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::session::ScalarFnSessionExt;

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
        let expr_id = ScalarFnId::new(expr.id.as_str());
        let children = expr
            .children
            .iter()
            .map(|e| Expression::from_proto(e, session))
            .collect::<VortexResult<Vec<_>>>()?;

        let scalar_fn = if let Some(vtable) = session.scalar_fns().registry().find(&expr_id) {
            vtable.deserialize(expr.metadata(), session)?
        } else if session.allows_unknown() {
            ForeignScalarFnVTable::make_scalar_fn(expr_id, expr.metadata().to_vec(), children.len())
        } else {
            return Err(vortex_err!("unknown expression id: {}", expr_id));
        };

        Expression::try_new(scalar_fn, children)
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
    use vortex_session::VortexSession;

    use super::ExprSerializeProtoExt;
    use crate::array_session;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::or;
    use crate::expr::root;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;
    use crate::scalar_fn::session::ScalarFnSession;

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
        let deser_expr = Expression::from_proto(&s_expr, &array_session()).unwrap();

        assert_eq!(&deser_expr, &expr);
    }

    #[test]
    fn unknown_expression_id_allow_unknown() {
        let session = VortexSession::empty()
            .with::<ScalarFnSession>()
            .allow_unknown();

        let expr_proto = pb::Expr {
            id: "vortex.test.foreign_scalar_fn".to_string(),
            metadata: Some(vec![1, 2, 3, 4]),
            children: vec![root().serialize_proto().unwrap()],
        };

        let expr = Expression::from_proto(&expr_proto, &session).unwrap();
        assert_eq!(expr.id().as_ref(), "vortex.test.foreign_scalar_fn");

        let roundtrip = expr.serialize_proto().unwrap();
        assert_eq!(roundtrip.id, expr_proto.id);
        assert_eq!(roundtrip.metadata(), expr_proto.metadata());
        assert_eq!(roundtrip.children.len(), 1);
    }
}
