// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod foreign;

pub use foreign::*;
use itertools::Itertools;
use vortex_array::proto::expr as pb;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::Expression;
use crate::expression::unbound_variable;

pub trait ExprSerializeProtoExt {
    /// Serialize the expression to its protobuf representation.
    fn serialize_proto(&self) -> VortexResult<pb::Expr>;
}

/// The wire id for [`Expression::Root`], retained from when `Root` was a scalar function so that
/// already-serialized expressions keep round-tripping.
pub(crate) const ROOT_ID: &str = "vortex.root";

const MAX_EXPR_PROTO_DEPTH: usize = 100;

fn ensure_proto_depth(depth: usize) -> VortexResult<()> {
    vortex_ensure!(
        depth <= MAX_EXPR_PROTO_DEPTH,
        "expression nesting depth exceeds the maximum of {MAX_EXPR_PROTO_DEPTH}"
    );
    Ok(())
}

impl ExprSerializeProtoExt for Expression {
    fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        self.serialize_proto_at_depth(1)
    }
}

impl Expression {
    fn serialize_proto_at_depth(&self, depth: usize) -> VortexResult<pb::Expr> {
        ensure_proto_depth(depth)?;
        if let Expression::Variable(name) = self {
            return Err(unbound_variable(name));
        }
        let Some(scalar_fn) = self.as_scalar() else {
            return Ok(pb::Expr {
                id: ROOT_ID.to_string(),
                children: vec![],
                metadata: Some(vec![]),
            });
        };

        let children = self
            .children()
            .iter()
            .map(|child| child.serialize_proto_at_depth(depth + 1))
            .try_collect()?;

        let metadata = scalar_fn.options().serialize()?.ok_or_else(|| {
            vortex_err!(
                "Expression '{}' is not serializable: {}",
                scalar_fn.id(),
                self
            )
        })?;

        Ok(pb::Expr {
            id: scalar_fn.id().to_string(),
            children,
            metadata: Some(metadata),
        })
    }
}

impl Expression {
    pub fn from_proto(expr: &pb::Expr, session: &VortexSession) -> VortexResult<Expression> {
        Self::from_proto_at_depth(expr, session, 1)
    }

    fn from_proto_at_depth(
        expr: &pb::Expr,
        session: &VortexSession,
        depth: usize,
    ) -> VortexResult<Expression> {
        ensure_proto_depth(depth)?;
        // Root is not a registered scalar fn, so it must be resolved before the registry lookup.
        if expr.id == ROOT_ID {
            vortex_ensure!(
                expr.children.is_empty(),
                "root expression must have no children, got {}",
                expr.children.len()
            );
            return Ok(Expression::Root);
        }

        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let expr_id = ScalarFnId::new(expr.id.as_str());
        let children = expr
            .children
            .iter()
            .map(|e| Expression::from_proto_at_depth(e, session, depth + 1))
            .collect::<VortexResult<Vec<_>>>()?;

        let scalar_fn = if let Some(vtable) = session.scalar_fns().registry().get(&expr_id) {
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
    use std::sync::LazyLock;

    use prost::Message;
    use vortex_array::proto::expr as pb;
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::ExprSerializeProtoExt;
    use super::MAX_EXPR_PROTO_DEPTH;
    use crate::Expression;
    use crate::and;
    use crate::eq;
    use crate::get_item;
    use crate::lit;
    use crate::lt;
    use crate::not;
    use crate::or;
    use crate::root;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ScalarFnSession>();
        session.allow_unknown();
        session
    });

    #[test]
    fn expression_serde() {
        let expr: Expression = or(
            and(lt(lit(1), get_item("a", root())), lit(1)),
            eq(lit(1), root()),
        );

        let s_expr = expr.serialize_proto().unwrap();
        let buf = s_expr.encode_to_vec();
        let s_expr = pb::Expr::decode(buf.as_slice()).unwrap();
        let deser_expr = Expression::from_proto(&s_expr, &VortexSession::empty()).unwrap();

        assert_eq!(&deser_expr, &expr);
    }

    #[test]
    fn unknown_expression_id_allow_unknown() {
        let expr_proto = pb::Expr {
            id: "vortex.test.foreign_scalar_fn".to_string(),
            metadata: Some(vec![1, 2, 3, 4]),
            children: vec![root().serialize_proto().unwrap()],
        };

        let expr = Expression::from_proto(&expr_proto, &SESSION).unwrap();
        assert_eq!(
            expr.as_scalar().map(|f| f.id().as_ref().to_string()),
            Some("vortex.test.foreign_scalar_fn".to_string())
        );

        let roundtrip = expr.serialize_proto().unwrap();
        assert_eq!(roundtrip.id, expr_proto.id);
        assert_eq!(roundtrip.metadata(), expr_proto.metadata());
        assert_eq!(roundtrip.children, expr_proto.children);
    }

    fn nested_not(depth: usize) -> Expression {
        (1..depth).fold(root(), |expr, _| not(expr))
    }

    fn nested_not_proto(depth: usize) -> VortexResult<pb::Expr> {
        let not_proto = not(root()).serialize_proto()?;
        Ok(
            (1..depth).fold(root().serialize_proto()?, |child, _| pb::Expr {
                children: vec![child],
                ..not_proto.clone()
            }),
        )
    }

    #[test]
    fn serialize_proto_depth_limit() -> VortexResult<()> {
        nested_not(MAX_EXPR_PROTO_DEPTH).serialize_proto()?;

        let err = nested_not(MAX_EXPR_PROTO_DEPTH + 1)
            .serialize_proto()
            .unwrap_err();
        assert!(err.to_string().contains("nesting depth"), "{err}");
        Ok(())
    }

    #[test]
    fn from_proto_depth_limit() -> VortexResult<()> {
        let expr = Expression::from_proto(&nested_not_proto(MAX_EXPR_PROTO_DEPTH)?, &SESSION)?;
        assert_eq!(expr, nested_not(MAX_EXPR_PROTO_DEPTH));

        let err = Expression::from_proto(&nested_not_proto(MAX_EXPR_PROTO_DEPTH + 1)?, &SESSION)
            .unwrap_err();
        assert!(err.to_string().contains("nesting depth"), "{err}");
        Ok(())
    }

    #[test]
    fn encoded_proto_depth_limit() -> VortexResult<()> {
        let expr = nested_not(MAX_EXPR_PROTO_DEPTH);
        let buf = expr.serialize_proto()?.encode_to_vec();
        let decoded = pb::Expr::decode(buf.as_slice())?;
        assert_eq!(Expression::from_proto(&decoded, &SESSION)?, expr);

        let buf = nested_not_proto(MAX_EXPR_PROTO_DEPTH + 1)?.encode_to_vec();
        let decoded = pb::Expr::decode(buf.as_slice())?;
        let err = Expression::from_proto(&decoded, &SESSION).unwrap_err();
        assert!(err.to_string().contains("nesting depth"), "{err}");
        Ok(())
    }

    #[test]
    fn deeply_nested_expression_errors() -> VortexResult<()> {
        let depth = 10_000;
        assert!(nested_not(depth).serialize_proto().is_err());

        let mut proto = nested_not_proto(depth)?;
        assert!(Expression::from_proto(&proto, &SESSION).is_err());
        while let Some(child) = proto.children.pop() {
            proto = child;
        }
        Ok(())
    }
}
