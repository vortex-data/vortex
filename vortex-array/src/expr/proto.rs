// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::expr::Expression;
use crate::expr::expression::unbound_variable;
use crate::proto::expr as pb;
use crate::scalar_fn::ErasedOptions;
use crate::scalar_fn::ForeignScalarFnOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::session::ScalarFnSessionExt;

pub trait ExprSerializeProtoExt {
    /// Serialize the expression to its protobuf representation.
    ///
    /// The scalar functions the expression calls are looked up in `session` to serialize their
    /// options.
    fn serialize_proto(&self, session: &VortexSession) -> VortexResult<pb::Expr>;
}

/// The wire id for [`Expression::Root`], retained from when `Root` was a scalar function so that
/// already-serialized expressions keep round-tripping.
pub(crate) const ROOT_ID: &str = "vortex.root";

impl ExprSerializeProtoExt for Expression {
    fn serialize_proto(&self, session: &VortexSession) -> VortexResult<pb::Expr> {
        let (id, options, children) = match self {
            Expression::Variable(name) => return Err(unbound_variable(name)),
            Expression::Root => {
                return Ok(pb::Expr {
                    id: ROOT_ID.to_string(),
                    children: vec![],
                    metadata: Some(vec![]),
                });
            }
            Expression::Scalar {
                id,
                options,
                children,
            } => (*id, options, children),
        };

        let children = children
            .iter()
            .map(|child| child.serialize_proto(session))
            .try_collect()?;

        // Options read for an unknown id are written back as they were read.
        let metadata = match options.downcast_ref::<ForeignScalarFnOptions>() {
            Some(foreign) => Some(foreign.metadata().to_vec()),
            None => session
                .scalar_fn_plugin(id)?
                .serialize_any(options.as_any())?,
        };
        let metadata =
            metadata.ok_or_else(|| vortex_err!("Expression '{id}' is not serializable: {self}"))?;

        Ok(pb::Expr {
            id: id.to_string(),
            children,
            metadata: Some(metadata),
        })
    }
}

impl Expression {
    pub fn from_proto(expr: &pb::Expr, session: &VortexSession) -> VortexResult<Expression> {
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
            .map(|e| Expression::from_proto(e, session))
            .collect::<VortexResult<Vec<_>>>()?;

        if let Some(plugin) = session.scalar_fns().registry().get(&expr_id) {
            let scalar_fn = plugin.deserialize(expr.metadata(), session)?;
            Expression::try_new(scalar_fn, children)
        } else if session.allows_unknown() {
            let options = ForeignScalarFnOptions::new(expr.metadata().to_vec(), children.len());
            Ok(Expression::new(
                expr_id,
                ErasedOptions::new(options),
                children,
            ))
        } else {
            Err(vortex_err!("unknown expression id: {}", expr_id))
        }
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
    use vortex_session::VortexSession;

    use super::ExprSerializeProtoExt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::or;
    use crate::expr::root;
    use crate::proto::expr as pb;
    use crate::scalar_fn::ForeignScalarFnVTable;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;
    use crate::scalar_fn::session::ScalarFnSession;

    fn root_dtype() -> DType {
        DType::Bool(Nullability::NonNullable)
    }

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ScalarFnSession>();
        session.allow_unknown();
        session
    });

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

        let s_expr = expr.serialize_proto(&SESSION).unwrap();
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
            children: vec![root().serialize_proto(&SESSION).unwrap()],
        };

        let expr = Expression::from_proto(&expr_proto, &SESSION).unwrap();
        assert_eq!(
            expr.id().map(|id| id.as_ref().to_string()),
            Some("vortex.test.foreign_scalar_fn".to_string())
        );
        // The placeholder binds, so the rest of the expression can be type-checked.
        let bound = expr.bind(&root_dtype(), &SESSION).unwrap();
        assert!(bound.is::<ForeignScalarFnVTable>());

        let roundtrip = expr.serialize_proto(&SESSION).unwrap();
        assert_eq!(roundtrip.id, expr_proto.id);
        assert_eq!(roundtrip.metadata(), expr_proto.metadata());
        assert_eq!(roundtrip.children, expr_proto.children);
    }
}
