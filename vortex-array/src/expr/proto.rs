// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::expr::Expression;
use crate::expr::Lambda;
use crate::expr::Variable;
use crate::scalar_fn::ForeignScalarFnVTable;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::session::ScalarFnSessionExt;

pub trait ExprSerializeProtoExt {
    /// Serialize the expression to its protobuf representation.
    fn serialize_proto(&self) -> VortexResult<pb::Expr>;
}

/// The wire id for [`Expression::Root`], retained from when `Root` was a scalar function so that
/// already-serialized expressions keep round-tripping.
pub(crate) const ROOT_ID: &str = "vortex.root";

/// The wire id for [`Expression::Variable`].
pub(crate) const VARIABLE_ID: &str = "vortex.var";

/// The wire id for [`Expression::Lambda`].
pub(crate) const LAMBDA_ID: &str = "vortex.lambda";

/// The wire id for the dedicated list-transform syntax node.
pub(crate) const LIST_TRANSFORM_ID: &str = "vortex.list_transform";

impl Lambda {
    /// Serialize this lambda to its protobuf representation.
    fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        Ok(pb::Expr {
            id: LAMBDA_ID.to_string(),
            children: vec![self.body().serialize_proto()?],
            metadata: Some(
                pb::LambdaOpts {
                    params: self
                        .params()
                        .iter()
                        .map(|variable| variable.name().to_string())
                        .collect(),
                }
                .encode_to_vec(),
            ),
        })
    }

    /// Deserialize a lambda expression whose id is [`LAMBDA_ID`].
    fn from_proto(expr: &pb::Expr, session: &VortexSession) -> VortexResult<Self> {
        vortex_ensure!(
            expr.children.len() == 1,
            "a lambda must have exactly one child, its body, got {}",
            expr.children.len()
        );
        let options = pb::LambdaOpts::decode(expr.metadata())?;
        Self::try_new(
            options.params.into_iter().map(Variable::new),
            Expression::from_proto(&expr.children[0], session)?,
        )
    }
}

impl Variable {
    /// Serialize this variable to its protobuf representation.
    fn serialize_proto(&self) -> pb::Expr {
        pb::Expr {
            id: VARIABLE_ID.to_string(),
            children: vec![],
            metadata: Some(
                pb::VariableOpts {
                    name: self.name().to_string(),
                }
                .encode_to_vec(),
            ),
        }
    }

    /// Deserialize a variable expression whose id is [`VARIABLE_ID`].
    fn from_proto(expr: &pb::Expr) -> VortexResult<Self> {
        vortex_ensure!(
            expr.children.is_empty(),
            "a variable must have no children, got {}",
            expr.children.len()
        );
        let options = pb::VariableOpts::decode(expr.metadata())?;
        Ok(Self::new(options.name))
    }
}

fn serialize_scalar(
    expression: &Expression,
    scalar_fn: &ScalarFnRef,
    children: &[Expression],
) -> VortexResult<pb::Expr> {
    let children = children
        .iter()
        .map(|child| child.serialize_proto())
        .try_collect()?;

    let metadata = scalar_fn.options().serialize()?.ok_or_else(|| {
        vortex_err!(
            "Expression '{}' is not serializable: {expression}",
            scalar_fn.id()
        )
    })?;

    Ok(pb::Expr {
        id: scalar_fn.id().to_string(),
        children,
        metadata: Some(metadata),
    })
}

impl ExprSerializeProtoExt for Expression {
    fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        match self {
            Expression::Root => Ok(pb::Expr {
                id: ROOT_ID.to_string(),
                children: vec![],
                metadata: Some(vec![]),
            }),
            Expression::Variable(variable) => Ok(variable.serialize_proto()),
            Expression::Lambda(lambda) => lambda.serialize_proto(),
            Expression::ListTransform { children } => Ok(pb::Expr {
                id: LIST_TRANSFORM_ID.to_string(),
                children: vec![
                    children[0].serialize_proto()?,
                    children[1].serialize_proto()?,
                ],
                metadata: Some(vec![]),
            }),
            Expression::Scalar {
                scalar_fn,
                children,
            } => serialize_scalar(self, scalar_fn, children),
        }
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

        if expr.id == VARIABLE_ID {
            return Ok(Variable::from_proto(expr)?.into());
        }

        if expr.id == LAMBDA_ID {
            return Ok(Lambda::from_proto(expr, session)?.into());
        }

        if expr.id == LIST_TRANSFORM_ID {
            vortex_ensure!(
                expr.children.len() == 2,
                "list_transform() must have a list and lambda child, got {}",
                expr.children.len()
            );
            return Expression::try_new_list_transform(
                Expression::from_proto(&expr.children[0], session)?,
                Expression::from_proto(&expr.children[1], session)?,
            );
        }

        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let expr_id = ScalarFnId::new(expr.id.as_str());
        let children = expr
            .children
            .iter()
            .map(|e| Expression::from_proto(e, session))
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
    use prost::Message;
    use vortex_error::VortexResult;
    use vortex_proto::expr as pb;
    use vortex_session::VortexSession;

    use super::ExprSerializeProtoExt;
    use crate::array_session;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lambda;
    use crate::expr::lit;
    use crate::expr::or;
    use crate::expr::root;
    use crate::expr::var;
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
    fn variable_serde() -> VortexResult<()> {
        let expression = var("value");
        let encoded = expression.serialize_proto()?.encode_to_vec();
        let proto = pb::Expr::decode(encoded.as_slice())?;

        assert_eq!(
            Expression::from_proto(&proto, &array_session())?,
            expression
        );
        Ok(())
    }

    #[test]
    fn variable_rejects_children() -> VortexResult<()> {
        let mut proto = var("value").serialize_proto()?;
        proto.children.push(root().serialize_proto()?);

        assert!(Expression::from_proto(&proto, &array_session()).is_err());
        Ok(())
    }

    #[test]
    fn lambda_serde() -> VortexResult<()> {
        let expression = lambda(["x", "y"], eq(var("x"), var("y")))?;
        let encoded = expression.serialize_proto()?.encode_to_vec();
        let proto = pb::Expr::decode(encoded.as_slice())?;

        assert_eq!(
            Expression::from_proto(&proto, &array_session())?,
            expression
        );
        Ok(())
    }

    #[test]
    fn lambda_requires_exactly_one_body() -> VortexResult<()> {
        let mut without_body = lambda(["x"], var("x"))?.serialize_proto()?;
        without_body.children.clear();
        assert!(Expression::from_proto(&without_body, &array_session()).is_err());

        let mut with_two_bodies = lambda(["x"], var("x"))?.serialize_proto()?;
        with_two_bodies.children.push(root().serialize_proto()?);
        assert!(Expression::from_proto(&with_two_bodies, &array_session()).is_err());
        Ok(())
    }

    #[test]
    fn unknown_expression_id_allow_unknown() {
        let session = VortexSession::empty().with::<ScalarFnSession>();
        session.allow_unknown();

        let expr_proto = pb::Expr {
            id: "vortex.test.foreign_scalar_fn".to_string(),
            metadata: Some(vec![1, 2, 3, 4]),
            children: vec![root().serialize_proto().unwrap()],
        };

        let expr = Expression::from_proto(&expr_proto, &session).unwrap();
        assert_eq!(
            expr.as_scalar().map(|f| f.id().as_ref().to_string()),
            Some("vortex.test.foreign_scalar_fn".to_string())
        );

        let roundtrip = expr.serialize_proto().unwrap();
        assert_eq!(roundtrip.id, expr_proto.id);
        assert_eq!(roundtrip.metadata(), expr_proto.metadata());
        assert_eq!(roundtrip.children.len(), 1);
    }
}
