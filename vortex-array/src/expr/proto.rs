// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_proto::dtype as dtype_pb;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::BoundExpr;
use crate::scalar::Scalar;
use crate::scalar_fn::ForeignScalarFnVTable;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::session::ScalarFnSessionExt;

pub trait ExprSerializeProtoExt {
    /// Serialize the expression to its protobuf representation.
    fn serialize_proto(&self) -> VortexResult<pb::Expr>;
}

impl ExprSerializeProtoExt for BoundExpr {
    fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        match self {
            BoundExpr::Root(dtype) => Ok(pb::Expr {
                id: "vortex.root".to_string(),
                children: vec![],
                metadata: Some(dtype_pb::DType::try_from(dtype)?.encode_to_vec()),
            }),
            BoundExpr::Literal(scalar) => Ok(pb::Expr {
                id: "vortex.literal".to_string(),
                children: vec![],
                metadata: Some(
                    pb::LiteralOpts {
                        value: Some(scalar.into()),
                    }
                    .encode_to_vec(),
                ),
            }),
            BoundExpr::Placeholder(placeholder) => {
                vortex_bail!("Placeholder '{}' is not serializable", placeholder.id())
            }
            BoundExpr::Call(call) => {
                let children = call
                    .args()
                    .iter()
                    .map(|child| child.serialize_proto())
                    .try_collect()?;

                let metadata = call.function().options().serialize()?.ok_or_else(|| {
                    vortex_err!(
                        "BoundExpr '{}' is not serializable: {}",
                        call.function().id(),
                        self
                    )
                })?;

                Ok(pb::Expr {
                    id: call.function().id().to_string(),
                    children,
                    metadata: Some(metadata),
                })
            }
        }
    }
}

impl BoundExpr {
    /// Deserialize a bound expression from protobuf.
    ///
    /// Root nodes use the historical `"vortex.root"` id but now store their bound scope dtype in
    /// metadata. Legacy empty-metadata Root protobufs return an explicit error. The expression
    /// protobuf envelope is not embedded in the Vortex file format; in-repo consumers only
    /// round-trip it in tests.
    pub fn from_proto(expr: &pb::Expr, session: &VortexSession) -> VortexResult<BoundExpr> {
        if expr.id == "vortex.literal" {
            vortex_ensure!(
                expr.children.is_empty(),
                "Literal expression expected 0 children, got {}",
                expr.children.len()
            );
            let opts = pb::LiteralOpts::decode(expr.metadata())?;
            return Ok(BoundExpr::Literal(Scalar::from_proto(
                opts.value
                    .as_ref()
                    .ok_or_else(|| vortex_err!("Literal metadata missing value"))?,
                session,
            )?));
        }

        if expr.id == "vortex.root" {
            vortex_ensure!(
                expr.children.is_empty(),
                "Root expression expected 0 children, got {}",
                expr.children.len()
            );
            if expr.metadata().is_empty() {
                vortex_bail!("Root expression metadata missing bound scope dtype");
            }
            let dtype = dtype_pb::DType::decode(expr.metadata())?;
            return Ok(BoundExpr::Root(DType::from_proto(&dtype, session)?));
        }

        let expr_id = ScalarFnId::new(expr.id.as_str());
        let children = expr
            .children
            .iter()
            .map(|e| BoundExpr::from_proto(e, session))
            .collect::<VortexResult<Vec<_>>>()?;

        let scalar_fn = if let Some(vtable) = session.scalar_fns().registry().find(&expr_id) {
            vtable.deserialize(expr.metadata(), session)?
        } else if session.allows_unknown() {
            ForeignScalarFnVTable::make_scalar_fn(expr_id, expr.metadata().to_vec(), children.len())
        } else {
            return Err(vortex_err!("unknown expression id: {}", expr_id));
        };

        BoundExpr::try_new(scalar_fn, children)
    }
}

/// Deserialize a [`BoundExpr`] from the protobuf representation.
#[deprecated(note = "Use BoundExpr::from_proto instead")]
pub fn deserialize_expr_proto(expr: &pb::Expr, session: &VortexSession) -> VortexResult<BoundExpr> {
    BoundExpr::from_proto(expr, session)
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_proto::expr as pb;
    use vortex_session::VortexSession;

    use super::ExprSerializeProtoExt;
    use crate::LEGACY_SESSION;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::BoundExpr;
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
        let scope = DType::Struct(
            StructFields::new(
                ["a"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let expr: BoundExpr = or(
            and(
                between(
                    get_item("a", root(scope.clone())),
                    lit(1),
                    lit(10),
                    BetweenOptions {
                        lower_strict: StrictComparison::Strict,
                        upper_strict: StrictComparison::Strict,
                    },
                ),
                eq(get_item("a", root(scope.clone())), lit(7)),
            ),
            eq(get_item("a", root(scope.clone())), lit(3)),
        );

        let s_expr = expr.serialize_proto().unwrap();
        let buf = s_expr.encode_to_vec();
        let s_expr = pb::Expr::decode(buf.as_slice()).unwrap();
        let deser_expr = BoundExpr::from_proto(&s_expr, &LEGACY_SESSION).unwrap();

        assert_eq!(&deser_expr, &expr);
        assert_eq!(
            root(scope.clone()).serialize_proto().unwrap().id,
            "vortex.root"
        );
        assert_eq!(
            BoundExpr::from_proto(
                &root(scope.clone()).serialize_proto().unwrap(),
                &LEGACY_SESSION
            )
            .unwrap(),
            root(scope)
        );
    }

    #[test]
    fn legacy_empty_root_metadata_errors() {
        let expr_proto = pb::Expr {
            id: "vortex.root".to_string(),
            metadata: Some(vec![]),
            children: vec![],
        };

        assert!(BoundExpr::from_proto(&expr_proto, &LEGACY_SESSION).is_err());
    }

    #[test]
    fn unknown_expression_id_allow_unknown() {
        let session = VortexSession::empty()
            .with::<ScalarFnSession>()
            .allow_unknown();

        let expr_proto = pb::Expr {
            id: "vortex.test.foreign_scalar_fn".to_string(),
            metadata: Some(vec![1, 2, 3, 4]),
            children: vec![
                root(DType::Bool(Nullability::NonNullable))
                    .serialize_proto()
                    .unwrap(),
            ],
        };

        let expr = BoundExpr::from_proto(&expr_proto, &session).unwrap();
        assert_eq!(
            expr.as_call().unwrap().function().id().as_ref(),
            "vortex.test.foreign_scalar_fn"
        );

        let roundtrip = expr.serialize_proto().unwrap();
        assert_eq!(roundtrip.id, expr_proto.id);
        assert_eq!(roundtrip.metadata(), expr_proto.metadata());
        assert_eq!(roundtrip.children.len(), 1);
    }
}
