// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::{VortexResult, vortex_err};
use vortex_proto::expr as pb;

use crate::registry::ExprRegistry;
use crate::{ExprRef, VortexExpr};

pub trait ExprSerializeProtoExt {
    /// Serialize the expression to its protobuf representation.
    fn serialize_proto(&self) -> VortexResult<pb::Expr>;
}

impl ExprSerializeProtoExt for dyn VortexExpr + '_ {
    fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        let children = self
            .children()
            .into_iter()
            .map(|child| child.serialize_proto())
            .try_collect()?;

        let metadata = self.metadata().ok_or_else(|| {
            vortex_err!("Expression '{}' is not serializable: {}", self.id(), self)
        })?;

        Ok(pb::Expr {
            id: self.id().to_string(),
            children,
            metadata: Some(metadata),
        })
    }
}

/// Deserialize a [`ExprRef`] from the protobuf representation.
pub fn deserialize_expr_proto(expr: &pb::Expr, registry: &ExprRegistry) -> VortexResult<ExprRef> {
    let expr_id = expr.id.as_str();
    let encoding = registry
        .get(expr_id)
        .ok_or_else(|| vortex_err!("unknown expression id: {}", expr_id))?;

    let children = expr
        .children
        .iter()
        .map(|e| deserialize_expr_proto(e, registry))
        .collect::<VortexResult<Vec<_>>>()?;

    encoding.build(expr.metadata(), children)
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::compute::{BetweenOptions, StrictComparison};
    use vortex_proto::expr as pb;

    use crate::proto::{ExprSerializeProtoExt, deserialize_expr_proto};
    use crate::registry::ExprRegistryExt;
    use crate::{ExprRef, ExprRegistry, and, between, eq, get_item, lit, or, root};

    #[test]
    fn expression_serde() {
        let registry = ExprRegistry::default();
        let expr: ExprRef = or(
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
        let deser_expr = deserialize_expr_proto(&s_expr, &registry).unwrap();

        assert_eq!(&deser_expr, &expr);
    }
}
