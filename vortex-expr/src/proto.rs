use itertools::Itertools;
use vortex_error::{VortexResult, vortex_err};
use vortex_proto::exprs as pb;

use crate::registry::ExprRegistry;
use crate::{ExprRef, VortexExprExt};

pub trait ExprSerializeProtoExt {
    /// Serialize the expression to its protobuf representation.
    fn serialize_proto(&self) -> VortexResult<pb::Expr>;
}

impl ExprSerializeProtoExt for ExprRef {
    fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        let children = self
            .children()
            .into_iter()
            .map(|child| child.serialize_proto())
            .try_collect()?;

        let options = self
            .serialize_options()
            .ok_or_else(|| vortex_err!("Expression is not serializable {}", self))?;

        Ok(pb::Expr {
            id: self.id().to_string(),
            children,
            options,
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

    encoding.deserialize(expr.options().unwrap_or(&[]), children)
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::compute::{BetweenOptions, StrictComparison};
    use vortex_proto::exprs as pb;

    use crate::proto::ExprSerializeProtoExt;
    use crate::{Between, ExprRef, and, deserialize_expr_proto, eq, get_item, lit, or, root};

    #[test]
    fn expression_serde() {
        let registry = ExprRegistry::default();
        let expr: ExprRef = or(
            and(
                Between::between(
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
