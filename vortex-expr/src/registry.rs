use std::sync::LazyLock;

use expr::Expr;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::{VortexResult, vortex_err};
use vortex_proto::expr;

use crate::between::proto::BetweenSerde;
use crate::binary::proto::BinarySerde;
use crate::get_item::proto::GetItemSerde;
use crate::identity::proto::IdentitySerde;
use crate::like::proto::LikeSerde;
use crate::literal::proto::LiteralSerde;
use crate::merge::proto::MergeSerde;
use crate::not::proto::NotSerde;
use crate::pack::proto::PackSerde;
use crate::select::proto::SelectSerde;
use crate::{ExprDeserialize, ExprRef};

const EXPRESSIONS: &[&'static dyn ExprDeserialize] = &[
    &BetweenSerde,
    &BinarySerde,
    &GetItemSerde,
    &IdentitySerde,
    &LikeSerde,
    &LiteralSerde,
    &MergeSerde,
    &NotSerde,
    &PackSerde,
    &SelectSerde,
];

static EXPRESSIONS_REGISTRY: LazyLock<HashMap<&'static str, &&'static dyn ExprDeserialize>> =
    LazyLock::new(move || EXPRESSIONS.iter().map(|e| (e.id(), e)).collect());

pub fn deserialize_expr(expr: &Expr) -> VortexResult<ExprRef> {
    let expr_id = expr.id.as_str();
    let deserializer = EXPRESSIONS_REGISTRY
        .get(expr_id)
        .ok_or_else(|| vortex_err!("unknown expression id: {}", expr_id))?;
    let children = expr
        .children
        .iter()
        .map(deserialize_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    deserializer.deserialize(
        expr.kind
            .as_ref()
            .ok_or_else(|| vortex_err!("empty_kind"))?
            .kind
            .as_ref()
            .ok_or_else(|| vortex_err!("empty kind inner"))?,
        children,
    )
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::compute::{BetweenOptions, StrictComparison};
    use vortex_proto::expr::Expr;

    use crate::{
        Between, ExprRef, VortexExprExt, and, deserialize_expr, eq, get_item, ident, lit, or,
    };

    #[test]
    fn expression_serde() {
        let expr: ExprRef = or(
            and(
                Between::between(
                    lit(1),
                    ident(),
                    get_item("a", ident()),
                    BetweenOptions {
                        lower_strict: StrictComparison::Strict,
                        upper_strict: StrictComparison::Strict,
                    },
                ),
                lit(1),
            ),
            eq(lit(1), ident()),
        );

        let s_expr = expr.serialize().unwrap();
        let buf = s_expr.encode_to_vec();
        let s_expr = Expr::decode(buf.as_slice()).unwrap();
        let deser_expr = deserialize_expr(&s_expr).unwrap();

        assert_eq!(&deser_expr, &expr);
    }
}
