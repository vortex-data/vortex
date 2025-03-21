use std::sync::LazyLock;

use expr::Expr;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::{VortexResult, vortex_err};
use vortex_proto::expr;

use crate::binary::proto::BinarySerde;
use crate::get_item::proto::GetItemSerde;
use crate::identity::proto::IdentitySerde;
use crate::literal::proto::LiteralSerde;
use crate::merge::proto::MergeSerde;
use crate::not::proto::NotSerde;
use crate::pack::proto::PackSerde;
use crate::select::proto::SelectSerde;
use crate::{ExprDeserialize, ExprRef};

const EXPRESSIONS: &[&'static dyn ExprDeserialize] = &[
    &BinarySerde,
    &LiteralSerde,
    &GetItemSerde,
    &IdentitySerde,
    &NotSerde,
    &SelectSerde,
    &PackSerde,
    &MergeSerde,
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
