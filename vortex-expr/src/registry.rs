#![cfg(feature = "proto")]

use std::sync::LazyLock;

use vortex_array::aliases::hash_map::HashMap;
use vortex_error::{VortexResult, vortex_err};

use crate::binary::proto::BinarySerde;
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
    &IdentitySerde,
    &NotSerde,
    &SelectSerde,
    &PackSerde,
    &MergeSerde,
];

static EXPRESSIONS_REGISTRY: LazyLock<HashMap<&'static str, &&'static dyn ExprDeserialize>> =
    LazyLock::new(move || EXPRESSIONS.into_iter().map(|e| (e.id(), e)).collect());

pub fn deserialize_expr(expr: &vortex_proto::expr::Expr) -> VortexResult<ExprRef> {
    let id = expr.id.as_str();
    let deser = EXPRESSIONS_REGISTRY
        .get(id)
        .ok_or_else(|| vortex_err!("unknown expression id: {}", id))?;
    let children = expr
        .children
        .iter()
        .map(deserialize_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(deser.deserialize(
        &expr.attributes.as_ref().unwrap().kind.as_ref().unwrap(),
        children,
    )?)
}
