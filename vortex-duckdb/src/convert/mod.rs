// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod dtype;
mod expr;
mod scalar;
mod table_filter;
mod vector;

use std::sync::Arc;

pub use dtype::FromLogicalType;
pub use expr::can_push_expression;
pub use expr::try_from_bound_expression;
pub use scalar::*;
pub use table_filter::try_from_table_filter;
pub use table_filter::try_from_virtual_column_filter;
pub use vector::data_chunk_to_vortex;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::expr::BoundExpr;
use vortex::expr::try_and_collect;
use vortex::expr::try_or_collect;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::operators::Operator;

/// Collects expressions into a single balanced binary tree under `operator`, fallibly.
///
/// Balanced like [`vortex::expr::and_collect`], avoiding the deep nesting a left fold would
/// produce for wide pushed conjunctions; fallible because the inputs come from the engine.
pub(crate) fn collect_binary(
    exprs: impl IntoIterator<Item = BoundExpr>,
    operator: Operator,
) -> VortexResult<Option<BoundExpr>> {
    match operator {
        Operator::And => try_and_collect(exprs),
        Operator::Or => try_or_collect(exprs),
        _ => vortex_bail!("collect_binary only supports And/Or, got {}", operator),
    }
}

/// Builds a list scalar from same-dtyped elements, fallibly (engine-supplied IN lists).
pub(crate) fn try_list_scalar(elements: Vec<Scalar>) -> VortexResult<Scalar> {
    let Some(dtype) = elements.first().map(|scalar| scalar.dtype().clone()) else {
        vortex_bail!("IN list must have at least one value");
    };

    let values = elements
        .into_iter()
        .map(|scalar| {
            vortex_ensure!(
                scalar.dtype() == &dtype,
                "IN list values must have matching dtypes, got {} and {}",
                dtype,
                scalar.dtype()
            );
            Ok(scalar.into_value())
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Scalar::try_new(
        DType::List(Arc::new(dtype), Nullability::Nullable),
        Some(ScalarValue::Tuple(values)),
    )
}
