// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comparisons with the float semantics of DuckDB.
//!
//! DuckDB makes `-0.0` equal to `+0.0`, makes all `NaN` values equal, and orders `NaN` above all
//! other values. By default, Vortex orders `NaN` values by sign and payload, so each comparison
//! that DuckDB pushes down uses [`compare_mode`]. DuckDB does not apply a pushed table filter again, so
//! a comparison with other semantics would add or drop rows.

use vortex::dtype::DType;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::compare_with_mode;
use vortex::scalar_fn::builtin::operators::CompareMode;
use vortex::scalar_fn::builtin::operators::CompareOperator;
use vortex::scalar_fn::builtin::operators::FloatOrder;

/// The comparison semantics of DuckDB.
const DUCKDB_MODE: CompareMode = CompareMode::new().with_float_order(FloatOrder::Canonical);

/// Returns the compare mode of DuckDB for operands of `dtype`. It is the default mode when
/// `dtype` holds no floats.
pub(super) fn compare_mode(dtype: &DType) -> CompareMode {
    DUCKDB_MODE.for_dtype(dtype)
}

/// Returns `lhs <operator> rhs` for operands of `dtype`.
pub(super) fn compare(
    operator: CompareOperator,
    dtype: &DType,
    lhs: Expression,
    rhs: Expression,
) -> Expression {
    compare_with_mode(operator, compare_mode(dtype), lhs, rhs)
}

/// Returns `lower <(=) input <(=) upper` for operands of `dtype`, as a conjunction of comparisons
/// in `mode`.
///
/// The `Between` scalar function compares in the default mode. Use this function when the compare
/// mode of `dtype` is a different mode.
pub(super) fn between_with_mode(
    mode: CompareMode,
    input: Expression,
    lower: Expression,
    lower_inclusive: bool,
    upper: Expression,
    upper_inclusive: bool,
) -> Expression {
    let lower_operator = if lower_inclusive {
        CompareOperator::Gte
    } else {
        CompareOperator::Gt
    };
    let upper_operator = if upper_inclusive {
        CompareOperator::Lte
    } else {
        CompareOperator::Lt
    };

    and(
        compare_with_mode(lower_operator, mode, input.clone(), lower),
        compare_with_mode(upper_operator, mode, input, upper),
    )
}
