// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comparisons with the float semantics of DuckDB.
//!
//! DuckDB makes `-0.0` equal to `+0.0`, makes all `NaN` values equal, and orders `NaN` above all
//! other values. Vortex orders floats in total order by default, so each float comparison that
//! DuckDB pushes down uses [`DUCKDB_COMPARE_MODE`]. DuckDB does not apply a pushed table filter
//! again, so a comparison with other semantics would add or drop rows.

use vortex::dtype::DType;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::compare_with_mode;
use vortex::expr::lit;
use vortex::expr::or_collect;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::CompareMode;
use vortex::scalar_fn::fns::operators::CompareOperator;
use vortex::scalar_fn::fns::operators::NanOrder;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::operators::SignedZeros;

/// The float comparison semantics of DuckDB.
pub(super) const DUCKDB_COMPARE_MODE: CompareMode =
    CompareMode::new(SignedZeros::Equal, NanOrder::Canonical);

/// Returns the compare mode for operands of `dtype`.
pub(super) fn compare_mode(dtype: &DType) -> CompareMode {
    if holds_float(dtype) {
        DUCKDB_COMPARE_MODE
    } else {
        CompareMode::TOTAL_ORDER
    }
}

/// Returns true if values of `dtype` hold floats, at the top level or in a nested type.
///
/// DuckDB compares the floats in structs, lists and maps with the same semantics as top-level
/// floats.
pub(super) fn holds_float(dtype: &DType) -> bool {
    match dtype {
        DType::Primitive(ptype, _) => ptype.is_float(),
        DType::Extension(ext) => holds_float(ext.storage_dtype()),
        DType::List(element, _) | DType::FixedSizeList(element, ..) => holds_float(element),
        DType::Struct(fields, _) => fields.fields().any(|field| holds_float(&field)),
        DType::Map(map, _) => holds_float(&map.key_dtype()) || holds_float(&map.value_dtype()),
        DType::Null
        | DType::Bool(_)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_)
        | DType::Union(..)
        | DType::Variant(_) => false,
    }
}

/// Returns `lhs <operator> rhs` for operands of `dtype`.
pub(super) fn compare(
    operator: Operator,
    dtype: &DType,
    lhs: Expression,
    rhs: Expression,
) -> Expression {
    let mode = compare_mode(dtype);

    match operator.to_compare_operator() {
        Some(operator) if !mode.is_total_order() => compare_with_mode(operator, mode, lhs, rhs),
        _ => Binary.new_expr(operator.into(), [lhs, rhs]),
    }
}

/// Returns `element IN (values)` for float values, as a disjunction of equalities.
///
/// The `IN` scalar function compares the bits of floats, so it cannot express DuckDB semantics.
/// A null value makes a non-match null, as in SQL.
pub(super) fn float_in_list(element: &Expression, values: Vec<Scalar>) -> Expression {
    or_collect(values.into_iter().map(|value| {
        compare_with_mode(
            CompareOperator::Eq,
            DUCKDB_COMPARE_MODE,
            element.clone(),
            lit(value),
        )
    }))
    .unwrap_or_else(|| lit(false))
}

/// Returns `lower <(=) input <(=) upper` for float operands, as a conjunction of comparisons.
///
/// The `Between` scalar function compares in total order, so it cannot express DuckDB semantics.
pub(super) fn float_between(
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
        compare_with_mode(lower_operator, DUCKDB_COMPARE_MODE, input.clone(), lower),
        compare_with_mode(upper_operator, DUCKDB_COMPARE_MODE, input, upper),
    )
}
