// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One late-materialization key preserves the association between a file and its rows.

use std::collections::BTreeSet;

use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::DecimalDType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;

use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb::ExpressionClass;
use crate::duckdb::ExpressionRef;
use crate::duckdb::ExtractedValue;
use crate::duckdb::TableFilterClass;
use crate::duckdb::TableFilterRef;
use crate::duckdb::ValueRef;

// Kept in sync with VORTEX_ROW_ID_COLUMN in cpp/include/table_function.hpp.
pub(crate) const ROW_ID_COLUMN_IDX: u64 = 9223372036854775811;
pub(crate) const ROW_ID_NAME: &str = "__vortex_row_id";

fn encode(file: u64, row: u64) -> VortexResult<i128> {
    i128::from(file)
        .checked_mul(1i128 << 64)
        .and_then(|prefix| prefix.checked_add(i128::from(row)))
        .ok_or_else(|| vortex_err!("file index exceeds the packed row identifier range"))
}

#[derive(Debug)]
pub(crate) struct RowIds(BTreeSet<i128>);

impl RowIds {
    pub(crate) fn from_filter(filter: &TableFilterRef) -> Option<Self> {
        match filter.as_class() {
            TableFilterClass::Optional(child) => Self::from_filter(child),
            TableFilterClass::InFilter(values) => Self::from_values(values.iter()),
            TableFilterClass::ConstantComparison(comparison)
                if comparison.operator
                    == DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL =>
            {
                Self::from_values([comparison.value].into_iter())
            }
            TableFilterClass::ExpressionRef(expression) => Self::from_expression(expression),
            TableFilterClass::ConjunctionAnd(conjunction) => conjunction
                .children()
                .filter_map(Self::from_filter)
                .reduce(|left, right| Self(left.0.intersection(&right.0).copied().collect())),
            TableFilterClass::ConjunctionOr(conjunction) => {
                let children = conjunction
                    .children()
                    .map(Self::from_filter)
                    .collect::<Option<Vec<_>>>()?;
                Some(Self(
                    children.into_iter().flat_map(|child| child.0).collect(),
                ))
            }
            _ => None,
        }
    }

    fn from_expression(expression: &ExpressionRef) -> Option<Self> {
        match expression.as_class()? {
            ExpressionClass::BoundFunction(function) => Self::from_expression(function.optional()?),
            ExpressionClass::BoundOperator(operator)
                if operator.op == DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN =>
            {
                let mut children = operator.children();
                if !matches!(
                    children.next()?.as_class()?,
                    ExpressionClass::BoundRef | ExpressionClass::BoundColumnRef(_)
                ) {
                    return None;
                }
                let values = children
                    .map(|child| match child.as_class()? {
                        ExpressionClass::BoundConstant(constant) => Some(constant.value),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                Self::from_values(values.into_iter())
            }
            ExpressionClass::BoundComparison(comparison)
                if comparison.op == DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL =>
            {
                let ExpressionClass::BoundConstant(constant) = comparison.right.as_class()? else {
                    return None;
                };
                if !matches!(
                    comparison.left.as_class()?,
                    ExpressionClass::BoundRef | ExpressionClass::BoundColumnRef(_)
                ) {
                    return None;
                }
                Self::from_values([constant.value].into_iter())
            }
            _ => None,
        }
    }

    fn from_values<'a>(values: impl Iterator<Item = &'a ValueRef>) -> Option<Self> {
        let mut ids = BTreeSet::new();
        for value in values {
            let value = match value.extract() {
                ExtractedValue::HugeInt(value) | ExtractedValue::Decimal(_, 0, value) => value,
                ExtractedValue::UBigInt(value) => i128::from(value),
                ExtractedValue::BigInt(value) => i128::from(value),
                ExtractedValue::Null => continue,
                _ => return None,
            };
            if value >= 0 {
                ids.insert(value);
            }
        }
        Some(Self(ids))
    }

    pub(crate) fn selection(
        &self,
        file: u64,
        row_count: u64,
        selection: &Selection,
    ) -> VortexResult<Selection> {
        let first = encode(file, 0)?;
        let last = encode(file, u64::MAX)?;
        let mut rows = Vec::new();
        for &id in self.0.range(first..=last) {
            let row = u64::try_from(id - first)?;
            let keep = match selection {
                Selection::All => true,
                Selection::IncludeByIndex(indices) => indices.binary_search(&row).is_ok(),
                Selection::ExcludeByIndex(indices) => indices.binary_search(&row).is_err(),
                Selection::IncludeRoaring(indices) => indices.contains(row),
                Selection::ExcludeRoaring(indices) => !indices.contains(row),
            };
            if row < row_count && keep {
                rows.push(row);
            }
        }
        Ok(Selection::IncludeByIndex(StrictSortedBuffer::try_new(
            Buffer::from(rows),
        )?))
    }
}

pub(crate) fn replace_row_id(
    array: StructArray,
    field: usize,
    file: u64,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    let rows = array
        .unmasked_field(field)
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let prefix = encode(file, 0)?;
    let values = Buffer::from_iter(
        rows.as_slice::<u64>()
            .iter()
            .map(|&row| prefix | i128::from(row)),
    );
    // Decimal's I128 exporter shares HUGEINT's physical representation. This column is only
    // an in-memory carrier; DuckDB's declared virtual-column type is HUGEINT.
    let packed =
        DecimalArray::new(values, DecimalDType::new(38, 0), Validity::NonNullable).into_array();
    StructArray::try_new(
        array.names().clone(),
        array
            .iter_unmasked_fields()
            .enumerate()
            .map(|(index, value)| {
                if index == field {
                    packed.clone()
                } else {
                    value.clone()
                }
            }),
        array.len(),
        array.struct_validity(),
    )
}

#[cfg(test)]
mod tests;
