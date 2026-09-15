// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::ops::Range;

use num_traits::AsPrimitive as _;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::col;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::select;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::scalar::Scalar;
use vortex::scan::selection::Selection;
use vortex_utils::aliases::hash_set::HashSet;

use crate::convert::try_from_table_filter;
use crate::convert::try_from_virtual_column_filter;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFilterClass;
use crate::duckdb::TableFilterSetRef;
use crate::table_function::ColumnAggregate;

// See MultiFileReader for constants

/// "file_row_number" virtual column
pub(crate) static FILE_ROW_NUMBER_COLUMN_IDX: u64 = 9223372036854775809;

/// See duckdb/src/common/constants.cpp
pub(crate) fn is_virtual_column(id: u64) -> bool {
    id >= 9223372036854775808u64
}

#[derive(Debug, Clone)]
pub struct DuckdbField {
    pub name: String,
    pub logical_type: LogicalType,
    pub dtype: DType,
    /// Expression to use instead of get_item(col, root()), e.g. len(col).
    /// It does not include column name so it's just "len" and not "len(col)"
    pub projection_expr: Option<Expression>,
}

pub struct Projection {
    pub projection: Expression,
    pub file_row_number_column_pos: Option<usize>,
}

impl Projection {
    pub fn new(projection_ids: &[u64], column_ids: &[u64], column_fields: &[DuckdbField]) -> Self {
        let projection_ids: HashSet<u64> = projection_ids.iter().copied().collect();
        // If projection ids are empty, use column_ids.
        // See duckdb/src/planner/operator/logical_get.cpp#L168
        let is_projected =
            |pos: usize| projection_ids.is_empty() || projection_ids.contains(&(pos as u64));

        let mut exprs = Vec::with_capacity(column_ids.len() + 1);
        let mut file_row_number_column_pos = None;
        let mut is_star = true;
        let mut real_column_count = 0;

        // DuckDB uses u64 as column indices but Rust uses usize
        for (column_pos, &column_id) in column_ids.iter().enumerate() {
            if column_id == FILE_ROW_NUMBER_COLUMN_IDX {
                is_star = false;
                if is_projected(column_pos) {
                    file_row_number_column_pos = Some(exprs.len());
                } else {
                    // filter-only column needs to be emitted only for output
                    // vector position match, it will never be read
                    let dtype = DType::Primitive(PType::U64, Nullability::Nullable);
                    exprs.push(("file_row_number", lit(Scalar::null(dtype))));
                }
                continue;
            }

            if is_virtual_column(column_id) {
                continue;
            }

            let field_idx: usize = column_id.as_();
            let column_field = &column_fields[field_idx];
            let name = column_field.name.as_str();

            if !is_projected(column_pos) {
                is_star = false;
                // filter-only column needs to be emitted only for output
                // vector position match, it will never be read
                exprs.push((name, lit(Scalar::null(column_field.dtype.as_nullable()))));
                continue;
            }

            // In SELECT * DuckDB requests all columns from 0 to column_fields in
            // increasing order. After removing virtual columns, compare column_id
            // with (0..column_fields.len()) range.
            is_star &= column_id == real_column_count;

            // Example: if we SELECT len(str), we can't use root() as we try to
            // pushdown scalar functions.
            let expr = match &column_field.projection_expr {
                None => get_item(name, root()),
                Some(func) => {
                    is_star = false;
                    func.clone()
                }
            };
            exprs.push((name, expr));
            real_column_count += 1;
        }
        // Duckdb can request less columns than there are in table i.e. [0, 1] with
        // 5 columns total.
        is_star &= real_column_count == column_fields.len() as u64;

        if is_star {
            return Projection {
                projection: root(),
                file_row_number_column_pos: None,
            };
        }
        if file_row_number_column_pos.is_some() {
            // row_idx will be moved to correct position in scan(), prepend here
            exprs.insert(0, ("file_row_number", row_idx()));
        }
        Self {
            projection: pack(exprs, false.into()),
            file_row_number_column_pos,
        }
    }

    // Create a projection for aggregate scan
    pub fn new_aggregate(aggregates: &[ColumnAggregate], fields: &[DuckdbField]) -> Self {
        let mut exprs = Vec::with_capacity(aggregates.len());
        let mut seen: HashSet<u64> = HashSet::with_capacity(aggregates.len());
        let mut has_columns_with_expr = false;
        for aggregate in aggregates {
            let ColumnAggregate::Real { projection_id, .. } = aggregate else {
                continue;
            };
            if !seen.insert(*projection_id) {
                continue;
            }
            let projection_id: usize = projection_id.as_();
            let field = &fields[projection_id];
            let expr = match &field.projection_expr {
                None => get_item(field.name.as_str(), root()),
                Some(func) => {
                    has_columns_with_expr = true;
                    func.clone()
                }
            };
            exprs.push((field.name.as_str(), expr));
        }
        let projection = if has_columns_with_expr {
            pack(exprs, false.into())
        } else {
            let names = exprs.into_iter().map(|(name, _)| name).collect::<Vec<_>>();
            select(names, root())
        };
        Projection {
            projection,
            file_row_number_column_pos: None,
        }
    }
}

pub struct Filter {
    pub filter: Option<BoundExpression>,
    pub row_selection: Selection,
    pub row_range: Option<Range<u64>>,
    pub has_non_optional_filter: bool,
}

fn push_filter_expr(filter_exprs: &mut Vec<Expression>, expr: &Expression) {
    if !filter_exprs.iter().any(|existing| existing == expr) {
        filter_exprs.push(expr.clone());
    }
}

impl Filter {
    /// Creates a table filter expression, row selection, and row range from the table filter set,
    /// column metadata, additional filter expressions, and the top-level DType.
    pub fn new(
        table_filter_set: Option<&TableFilterSetRef>,
        column_ids: &[u64],
        column_fields: &[DuckdbField],
        additional_filters: &[Expression],
        dtype: &DType,
    ) -> VortexResult<Self> {
        let mut has_non_optional_filter = false;

        let mut table_filter_exprs = Vec::new();
        if let Some(filter) = table_filter_set {
            for (idx, ex) in filter.into_iter().filter(|(idx, _)| {
                let idx_u: usize = idx.as_();
                !is_virtual_column(column_ids[idx_u])
            }) {
                has_non_optional_filter |= !matches!(ex.as_class(), TableFilterClass::Optional(_));

                let idx_u: usize = idx.as_();
                let col_idx: usize = column_ids[idx_u].as_();
                let name = &column_fields.get(col_idx).vortex_expect("exists").name;
                if let Some(expr) = try_from_table_filter(ex, &col(name.as_str()), dtype)? {
                    push_filter_expr(&mut table_filter_exprs, &expr);
                }
            }
        }

        for expr in additional_filters {
            push_filter_expr(&mut table_filter_exprs, expr);
        }

        let mut row_selection = Selection::All;
        let mut row_range = None;
        if let Some(filter) = table_filter_set {
            for (idx, expression) in filter.into_iter() {
                let idx: usize = idx.as_();
                if column_ids[idx] == FILE_ROW_NUMBER_COLUMN_IDX {
                    (row_selection, row_range) = try_from_virtual_column_filter(expression)?;
                }
            }
        };

        let filter = and_collect(table_filter_exprs)
            .map(|expr| expr.optimize_recursive(dtype)?.bind(dtype))
            .transpose()?;

        let out = Self {
            filter,
            row_selection,
            row_range,
            has_non_optional_filter,
        };
        Ok(out)
    }
}

pub fn extract_schema_from_dtype(dtype: &DType) -> VortexResult<Vec<DuckdbField>> {
    let struct_dtype = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("Vortex file must contain a struct array at the top level"))?;

    let len = struct_dtype.names().len();
    let mut fields = Vec::with_capacity(len);

    for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
        let logical_type = LogicalType::try_from(&field_dtype)?;
        fields.push(DuckdbField {
            name: field_name.to_string(),
            logical_type,
            dtype: field_dtype,
            projection_expr: None,
        });
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use vortex::dtype::DType;
    use vortex::expr::lit;
    use vortex::expr::pack;
    use vortex::expr::root;
    use vortex::layout::layouts::row_idx::row_idx;

    use super::*;

    fn field(name: &str) -> DuckdbField {
        DuckdbField {
            name: name.to_owned(),
            logical_type: LogicalType::null(),
            dtype: DType::Null,
            projection_expr: None,
        }
    }

    #[test]
    fn test_select_star() {
        let ids = [0, 1, 2];
        let mut fields = [field("a"), field("b"), field("c")];

        assert_eq!(Projection::new(&[], &ids, &fields).projection, root());

        // file_row_number turns star into an explicit pack with row_idx first
        let ids = [FILE_ROW_NUMBER_COLUMN_IDX, 0, 1, 2];
        let result = Projection::new(&[], &ids, &fields);
        let expected = pack(
            [
                ("file_row_number", row_idx()),
                ("a", get_item("a", root())),
                ("b", get_item("b", root())),
                ("c", get_item("c", root())),
            ],
            false.into(),
        );
        assert_eq!(result.projection, expected);
        assert_eq!(result.file_row_number_column_pos, Some(0));

        let ids = [0, 1];
        assert_ne!(Projection::new(&[], &ids, &fields).projection, root());

        let ids = [0, 2, 2];
        assert_ne!(Projection::new(&[], &ids, &fields).projection, root());

        let ids = [2, 1, 0];
        assert_ne!(Projection::new(&[], &ids, &fields).projection, root());

        // If any column has a projection expression, we can't use SELECT *
        fields[0].projection_expr = Some(lit(true));
        let ids = [0, 1, 2];
        assert_ne!(Projection::new(&[], &ids, &fields).projection, root());
    }

    #[test]
    fn test_projections() {
        let fields = [field("a"), field("b"), field("c")];

        let ids = [0, 1, 2];
        let projection = Projection::new(&[0, 2], &ids, &fields).projection;
        let expected = pack(
            [
                ("a", get_item("a", root())),
                ("b", lit(Scalar::null(DType::Null))),
                ("c", get_item("c", root())),
            ],
            false.into(),
        );
        assert_eq!(projection, expected);

        let ids = [FILE_ROW_NUMBER_COLUMN_IDX, 0];
        let result = Projection::new(&[1], &ids, &fields);
        let frn_dtype = DType::Primitive(PType::U64, Nullability::Nullable);
        let expected = pack(
            [
                ("file_row_number", lit(Scalar::null(frn_dtype))),
                ("a", get_item("a", root())),
            ],
            false.into(),
        );
        assert_eq!(result.projection, expected);
        assert_eq!(result.file_row_number_column_pos, None);

        let ids = [0, FILE_ROW_NUMBER_COLUMN_IDX, 1];
        let result = Projection::new(&[0, 1], &ids, &fields);
        let expected = pack(
            [
                ("file_row_number", row_idx()),
                ("a", get_item("a", root())),
                ("b", lit(Scalar::null(DType::Null))),
            ],
            false.into(),
        );
        assert_eq!(result.projection, expected);
        assert_eq!(result.file_row_number_column_pos, Some(1));
    }

    #[test]
    fn test_push_filter_expr_preserves_order() {
        let first = col("first");
        let second = col("second");

        let mut filter_exprs = Vec::new();
        push_filter_expr(&mut filter_exprs, &first);
        push_filter_expr(&mut filter_exprs, &second);
        push_filter_expr(&mut filter_exprs, &first);

        assert_eq!(filter_exprs, vec![first, second]);
    }
}
