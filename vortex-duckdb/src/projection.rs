// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::ops::Range;

use num_traits::AsPrimitive as _;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::col;
use vortex::expr::get_item;
use vortex::expr::merge;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::select;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::scan::selection::Selection;
use vortex_utils::aliases::hash_set::HashSet;

use crate::convert::try_from_table_filter;
use crate::convert::try_from_virtual_column_filter;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFilterClass;
use crate::duckdb::TableFilterSetRef;

// See MultiFileReader for constants

/// "file_index" virtual column
static FILE_INDEX_COLUMN_IDX: u64 = 9223372036854775810;
/// "file_row_number" virtual column
static FILE_ROW_NUMBER_COLUMN_IDX: u64 = 9223372036854775809;

/// See duckdb/src/common/constants.cpp
fn is_virtual_column(id: u64) -> bool {
    id >= 9223372036854775808u64
}

#[derive(Debug, Clone)]
pub struct DuckdbField {
    pub name: String,
    pub logical_type: LogicalType,
    pub dtype: DType,
    /// Function to use instead of get_item(col, root()), e.g. len(col).
    /// It does not include column name so it's just "len" and not "len(col)"
    pub projection_fn: Option<Expression>,
}

pub struct Projection {
    pub projection: Expression,
    pub file_index_column_pos: Option<usize>,
    pub file_row_number_column_pos: Option<usize>,
}

impl Projection {
    pub fn new(
        projection_ids: Option<&[u64]>,
        column_ids: &[u64],
        column_fields: &[DuckdbField],
    ) -> Self {
        // If projection ids are empty, use column_ids.
        // See duckdb/src/planner/operator/logical_get.cpp#L168
        let (ids, has_projection_ids) = match projection_ids {
            Some(ids) => (ids, true),
            None => (column_ids, false),
        };

        let mut file_index_column_pos = None;
        let mut file_row_number_column_pos = None;
        let mut is_star = true;
        let mut real_column_count = 0;
        let mut fn_col_count = 0;

        // DuckDB uses u64 as column indices but Rust uses usize
        for (column_pos, &column_id) in ids.iter().enumerate() {
            let column_id = if has_projection_ids {
                let column_id: usize = column_id.as_();
                column_ids[column_id]
            } else {
                column_id
            };

            if column_id == FILE_INDEX_COLUMN_IDX {
                file_index_column_pos = Some(column_pos);
                continue;
            }
            if column_id == FILE_ROW_NUMBER_COLUMN_IDX {
                file_row_number_column_pos = Some(column_pos);
                continue;
            }
            if is_virtual_column(column_id) {
                continue;
            }

            // In SELECT * DuckDB requests all columns from 0 to column_fields in
            // increasing order. After removing virtual columns, compare column_id
            // with (0..column_fields.len()) range.
            is_star &= column_id == real_column_count;

            // Example: if we SELECT len(str), we can't use root() as we try to
            // pushdown scalar functions.
            let column_id: usize = column_id.as_();
            let is_projected_col = column_fields[column_id].projection_fn.is_some();
            fn_col_count += is_projected_col as usize;
            is_star &= !is_projected_col;

            real_column_count += 1;
        }
        // Duckdb can request less columns than there are in table i.e. [0, 1] with
        // 5 columns total.
        is_star &= real_column_count == column_fields.len() as u64;

        let has_file_row_number = file_row_number_column_pos.is_some();
        if is_star {
            let projection = if has_file_row_number {
                // row_idx will be moved to correct position in scan(), prepend here
                let row_idx_struct = pack([("file_row_number", row_idx())], false.into());
                merge([row_idx_struct, root()])
            } else {
                root()
            };
            return Projection {
                projection,
                file_index_column_pos,
                file_row_number_column_pos,
            };
        }

        let has_fn_columns = fn_col_count > 0;
        let mut all_exprs = Vec::with_capacity(
            (ids.len() + has_file_row_number as usize) * has_fn_columns as usize,
        );
        let mut named_fields = Vec::with_capacity(ids.len() * !has_fn_columns as usize);

        if has_file_row_number && has_fn_columns {
            // row_idx will be moved to correct position in scan(), prepend here
            all_exprs.push(("file_row_number", row_idx()));
        }

        for &column_id in ids {
            let column_id = if has_projection_ids {
                let column_id: usize = column_id.as_();
                column_ids[column_id]
            } else {
                column_id
            };
            if is_virtual_column(column_id) {
                continue;
            }
            let column_id: usize = column_id.as_();
            let name = column_fields[column_id].name.as_str();
            if !has_fn_columns {
                named_fields.push(name);
                continue;
            }

            let column_field = &column_fields[column_id];
            let expr = match &column_field.projection_fn {
                None => get_item(name, root()),
                Some(func) => func.clone(),
            };
            all_exprs.push((name, expr));
        }

        let projection = if has_fn_columns {
            // If file_row_number was requested, it's in all_exprs as first
            // element
            pack(all_exprs, false.into())
        } else if has_file_row_number {
            let select = select(named_fields, root());
            // Here we need to prepend it manually
            // row_idx will be moved to correct position in scan()
            let row_idx_struct = pack([("file_row_number", row_idx())], false.into());
            merge([row_idx_struct, select])
        } else {
            select(named_fields, root())
        };

        Self {
            projection,
            file_index_column_pos,
            file_row_number_column_pos,
        }
    }
}

pub struct Filter {
    pub filter: Option<Expression>,
    pub row_selection: Selection,
    pub row_range: Option<Range<u64>>,
    pub file_selection: Selection,
    pub file_range: Option<Range<u64>>,
    pub has_non_optional_filter: bool,
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

        let mut table_filter_exprs: HashSet<Expression> = if let Some(filter) = table_filter_set {
            filter
                .into_iter()
                .filter(|(idx, _)| {
                    let idx_u: usize = idx.as_();
                    !is_virtual_column(column_ids[idx_u])
                })
                .map(|(idx, ex)| {
                    has_non_optional_filter |=
                        !matches!(ex.as_class(), TableFilterClass::Optional(_));

                    let idx_u: usize = idx.as_();
                    let col_idx: usize = column_ids[idx_u].as_();
                    let name = &column_fields.get(col_idx).vortex_expect("exists").name;
                    try_from_table_filter(ex, &col(name.as_str()), dtype)
                })
                .collect::<VortexResult<Option<HashSet<_>>>>()?
                .unwrap_or_else(HashSet::new)
        } else {
            HashSet::new()
        };

        table_filter_exprs.extend(additional_filters.iter().cloned());

        let mut file_selection = Selection::All;
        let mut row_selection = Selection::All;
        let mut row_range = None;
        let mut file_range = None;
        if let Some(filter) = table_filter_set {
            for (idx, expression) in filter.into_iter() {
                let idx: usize = idx.as_();
                if column_ids[idx] == FILE_ROW_NUMBER_COLUMN_IDX {
                    (row_selection, row_range) = try_from_virtual_column_filter(expression)?;
                }
                if column_ids[idx] == FILE_INDEX_COLUMN_IDX {
                    (file_selection, file_range) = try_from_virtual_column_filter(expression)?;
                }
            }
        };

        let out = Self {
            filter: and_collect(table_filter_exprs),
            row_selection,
            row_range,
            file_selection,
            file_range,
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
            projection_fn: None,
        });
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use vortex::dtype::DType;
    use vortex::expr::lit;
    use vortex::expr::merge;
    use vortex::expr::pack;
    use vortex::expr::root;
    use vortex::layout::layouts::row_idx::row_idx;

    use super::*;

    #[test]
    fn test_select_star() {
        let ids = [0, 1, 2];
        let mut fields = [
            DuckdbField {
                name: "".to_owned(),
                logical_type: LogicalType::null(),
                dtype: DType::Null,
                projection_fn: None,
            },
            DuckdbField {
                name: "".to_owned(),
                logical_type: LogicalType::null(),
                dtype: DType::Null,
                projection_fn: None,
            },
            DuckdbField {
                name: "".to_owned(),
                logical_type: LogicalType::null(),
                dtype: DType::Null,
                projection_fn: None,
            },
        ];

        assert_eq!(Projection::new(None, &ids, &fields).projection, root());

        let ids = [FILE_ROW_NUMBER_COLUMN_IDX, 0, 1, FILE_INDEX_COLUMN_IDX, 2];
        let exprs = Projection::new(None, &ids, &fields);
        let row_idx_struct = pack([("file_row_number", row_idx())], false.into());
        let root_with_virtual_cols = merge([row_idx_struct, root()]);

        assert_eq!(exprs.projection, root_with_virtual_cols);
        assert_eq!(exprs.file_index_column_pos, Some(3));
        assert_eq!(exprs.file_row_number_column_pos, Some(0));

        // projections can't be set in SELECT *.
        assert_ne!(
            Projection::new(Some(&[0, 1]), &ids, &fields).projection,
            root()
        );

        let ids = [0, 1];
        assert_ne!(Projection::new(None, &ids, &fields).projection, root());

        let ids = [0, 2, 2];
        assert_ne!(Projection::new(None, &ids, &fields).projection, root());

        let ids = [2, 1, 0];
        assert_ne!(Projection::new(None, &ids, &fields).projection, root());

        // If any column has a projection expression, we can't use SELECT *
        fields[0].projection_fn = Some(lit(true));
        let ids = [0, 1, 2];
        assert_ne!(Projection::new(None, &ids, &fields).projection, root());
    }
}
