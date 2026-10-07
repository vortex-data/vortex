// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;

use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::convert::FromLogicalType;
use crate::cpp;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFilterSet;
use crate::duckdb::TableFilterSetRef;

/// A VARIANT field DuckDB asks the scan to extract and emit directly, instead of the column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantExtract {
    /// Object field names from the VARIANT column to the field.
    pub path: Vec<String>,
    /// The dtype to emit the field as.
    pub dtype: DType,
}

pub struct TableInitInput<'a> {
    pub input: &'a cpp::duckdb_vx_tfunc_init_input,
}

impl Debug for TableInitInput<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("TableInitInput")
            .field("column_ids", &self.column_ids())
            .field("projection_ids", &self.projection_ids())
            .field("table_filter_set", &self.table_filter_set())
            .finish()
    }
}

impl<'a> TableInitInput<'a> {
    pub fn new(input: &'a cpp::duckdb_vx_tfunc_init_input) -> Self {
        Self { input }
    }

    pub fn column_ids(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.input.column_ids, self.input.column_ids_count) }
    }

    /// For each of [`Self::column_ids`], the VARIANT field to extract from it, if any.
    pub fn extracts(&self) -> VortexResult<Vec<Option<VariantExtract>>> {
        let count = self.input.column_ids_count;
        if self.input.extracts.is_null() {
            return Ok(vec![None; count]);
        }
        let extracts = unsafe { std::slice::from_raw_parts(self.input.extracts, count) };
        extracts
            .iter()
            .map(|extract| {
                if extract.path.is_null() {
                    return Ok(None);
                }
                let path = unsafe { std::slice::from_raw_parts(extract.path, extract.path_len) }
                    .iter()
                    .map(|name| {
                        let name = unsafe { CStr::from_ptr(*name) };
                        Ok(name
                            .to_str()
                            .map_err(|err| vortex_err!("Variant field name is not UTF-8: {err}"))?
                            .to_owned())
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                let logical_type = unsafe { LogicalType::borrow(extract.type_) };
                let dtype = DType::from_logical_type(logical_type, Nullability::Nullable)?;
                Ok(Some(VariantExtract { path, dtype }))
            })
            .collect()
    }

    pub fn projection_ids(&self) -> &[u64] {
        if self.input.projection_ids_count == 0 {
            // from_raw_parts requires a non-null pointer. C++'s empty vector
            // may have a null pointer.
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(self.input.projection_ids, self.input.projection_ids_count)
        }
    }

    /// Returns the table filter set for the table function.
    pub fn table_filter_set(&self) -> Option<&TableFilterSetRef> {
        let ptr = self.input.filters;
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { TableFilterSet::borrow(ptr) })
        }
    }
}
