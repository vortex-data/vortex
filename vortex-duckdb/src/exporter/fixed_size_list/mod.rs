// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Export support for Vortex FixedSizeListArray to DuckDB ARRAY type.
//!
//! DuckDB distinguishes between LIST (variable-size) and ARRAY (fixed-size) types.
//! The ARRAY type in DuckDB corresponds to Vortex's [`DType::FixedSizeList`], where all
//! lists have the same number of elements.
//!
//! [`DType::FixedSizeList`]: vortex::dtype::DType::FixedSizeList
use vortex::array::ExecutionCtx;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::error::VortexResult;

use super::ConversionCache;
use super::all_invalid;
use super::new_array_exporter_with_flatten;
use super::validity;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

#[cfg(test)]
mod tests;

/// Exporter for converting Vortex [`FixedSizeListArray`] to DuckDB ARRAY vectors.
struct FixedSizeListExporter {
    /// Exporter for the underlying elements array.
    elements_exporter: Box<dyn ColumnExporter>,
    /// The fixed number of elements in each list.
    list_size: u32,
    len: usize,
}

/// Creates a new exporter for converting a [`FixedSizeListArray`] to DuckDB ARRAY format.
pub(crate) fn new_exporter(
    array: FixedSizeListArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let list_size = array.list_size();
    let len = array.len();
    let parts = array.into_data_parts();
    let elements = parts.elements;
    let validity = parts.validity;

    if validity.definitely_all_null() {
        return Ok(all_invalid::new_exporter());
    }

    let elements_exporter = new_array_exporter_with_flatten(elements, cache, ctx, true)?;
    let exporter = Box::new(FixedSizeListExporter {
        elements_exporter,
        list_size,
        len,
    });
    validity::new_exporter(validity, len, exporter, ctx)
}

impl ColumnExporter for FixedSizeListExporter {
    // TODO(connor): Should `export` be `unsafe` instead? We have no way to verify this without
    // making an assertion.
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Verify that offset + len doesn't exceed the validity mask length.
        assert!(
            offset + len <= self.len,
            "Export range [{}, {}) exceeds array length {}",
            offset,
            offset + len,
            self.len
        );

        let list_size = self.list_size as usize;

        // Get the child vector for array elements and export the elements directly.
        let elements_vector = vector.array_vector_get_child_mut();
        let num_elements = len * list_size;
        self.elements_exporter
            .export(offset * list_size, num_elements, elements_vector, ctx)?;

        // We must flatten the child vector to ensure any child dictionary views (namely UTF-8
        // string views in dictionaries) are materialized, and pass the explicit element count so
        // constant vectors (whose C API default size is STANDARD_VECTOR_SIZE) flatten across all
        // `len * list_size` elements instead of truncating at 2048.
        // See https://github.com/vortex-data/vortex/pull/4610#issuecomment-3286676825 for a
        // detailed explanation on why we need this for now.
        //
        // SAFETY: `elements_vector` is the child of a DuckDB ARRAY vector with capacity for at
        // least `len * list_size` (`num_elements`) entries, and `elements_exporter` populated
        // `num_elements` entries (so flat children have capacity >= `num_elements`, dictionary and
        // sequence children already have size `num_elements`, and constant vectors/children can be
        // resized to `num_elements`).
        // `ListExporter` does not need this because it slices rather than flattening its child.
        unsafe { elements_vector.flatten_with_count(num_elements) };

        Ok(())
    }
}
