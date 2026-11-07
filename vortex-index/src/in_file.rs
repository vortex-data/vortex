// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Indexes stored inside a data file, as children of a `vortex.indexed` layout.
//!
//! Filters need nothing from this module: a scan of the file consults the index on its own. This
//! module opens the index content directly, which ranked search needs because a ranking is not a
//! filter the scan can push down.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_file::VortexFile;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutRef;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::Indexed;

use crate::rows::IndexRows;

/// Open the index of kind `vtable` built over top-level `column` of `file`, if it has one.
///
/// Only an unpartitioned index can be opened this way, since its rows are local to the whole
/// column.
pub fn open_in_file_index<V: IndexVTable>(
    vtable: V,
    file: &VortexFile,
    column: &str,
) -> VortexResult<Option<IndexRows<V>>> {
    let Some(indexed) = find_indexed(file.footer().layout(), column, false)? else {
        return Ok(None);
    };
    let layout = indexed.as_::<Indexed>();
    let id = vtable.id();
    let Some((position, spec)) = layout
        .indexes()
        .iter()
        .enumerate()
        .find(|(_, spec)| spec.id() == id && spec.partitioning().is_none())
    else {
        return Ok(None);
    };

    let options = vtable.deserialize_options(spec.options())?;
    let source_dtype = indexed.dtype().clone();
    if vtable.index_dtype(&source_dtype, &options).as_ref() != Some(spec.index_dtype()) {
        // The kind's schema changed since the file was written, so its content is not readable.
        return Ok(None);
    }
    let index_child = indexed.children()?.swap_remove(position + 1);
    let reader = index_child.new_reader(
        format!("{column}.index:{id}").into(),
        file.segment_source(),
        file.session(),
        &LayoutReaderContext::new(),
    )?;
    Ok(Some(IndexRows {
        vtable: Arc::new(vtable),
        options,
        source_dtype,
        index_dtype: spec.index_dtype().clone(),
        data_row_count: indexed.row_count(),
        reader,
        session: file.session().clone(),
    }))
}

/// The first `vortex.indexed` layout under top-level field `column`.
fn find_indexed(
    layout: &LayoutRef,
    column: &str,
    in_column: bool,
) -> VortexResult<Option<LayoutRef>> {
    if in_column && layout.is::<Indexed>() {
        return Ok(Some(Arc::clone(layout)));
    }
    for (child, child_type) in layout.children()?.iter().zip(layout.child_types()) {
        let in_child = match &child_type {
            LayoutChildType::Field(name) if !in_column => name.as_ref() == column,
            _ => in_column,
        };
        if !in_column && !in_child && matches!(child_type, LayoutChildType::Field(_)) {
            continue;
        }
        if let Some(found) = find_indexed(child, column, in_child)? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}
