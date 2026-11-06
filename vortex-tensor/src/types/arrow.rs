// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helpers for the Arrow canonical tensor extension plugins.

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::make_array;
use arrow_buffer::NullBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// Removes the nulls of `child` that sit under null `parent` rows.
///
/// Vortex tensor children are non-nullable, but Arrow producers (e.g. pyarrow) write nulls under
/// null parent rows. Those slots are undefined, so the nulls are dropped. A null under a valid
/// parent row is an error.
///
/// `width` is the number of child slots per parent row: 1 for struct children, the list size for
/// fixed-size list values.
pub(crate) fn drop_masked_nulls(
    child: ArrowArrayRef,
    parent: Option<&NullBuffer>,
    width: usize,
) -> VortexResult<ArrowArrayRef> {
    let Some(child_nulls) = child.nulls().filter(|nulls| nulls.null_count() > 0) else {
        return Ok(child);
    };

    for i in (0..child.len()).filter(|&i| child_nulls.is_null(i)) {
        vortex_ensure!(
            parent.is_some_and(|parent| parent.is_null(i / width)),
            "tensor child value at index {i} is null under a valid row"
        );
    }

    let data = child.to_data().into_builder().nulls(None).build()?;
    Ok(make_array(data))
}
