// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helpers for the Arrow canonical tensor extension plugins.
//!
//! Arrow tensor metadata describes the *physical* (row-major) layout, while Vortex tensor
//! metadata describes the *logical* layout. Both use the same permutation convention: logical
//! dimension `i` is physical dimension `permutation[i]`.

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::make_array;
use arrow_buffer::NullBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

/// Reorders per-dimension `logical` values into physical order.
///
/// `permutation` must be valid for `logical`, which Vortex tensor metadata guarantees.
pub(crate) fn to_physical<T: Clone>(logical: &[T], permutation: Option<&[usize]>) -> Vec<T> {
    let mut physical = logical.to_vec();
    if let Some(permutation) = permutation {
        for (i, &p) in permutation.iter().enumerate() {
            physical[p] = logical[i].clone();
        }
    }
    physical
}

/// Reorders per-dimension `physical` values into logical order.
///
/// Example: physical `[100, 200, 500]` with permutation `[2, 0, 1]` is logical `[500, 100, 200]`.
pub(crate) fn to_logical<T: Clone>(
    physical: &[T],
    permutation: Option<&[usize]>,
) -> VortexResult<Vec<T>> {
    let Some(permutation) = permutation else {
        return Ok(physical.to_vec());
    };

    vortex_ensure_eq!(
        permutation.len(),
        physical.len(),
        "permutation length ({}) must match dimension count ({})",
        permutation.len(),
        physical.len()
    );

    permutation
        .iter()
        .map(|&p| {
            physical
                .get(p)
                .cloned()
                .ok_or_else(|| vortex_err!("permutation index {p} is out of range"))
        })
        .collect()
}

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
