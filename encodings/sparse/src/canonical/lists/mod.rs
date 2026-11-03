// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode sparse lists with null or empty fills while keeping their compact elements shared.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::sparse_validity;
use crate::Sparse;

/// Reconstructs row metadata around the compact patch lists.
///
/// The caller has already made `values` contiguous and trimmed unused elements. Fill rows
/// contribute no elements, so only offsets, sizes, and validity need to cover the full row domain.
pub(super) fn empty_fill(
    patches: &Patches,
    values: ListViewArray,
    fill: &Scalar,
    len: usize,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let positions = patches
        .indices()
        .cast(PType::U64.into())?
        .execute::<PrimitiveArray>(ctx)?;
    let positions = positions.as_slice::<u64>();

    // A native List needs a terminal offset in the same type as its starting offsets. A view
    // can instead represent an end that exceeds that type, e.g. u8 offset 130 + size 130 = 260.
    let elements_len = u64::try_from(values.elements().len())
        .map_err(|err| vortex_err!("sparse list elements length: {err}"))?;
    let terminal_offset_fits = Scalar::from(elements_len)
        .cast(values.offsets().dtype())
        .is_ok();

    // Each patch must advance to the next compact list exactly once. Duplicate positions need
    // last-patch-wins semantics instead, so they use the view-based fallback below.
    if terminal_offset_fits && positions.windows(2).all(|pair| pair[0] < pair[1]) {
        let list = list_from_list_view(values, ctx)?;

        // Offsets change only after a patch row: a patch at p ends the preceding offset run
        // at p + 1. The final run extends through the terminal offset at index len.
        //
        // Patch positions [0, 3], compact offsets [0, 3, 6], output length 5:
        //   run ends:        [1, 4, 6]
        //   offset values:   [0, 3, 6]
        //   decoded offsets: [0, 3, 3, 3, 6, 6]
        let end = len
            .checked_add(1)
            .ok_or_else(|| vortex_err!("sparse list offsets length overflow"))?;
        let end =
            u64::try_from(end).map_err(|err| vortex_err!("sparse list offsets length: {err}"))?;
        let ends: Buffer<u64> = positions
            .iter()
            .map(|position| position + 1)
            .chain(std::iter::once(end))
            .collect();

        // Keep offsets as runs so consumers such as Arrow export can decode them in bulk,
        // without rebuilding list elements or expanding offsets to one entry per output row here.
        let offsets = RunEnd::try_new(
            PrimitiveArray::new(ends, Validity::NonNullable).into_array(),
            list.offsets().clone(),
            ctx,
        )?
        .into_array();
        let validity = sparse_validity(patches, fill, nullability, len, ctx)?;

        // SAFETY: compact list offsets are sorted and bounded by the shared elements. The
        // run ends produce exactly len + 1 offsets and the validity has len entries.
        return Ok(
            unsafe { ListArray::new_unchecked(list.elements().clone(), offsets, validity) }
                .into_array(),
        );
    }

    // Scatter starting offsets and sizes independently. Fill rows get an empty view (0, 0),
    // and Sparse's existing patch handling makes the last patch win at repeated positions.
    // Keeping these as separate arrays also avoids overflowing a narrow terminal offset.
    let offsets = values.offsets();
    let offsets = Sparse::try_new_from_patches(
        patches.clone().map_values(|_| Ok(offsets.clone()))?,
        Scalar::zero_value(offsets.dtype()),
    )?
    .into_array();

    let sizes = values.sizes();
    let sizes = Sparse::try_new_from_patches(
        patches.clone().map_values(|_| Ok(sizes.clone()))?,
        Scalar::zero_value(sizes.dtype()),
    )?
    .into_array();

    let validity = sparse_validity(patches, fill, nullability, len, ctx)?;

    // Keep the compact elements intact: per-patch slices of a lazy gather can repeatedly
    // materialize its index array. A null or empty fill needs no additional elements.
    //
    // SAFETY: patch views retain their validated offsets/sizes; all other views use (0, 0).
    // Both metadata arrays and the patched validity have the requested output length.
    Ok(
        unsafe {
            ListViewArray::new_unchecked(values.elements().clone(), offsets, sizes, validity)
        }
        .into_array(),
    )
}

#[cfg(test)]
mod tests;
