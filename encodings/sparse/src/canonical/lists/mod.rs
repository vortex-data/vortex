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

/// Expands sparse lists by constructing offsets over their existing elements.
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

    if !can_use_native_list(&values)? {
        return empty_fill_as_views(patches, values, fill, len, nullability, ctx);
    }

    let list = list_from_list_view(values, ctx)?;

    // Offsets change only after a patch row: a patch at p ends the preceding offset run
    // at p + 1. The final run extends through the terminal offset at index len.
    //
    // Rows: [a,b], null, null, [c,d], null. Shared elements: [a,b,c,d].
    //   run ends:        [1, 4, 6]
    //   offset values:   [0, 2, 4]
    //   decoded offsets: [0, 2, 2, 2, 4, 4]
    let end = len
        .checked_add(1)
        .ok_or_else(|| vortex_err!("sparse list offsets length overflow"))?;
    let end = u64::try_from(end).map_err(|err| vortex_err!("sparse list offsets length: {err}"))?;
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
    Ok(
        unsafe { ListArray::new_unchecked(list.elements().clone(), offsets, validity) }
            .into_array(),
    )
}

/// Checks whether the patch layout's terminal offset fits a native List's offsets.
fn can_use_native_list(values: &ListViewArray) -> VortexResult<bool> {
    // A native List needs a terminal offset in the same type as its starting offsets. A view
    // can instead represent an end that exceeds that type, e.g. u8 offset 130 + size 130 = 260.
    let elements_len = u64::try_from(values.elements().len())
        .map_err(|err| vortex_err!("sparse list elements length: {err}"))?;
    Ok(Scalar::from(elements_len)
        .cast(values.offsets().dtype())
        .is_ok())
}

/// Keeps shared elements and scatters view metadata for narrow offsets.
fn empty_fill_as_views(
    patches: &Patches,
    values: ListViewArray,
    fill: &Scalar,
    len: usize,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Scatter starting offsets and sizes independently. Fill rows get an empty view (0, 0).
    // Keeping these as separate arrays avoids overflowing a narrow terminal offset.
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
