// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading the low-bits child in bulk, which is an ordinary FastLanes `BitPacked` array.
//!
//! Only the bulk decode comes through here; a per-element read goes through the child's own
//! `scalar_at` in [`crate::access`].

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;

/// The low-bits child as a `BitPacked` view, if its bytes can be read where they lie:
/// FastLanes-packed, unpatched, host-resident and `u64`-aligned.
///
/// `unpacked_chunks` reinterprets those bytes as `&[u64]` without a check of its own, so the four
/// conditions are stated here. `None` means the low bits must be materialised.
pub(crate) fn readable_in_place(lower: &ArrayRef) -> Option<ArrayView<'_, BitPacked>> {
    // `as_opt`, never `as_`: with the experimental patched-array plugin enabled the slot comes back
    // from a file as `Patched(BitPacked)`, and a rewrite may replace it outright.
    let packed = lower.as_opt::<BitPacked>()?;
    (packed.patches().is_none()
        && packed
            .packed()
            .as_host_opt()
            .is_some_and(|buffer| buffer.is_aligned(Alignment::of::<u64>())))
    .then_some(packed)
}

/// The low bits of an already-windowed child, materialised when its bytes cannot be read in place.
///
/// The child spans the whole encoded sequence, so a caller windows it to the range being folded
/// first.
pub(crate) fn materialise(windowed: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u64>> {
    Ok(windowed
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<u64>())
}
