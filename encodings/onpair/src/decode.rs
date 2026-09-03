// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Helpers for turning [`OnPair`] slot children into the inputs the upstream
//! `onpair` decoder consumes.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;

/// Canonicalise a slot child to the decoder's native primitive width.
pub(crate) fn collect_widened<T: NativePType>(
    arr: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    let dtype = DType::Primitive(T::PTYPE, arr.dtype().nullability());
    Ok(arr
        .cast(dtype)?
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<T>())
}

pub(crate) fn code_boundary_at(
    codes_offsets: &ArrayRef,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    codes_offsets
        .execute_scalar(index, ctx)?
        .as_primitive()
        .as_::<usize>()
        .ok_or_else(|| vortex_err!("OnPair codes_offsets[{index}] is null"))
}

/// A validated, materialised window over an array's `codes`: the widened
/// per-row `codes_offsets` boundaries plus the codes they bound.
///
/// `slice` keeps the full `codes` child and only narrows `codes_offsets`, so
/// for a sliced array the window starts at `offsets[0] > 0`; [`row`] resolves
/// row indices relative to that start. Built once per query by the
/// compressed-domain compare kernel.
///
/// [`row`]: CodesWindow::row
pub(crate) struct CodesWindow {
    offsets: Buffer<u64>,
    codes: Buffer<u16>,
    code_start: usize,
}

impl CodesWindow {
    /// The codes for row `i`.
    pub(crate) fn row(&self, i: usize) -> &[u16] {
        &self.codes[self.local(i)..self.local(i + 1)]
    }

    /// Offset `i` rebased into the window's local `codes` slice. Offsets are
    /// bounded by `codes.len()` (a `usize`), so the conversion never truncates.
    fn local(&self, i: usize) -> usize {
        usize::try_from(self.offsets[i]).vortex_expect("code offset fits usize") - self.code_start
    }
}

/// Materialise the [`CodesWindow`] for every row of `array`: offsets must be
/// nondecreasing and end within the `codes` child. Codes themselves are
/// trusted to the upstream decoder/search primitives, which bounds-check them
/// in-loop and panic on a malformed value.
pub(crate) fn collect_codes_window(
    array: ArrayView<'_, OnPair>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<CodesWindow> {
    let len = array.len();
    let offsets = collect_widened::<u64>(array.codes_offsets(), ctx)?;
    vortex_ensure!(
        offsets.len() == len + 1,
        "OnPair codes_offsets has {} entries, expected len + 1 = {}",
        offsets.len(),
        len + 1
    );
    vortex_ensure!(
        offsets.is_sorted(),
        "OnPair codes_offsets must be nondecreasing"
    );
    let code_start = usize::try_from(offsets[0]).vortex_expect("code offset fits usize");
    let code_end = usize::try_from(offsets[len]).vortex_expect("code offset fits usize");
    vortex_ensure!(
        code_end <= array.codes().len(),
        "OnPair codes_offsets end {} exceeds codes len {}",
        code_end,
        array.codes().len()
    );
    let codes = collect_widened::<u16>(&array.codes().slice(code_start..code_end)?, ctx)?;
    Ok(CodesWindow {
        offsets,
        codes,
        code_start,
    })
}
