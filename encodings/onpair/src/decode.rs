// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Helpers for turning [`OnPair`] slot children into the inputs the upstream
//! `onpair` decoder consumes.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;

/// Canonicalise a slot child to the decoder's native primitive width.
pub(crate) fn collect_widened<T: NativePType>(
    arr: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    collect_widened_range(arr, 0..arr.len(), ctx)
}

/// Canonicalise `range` of a slot child to the decoder's native primitive width.
///
/// A child that `OnPair::execute` has already required to be primitive is widened in place,
/// without a cast array or a nested execution; any other child is cast and executed.
pub(crate) fn collect_widened_range<T: NativePType>(
    arr: &ArrayRef,
    range: Range<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    if let Some(primitive) = arr.as_opt::<Primitive>()
        && !primitive.dtype().is_nullable()
    {
        if primitive.ptype() == T::PTYPE {
            return Ok(primitive.into_owned().into_buffer::<T>().slice(range));
        }
        return match_each_integer_ptype!(primitive.ptype(), |P| {
            primitive.as_slice::<P>()[range]
                .iter()
                .map(|&value| {
                    T::from(value).ok_or_else(|| {
                        vortex_err!("OnPair value {value} does not fit {}", T::PTYPE)
                    })
                })
                .collect::<VortexResult<Vec<T>>>()
                .map(Buffer::from)
        });
    }
    let dtype = DType::Primitive(T::PTYPE, arr.dtype().nullability());
    Ok(arr
        .slice(range)?
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
        .as_opt::<usize>()
        .flatten()
        .ok_or_else(|| vortex_err!("OnPair codes_offsets[{index}] is null, negative, or too large"))
}

/// A validated, materialised window over an array's `codes`: the codes of its
/// rows plus the per-row boundaries in them.
///
/// `slice` keeps the full `codes` child and only narrows `codes_offsets`, so
/// for a sliced array the stored boundaries start at `offsets[0] > 0`. The
/// window holds only the rows' codes, with the boundaries shifted to start at
/// zero. Built once per query by the compressed-domain compare and LIKE
/// kernels.
pub(crate) struct CodesWindow {
    offsets: Buffer<u64>,
    codes: Buffer<u16>,
}

impl CodesWindow {
    /// The codes for row `i`.
    pub(crate) fn row(&self, i: usize) -> &[u16] {
        &self.codes[self.offset(i)..self.offset(i + 1)]
    }

    /// The codes of every row.
    pub(crate) fn codes(&self) -> &[u16] {
        &self.codes
    }

    /// The row boundaries in [`codes`](Self::codes), one more than the rows.
    pub(crate) fn row_offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// Offsets are bounded by `codes.len()` (a `usize`), so the conversion
    /// never truncates.
    fn offset(&self, i: usize) -> usize {
        usize::try_from(self.offsets[i]).vortex_expect("code offset fits usize")
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
    vortex_ensure_eq!(
        offsets.len(),
        len + 1,
        "OnPair codes_offsets must have len + 1 entries"
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
    let codes = collect_widened_range::<u16>(array.codes(), code_start..code_end, ctx)?;
    let offsets = if code_start == 0 {
        offsets
    } else {
        let start = offsets[0];
        let mut offsets = offsets.into_mut();
        offsets.iter_mut().for_each(|offset| *offset -= start);
        offsets.freeze()
    };
    Ok(CodesWindow { offsets, codes })
}
