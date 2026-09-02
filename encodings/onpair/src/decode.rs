// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Helpers for turning [`OnPair`] slot children into the inputs the upstream
//! `onpair` decoder consumes.

use onpair::CompactDictionaryView;
use onpair::Offset;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
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

/// A validated, materialised window over an array's `codes`: the per-row
/// `codes_offsets` boundaries normalized to `u32` or `u64`, plus the codes they
/// bound.
///
/// `slice` keeps the full `codes` child and only narrows `codes_offsets`, so
/// for a sliced array the stored window starts at `offsets[0] > 0`. The
/// materialized offsets are rebased to the sliced codes, making this directly
/// usable as an upstream [`onpair::ColumnView`] as well as for per-row access.
/// Built once per compressed-domain query.
///
/// [`row`]: CodesWindow::row
pub(crate) enum CodesWindow {
    U32(TypedCodesWindow<u32>),
    U64(TypedCodesWindow<u64>),
}

pub(crate) struct TypedCodesWindow<O> {
    offsets: Buffer<O>,
    codes: Buffer<u16>,
}

impl<O: Offset> TypedCodesWindow<O> {
    /// The codes for row `i`.
    pub(crate) fn row(&self, i: usize) -> &[u16] {
        &self.codes[self.local(i)..self.local(i + 1)]
    }

    /// Borrow this window in the form expected by OnPair's column search API.
    pub(crate) fn as_column_view<'a>(
        &'a self,
        dict: CompactDictionaryView<'a>,
    ) -> onpair::ColumnView<'a, O> {
        onpair::ColumnView {
            dict,
            codes: self.codes.as_slice(),
            row_offsets: self.offsets.as_slice(),
        }
    }

    /// Offset `i` into the window's local `codes` slice.
    fn local(&self, i: usize) -> usize {
        self.offsets[i].to_usize()
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
    let offsets_ptype = array.codes_offsets().dtype().as_ptype();
    match offsets_ptype {
        PType::I8 | PType::I16 | PType::I32 | PType::U8 | PType::U16 | PType::U32 => {
            collect_typed_codes_window::<u32>(array, ctx).map(CodesWindow::U32)
        }
        PType::I64 | PType::U64 => {
            collect_typed_codes_window::<u64>(array, ctx).map(CodesWindow::U64)
        }
        ptype => Err(vortex_err!(
            "OnPair codes_offsets must be integer, found {ptype}"
        )),
    }
}

fn collect_typed_codes_window<O>(
    array: ArrayView<'_, OnPair>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TypedCodesWindow<O>>
where
    O: NativePType + Offset + Ord,
{
    let len = array.len();
    let offsets = collect_widened::<O>(array.codes_offsets(), ctx)?;
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
    let code_start = offsets[0].to_usize();
    let code_end = offsets[len].to_usize();
    vortex_ensure!(
        code_end <= array.codes().len(),
        "OnPair codes_offsets end {} exceeds codes len {}",
        code_end,
        array.codes().len()
    );
    let codes = collect_widened::<u16>(&array.codes().slice(code_start..code_end)?, ctx)?;
    let offsets = if code_start == 0 {
        offsets
    } else {
        offsets
            .map_each_in_place(|offset| <O as Offset>::from_usize(offset.to_usize() - code_start))
            .freeze()
    };
    Ok(TypedCodesWindow { offsets, codes })
}
