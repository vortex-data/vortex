// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Helpers for turning [`OnPair`] slot children into the inputs the upstream
//! `onpair` decoder consumes.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

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

/// Read one `codes_offsets` boundary by point lookup. This decodes at most a
/// single chunk of the child — never the whole per-row offsets array — so the
/// callers that only need a row window (`scalar_at`, the canonical decode's
/// start/end bounds) don't pay to materialise every boundary.
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
