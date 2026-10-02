// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod blocked;
mod global;

pub(crate) use blocked::unpack_single_blocked;
pub use global::count_exceptions;
pub use global::unpack_single;
pub use global::unpack_single_primitive;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::NativePType;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidths;
use crate::unpack_iter::BitPacked as BitPackedUnpack;

/// Unpacks a bit-packed array into a primitive array.
pub fn unpack_array(
    array: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match array.bit_widths() {
        BitWidths::Global(_) => global::unpack_array(array, ctx),
        BitWidths::Blocked(offsets) => blocked::unpack_array(array, offsets, ctx),
    }
}

pub fn unpack_primitive_array<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match array.bit_widths() {
        BitWidths::Global(_) => global::unpack_primitive_array::<T>(array, ctx),
        BitWidths::Blocked(offsets) => blocked::unpack_primitive_array::<T>(array, offsets, ctx),
    }
}

/// Unpack a bit-packed array directly into a same-typed `PrimitiveBuilder`.
pub(crate) fn unpack_into_primitive_builder<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPacked>,
    builder: &mut PrimitiveBuilder<T>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match array.bit_widths() {
        BitWidths::Global(_) => global::unpack_into_primitive_builder(array, builder, ctx),
        BitWidths::Blocked(offsets) => {
            blocked::unpack_into_primitive_builder(array, offsets, builder, ctx)
        }
    }
}

/// Unpack a bit-packed array of physical type `F` into a `PrimitiveBuilder<T>`, applying `map`
/// to each value during decompression.
///
/// The caller must ensure that every valid source value is representable in `T` under `map`; no
/// per-value bounds check is performed.
pub(crate) fn unpack_map_into_builder<F, T, M>(
    array: ArrayView<'_, BitPacked>,
    builder: &mut PrimitiveBuilder<T>,
    ctx: &mut ExecutionCtx,
    map: M,
) -> VortexResult<()>
where
    F: BitPackedUnpack,
    T: NativePType,
    M: Fn(F) -> T,
{
    match array.bit_widths() {
        BitWidths::Global(_) => global::unpack_map_into_builder(array, builder, ctx, map),
        BitWidths::Blocked(offsets) => {
            blocked::unpack_map_into_builder(array, offsets, builder, ctx, map)
        }
    }
}
