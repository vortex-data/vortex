// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Bridge between [`OnPair`] slot children and the upstream `onpair` crate's
//! decompression API. We materialise the dictionary blob and the three
//! integer children into native-aligned host buffers once, then hand the
//! result to [`onpair::decompress_into`] / [`onpair::decompress_row_into`].
//! The hot decode loop lives in the `onpair` crate.

use num_traits::AsPrimitive;
use onpair::Parts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArraySlotsExt;

/// Materialised, host-resident copies of every read path's input.
///
/// Each integer child (`dict_offsets`, `codes`, `codes_offsets`) is a slot on
/// the outer `OnPair` array, possibly wrapped in a non-canonical encoding the
/// cascading compressor chose (e.g. FastLanes-bit-packed `codes`, narrowed
/// dict offsets). `collect` runs `execute::<PrimitiveArray>` once per child
/// and widens each to the decoder's native width (`u32` for both offset
/// arrays, `u16` for codes) so [`Self::as_parts`] can hand a borrowed
/// [`Parts`] view to the upstream decoder.
pub struct OwnedDecodeInputs {
    pub dict_bytes: ByteBuffer,
    pub dict_offsets: Buffer<u32>,
    pub codes: Buffer<u16>,
    pub code_boundaries: Buffer<u32>,
    pub bits: u32,
}

impl OwnedDecodeInputs {
    pub fn collect(array: ArrayView<'_, OnPair>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            dict_bytes: array.dict_bytes().clone(),
            dict_offsets: widen_to::<u32>(&to_primitive(array.dict_offsets(), ctx)?),
            codes: widen_to::<u16>(&to_primitive(array.codes(), ctx)?),
            code_boundaries: widen_to::<u32>(&to_primitive(array.codes_offsets(), ctx)?),
            bits: array.bits(),
        })
    }

    /// Borrowed [`Parts`] view consumed by `onpair::decompress*`.
    pub fn as_parts(&self) -> Parts<'_, u32> {
        Parts {
            dict_bytes: self.dict_bytes.as_slice(),
            dict_offsets: self.dict_offsets.as_slice(),
            bits: self.bits,
            codes: self.codes.as_slice(),
            code_boundaries: self.code_boundaries.as_slice(),
        }
    }
}

fn to_primitive(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    arr.clone().execute::<PrimitiveArray>(ctx)
}

/// Widen any integer-typed `PrimitiveArray` to `Buffer<T>`. If the underlying
/// ptype already matches `T` we share the existing buffer (an Arc refcount
/// bump, no copy); otherwise we dispatch on ptype and run an element-wise
/// `AsPrimitive::as_()` cast via [`widen`].
fn widen_to<T>(arr: &PrimitiveArray) -> Buffer<T>
where
    T: NativePType,
    u8: AsPrimitive<T>,
    i8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    i16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    i64: AsPrimitive<T>,
{
    if arr.ptype() == T::PTYPE {
        return arr.clone().into_buffer::<T>();
    }
    match_each_integer_ptype!(arr.ptype(), |P| { widen::<P, T>(arr.as_slice::<P>()) })
}

/// Element-wise widen from `&[P]` to `Buffer<T>` via [`AsPrimitive`].
/// Method-call casts side-step the `clippy::cast_*` lints that `as` triggers
/// on each ptype arm of `match_each_integer_ptype!`.
fn widen<P, T>(slice: &[P]) -> Buffer<T>
where
    P: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    let mut out = BufferMut::<T>::with_capacity(slice.len());
    for &v in slice {
        // SAFETY: capacity reserved above.
        unsafe { out.push_unchecked(v.as_()) };
    }
    out.freeze()
}
