// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::mem;

use fastlanes::BitPacking;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::WrappingMul;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::AffineArray;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::slope_term;
use crate::unpack_iter::for_each_packed_chunk;

/// One parameter per chunk: either shared by every chunk or read from a primitive child.
pub(crate) enum ChunkParam<T> {
    Constant(T),
    PerChunk(Buffer<T>),
}

impl<T: NativePType> ChunkParam<T> {
    /// Read a per-chunk parameter child, which must be constant or primitive.
    pub(crate) fn from_child(child: &ArrayRef) -> Self {
        match child.as_constant() {
            Some(scalar) => Self::Constant(
                scalar
                    .as_primitive()
                    .typed_value::<T>()
                    .vortex_expect("Affine parameters are non-nullable"),
            ),
            None => Self::PerChunk(child.as_::<Primitive>().into_owned().into_buffer::<T>()),
        }
    }

    #[inline(always)]
    pub(crate) fn get(&self, chunk: usize) -> T {
        match self {
            Self::Constant(value) => *value,
            Self::PerChunk(values) => values[chunk],
        }
    }

}

/// Decode an Affine array whose non-constant parameters are primitive, and whose `encoded` child
/// is primitive or bit-packed with the same offset.
pub fn decompress(array: &AffineArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    match array.encoded().as_opt::<BitPacked>() {
        Some(bp) if bp.offset() == array.offset() => {
            match_each_integer_ptype!(array.ptype(), |T| {
                fused_decompress_typed::<T>(array, bp, ctx)
            })
        }
        _ => match_each_integer_ptype!(array.ptype(), |T| {
            decompress_typed::<T>(array, ctx)
        }),
    }
}

impl<T: NativePType> Params<T> {
    fn new(array: &AffineArray) -> Self {
        Self {
            references: ChunkParam::from_child(array.references()),
            scales: ChunkParam::from_child(array.scales()),
            slopes: ChunkParam::from_child(array.slopes()),
            offset: usize::from(array.offset()),
            shift: array.slope_shift(),
        }
    }
}

fn decompress_typed<T>(
    array: &AffineArray,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
{
    let encoded = array.encoded().as_::<Primitive>().into_owned();
    if encoded.is_empty() {
        return Ok(encoded);
    }
    let validity = encoded.validity()?;
    let params = Params::<T>::new(array);
    let mut values = encoded.into_buffer::<T>().into_mut();
    // The first chunk may be partial when the array was sliced.
    let first_len = (FL_CHUNK_SIZE - params.offset).min(values.len());
    let (first, rest) = values.split_at_mut(first_len);
    let chunks = iter::once((params.offset, first))
        .chain(rest.chunks_mut(FL_CHUNK_SIZE).map(|c| (0, c)));
    for (chunk_idx, (start, chunk)) in chunks.enumerate() {
        params.apply_chunk(chunk_idx, start, chunk);
    }
    Ok(PrimitiveArray::new(values.freeze(), validity))
}

/// Unpack each bit-packed chunk into a scratch chunk and apply its model while it is in cache.
fn fused_decompress_typed<T>(
    array: &AffineArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: PhysicalPType<Physical: BitPacking> + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
{
    let len = bp.len();
    let params = Params::<T>::new(array);
    let offset = usize::from(bp.offset());
    let bit_width = bp.bit_width() as usize;
    let mut values = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
    let mut scratch = [T::Physical::default(); FL_CHUNK_SIZE];
    for_each_packed_chunk::<T, _>(
        bp.packed_slice::<T::Physical>(),
        bit_width,
        offset,
        len,
        |packed, range| {
            // SAFETY: `packed` holds one chunk at `bit_width` and `scratch` holds a whole chunk.
            unsafe { BitPacking::unchecked_unpack(bit_width, packed, &mut scratch) };
            // SAFETY: `T::Physical` is `T` with the same size and alignment.
            let unpacked =
                unsafe { mem::transmute::<&mut [T::Physical], &mut [T]>(&mut scratch[..]) };
            // `range` counts from the start of the first chunk, and the output starts at `offset`.
            let start = offset.saturating_sub(range.start);
            let chunk = &mut unpacked[start..range.len()];
            params.apply_chunk(range.start / FL_CHUNK_SIZE, start, chunk);
            values.extend_from_slice(chunk);
        },
    )?;

    if let Some(patches) = bp.patches() {
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patch_values.as_slice::<T>();
        match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
            for (&index, &value) in indices.as_slice::<P>().iter().zip(patch_values) {
                let index = <P as AsPrimitive<usize>>::as_(index) - patches.offset();
                let position = offset + index;
                values[index] = params.decode(position / FL_CHUNK_SIZE, position % FL_CHUNK_SIZE, value);
            }
        });
    }
    Ok(PrimitiveArray::new(values.freeze(), bp.validity()?))
}

struct Params<T> {
    references: ChunkParam<T>,
    scales: ChunkParam<T>,
    slopes: ChunkParam<i64>,
    offset: usize,
    shift: u8,
}

impl<T> Params<T>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
{
    /// Turn the residuals of chunk `chunk_idx`, whose first value sits at position `start` within
    /// the chunk, into values in place.
    #[inline]
    fn apply_chunk(&self, chunk_idx: usize, start: usize, chunk: &mut [T]) {
        let scale = self.scales.get(chunk_idx);
        let slope = self.slopes.get(chunk_idx);
        match (scale == T::one(), slope == 0) {
            (true, true) => self.apply::<false, false>(chunk_idx, start, chunk),
            (false, true) => self.apply::<true, false>(chunk_idx, start, chunk),
            (true, false) => self.apply::<false, true>(chunk_idx, start, chunk),
            (false, false) => self.apply::<true, true>(chunk_idx, start, chunk),
        }
    }

    /// `SCALE` and `SLOPE` drop the multiply and the slope term for chunks whose scale is one or
    /// whose slope is zero.
    #[inline(always)]
    fn apply<const SCALE: bool, const SLOPE: bool>(
        &self,
        chunk_idx: usize,
        start: usize,
        chunk: &mut [T],
    ) {
        let reference = self.references.get(chunk_idx);
        let scale = self.scales.get(chunk_idx);
        let slope = self.slopes.get(chunk_idx);
        for (j, value) in (start..).zip(chunk.iter_mut()) {
            let mut v = *value;
            if SCALE {
                v = v.wrapping_mul(&scale);
            }
            v = v.wrapping_add(&reference);
            if SLOPE {
                v = v.wrapping_add(&slope_term(slope, j, self.shift).as_());
            }
            *value = v;
        }
    }

    fn decode(&self, chunk_idx: usize, j: usize, encoded: T) -> T {
        decode_one(
            encoded,
            self.references.get(chunk_idx),
            self.scales.get(chunk_idx),
            self.slopes.get(chunk_idx),
            j,
            self.shift,
        )
    }
}

/// Decode the single element at chunk `chunk`, in-chunk position `j`, from its residual.
#[inline]
pub(crate) fn decode_one<T>(encoded: T, reference: T, scale: T, slope: i64, j: usize, shift: u8) -> T
where
    T: PrimInt + WrappingAdd + WrappingMul + 'static,
    i64: AsPrimitive<T>,
{
    encoded
        .wrapping_mul(&scale)
        .wrapping_add(&reference)
        .wrapping_add(&slope_term(slope, j, shift).as_())
}
