// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::mem;
use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use fastlanes::FoR;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::WrappingMul;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Affine;
use crate::AffineArray;
use vortex_array::IntoArray;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::slope_term;
use crate::affine::array::slope_term_split;
use crate::affine::array::split_slope;
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
    if let Some(dict) = array.encoded().as_opt::<Dict>()
        && dict.values().len() <= MAX_FUSED_DICT_VALUES
    {
        let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
        let values = dict.values().clone().execute::<PrimitiveArray>(ctx)?;
        if codes.all_valid(ctx)? && values.all_valid(ctx)? {
            return match_each_integer_ptype!(array.ptype(), |T| {
                match_each_unsigned_integer_ptype!(codes.ptype(), |C| {
                    decompress_dict::<T, C>(array, &codes, &values, ctx)
                })
            });
        }
    }
    if array.encoded().is::<Dict>() {
        // Not fusable: decode the dictionary, then the residuals as usual.
        let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?.into_array();
        let array = Affine::try_new(
            encoded,
            array.references().clone(),
            array.scales().clone(),
            array.slopes().clone(),
            array.offset(),
            array.slope_shift(),
        )?;
        return decompress(&array, ctx);
    }
    let encoded_ptype = array.encoded().dtype().as_ptype();
    if encoded_ptype.byte_width() < array.ptype().byte_width() {
        return match_each_integer_ptype!(array.ptype(), |T| {
            match_each_unsigned_integer_ptype!(encoded_ptype, |U| {
                decompress_narrow::<T, U>(array, ctx)
            })
        });
    }
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

/// Unpack each bit-packed chunk straight into the output and apply its model while it is in L1.
///
/// Chunks whose scale is one and slope is zero decode with FastLanes' fused unpack-and-add, exactly
/// like [`FoR`](crate::FoR). Partial first and last chunks unpack into a scratch chunk.
fn fused_decompress_typed<T>(
    array: &AffineArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: PhysicalPType<Physical: BitPacking + FoR> + AsPrimitive<T::Physical> + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
{
    let len = bp.len();
    let params = Params::<T>::new(array);
    let offset = usize::from(bp.offset());
    let bit_width = bp.bit_width() as usize;
    let mut values = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
    // SAFETY: `T::Physical` is `T` with the same size and alignment, and every unpack below
    // initializes the whole destination before it is read.
    let output = unsafe {
        mem::transmute::<&mut [MaybeUninit<T>], &mut [T::Physical]>(
            &mut values.spare_capacity_mut()[..len],
        )
    };
    let mut scratch = [T::Physical::default(); FL_CHUNK_SIZE];
    for_each_packed_chunk::<T, _>(
        bp.packed_slice::<T::Physical>(),
        bit_width,
        offset,
        len,
        |packed, range| {
            let chunk_idx = range.start / FL_CHUNK_SIZE;
            // `range` counts from the start of the first chunk, and the output starts at `offset`.
            let skip = offset.saturating_sub(range.start);
            let dst = &mut output[range.start + skip - offset..range.end - offset];
            if dst.len() == FL_CHUNK_SIZE {
                if params.is_flat(chunk_idx) {
                    let reference = params.references.get(chunk_idx).as_();
                    // SAFETY: `packed` holds one chunk at `bit_width` and `dst` is a whole chunk.
                    unsafe { FoR::unchecked_unfor_pack(bit_width, packed, reference, dst) };
                } else {
                    // SAFETY: as above.
                    unsafe { BitPacking::unchecked_unpack(bit_width, packed, dst) };
                    params.apply_chunk(chunk_idx, 0, physical_as_logical::<T>(dst));
                }
            } else {
                // SAFETY: `packed` holds one chunk at `bit_width` and `scratch` is a whole chunk.
                unsafe { BitPacking::unchecked_unpack(bit_width, packed, &mut scratch) };
                let chunk = &mut scratch[skip..range.len()];
                params.apply_chunk(chunk_idx, skip, physical_as_logical::<T>(chunk));
                dst.copy_from_slice(chunk);
            }
        },
    )?;
    // SAFETY: the loop above initialized every value.
    unsafe { values.set_len(len) };

    if let Some(patches) = bp.patches() {
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patch_values.as_slice::<T>();
        match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
            for (&index, &value) in indices.as_slice::<P>().iter().zip(patch_values) {
                let index = <P as AsPrimitive<usize>>::as_(index) - patches.offset();
                let position = offset + index;
                values[index] =
                    params.decode(position / FL_CHUNK_SIZE, position % FL_CHUNK_SIZE, value);
            }
        });
    }
    Ok(PrimitiveArray::new(values.freeze(), bp.validity()?))
}

/// Decode residuals stored in the unsigned type `U`, narrower than the array's type `T`.
///
/// Each chunk unpacks into a narrow scratch chunk that stays in L1, then one pass widens, scales,
/// adds the reference and slope, and writes every output value once.
fn decompress_narrow<T, U>(array: &AffineArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: PhysicalPType<Physical = U> + BitPacking + AsPrimitive<T> + AsPrimitive<u64>,
{
    let len = array.len();
    if len == 0 {
        return Ok(PrimitiveArray::new(Buffer::<T>::empty(), array.encoded().validity()?));
    }
    let params = Params::<T>::new(array);
    let mut values = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
    let output = &mut values.spare_capacity_mut()[..len];

    let validity = match array.encoded().as_opt::<BitPacked>() {
        Some(bp) if bp.offset() == array.offset() => {
            let offset = usize::from(bp.offset());
            let bit_width = bp.bit_width() as usize;
            let mut scratch = [U::default(); FL_CHUNK_SIZE];
            for_each_packed_chunk::<U, _>(
                bp.packed_slice::<U>(),
                bit_width,
                offset,
                len,
                |packed, range| {
                    // SAFETY: `packed` holds one chunk at `bit_width` and `scratch` is a whole
                    // chunk.
                    unsafe { BitPacking::unchecked_unpack(bit_width, packed, &mut scratch) };
                    // `range` counts from the start of the first chunk; the output starts at
                    // `offset`.
                    let skip = offset.saturating_sub(range.start);
                    let dst = &mut output[range.start + skip - offset..range.end - offset];
                    params.apply_narrow(range.start / FL_CHUNK_SIZE, skip, &scratch[skip..range.len()], dst);
                },
            )?;
            // SAFETY: the loop above initialized every value.
            unsafe { values.set_len(len) };
            if let Some(patches) = bp.patches() {
                let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
                let patch_values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
                let patch_values = patch_values.as_slice::<U>();
                match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
                    for (&index, &value) in indices.as_slice::<P>().iter().zip(patch_values) {
                        let index = <P as AsPrimitive<usize>>::as_(index) - patches.offset();
                        let position = offset + index;
                        values[index] = params.decode(
                            position / FL_CHUNK_SIZE,
                            position % FL_CHUNK_SIZE,
                            value.as_(),
                        );
                    }
                });
            }
            bp.validity()?
        }
        _ => {
            let encoded = array.encoded().as_::<Primitive>().into_owned();
            let residuals = encoded.as_slice::<U>();
            let first_len = (FL_CHUNK_SIZE - params.offset).min(len);
            let (first_src, rest_src) = residuals.split_at(first_len);
            let (first_dst, rest_dst) = output.split_at_mut(first_len);
            let chunks = iter::once((params.offset, first_src, first_dst)).chain(
                rest_src
                    .chunks(FL_CHUNK_SIZE)
                    .zip(rest_dst.chunks_mut(FL_CHUNK_SIZE))
                    .map(|(src, dst)| (0, src, dst)),
            );
            for (chunk_idx, (start, src, dst)) in chunks.enumerate() {
                params.apply_narrow(chunk_idx, start, src, dst);
            }
            // SAFETY: the loop above initialized every value.
            unsafe { values.set_len(len) };
            encoded.validity()?
        }
    };
    Ok(PrimitiveArray::new(values.freeze(), validity))
}

/// The largest dictionary [`decompress_dict`] scales per chunk: building the table costs one
/// multiply-add per entry per chunk, at most a quarter of one per value.
const MAX_FUSED_DICT_VALUES: usize = 256;

/// Decode dictionary-encoded residuals without materializing them.
///
/// Per chunk, the dictionary is scaled and offset once into a table, `table[k] = values[k] *
/// scale + reference`, so each value is one lookup plus the slope term.
fn decompress_dict<T, C>(
    array: &AffineArray,
    codes: &PrimitiveArray,
    values: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    C: NativePType + AsPrimitive<usize>,
{
    let len = array.len();
    let validity = array.encoded().validity()?;
    if len == 0 {
        return Ok(PrimitiveArray::new(Buffer::<T>::empty(), validity));
    }
    let params = Params::<T>::new(array);
    // Residuals of any integer type, as `T` by the same truncation the other paths use.
    let residuals: Vec<T> = match_each_integer_ptype!(values.ptype(), |V| {
        values
            .as_slice::<V>()
            .iter()
            .map(|&v| <u64 as AsPrimitive<T>>::as_(v as u64))
            .collect()
    });
    let codes = codes.as_slice::<C>();
    let mut table = vec![T::zero(); residuals.len()];
    let mut out = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
    let output = &mut out.spare_capacity_mut()[..len];

    let first_len = (FL_CHUNK_SIZE - params.offset).min(len);
    let (first_codes, rest_codes) = codes.split_at(first_len);
    let (first_dst, rest_dst) = output.split_at_mut(first_len);
    let chunks = iter::once((params.offset, first_codes, first_dst)).chain(
        rest_codes
            .chunks(FL_CHUNK_SIZE)
            .zip(rest_dst.chunks_mut(FL_CHUNK_SIZE))
            .map(|(codes, dst)| (0, codes, dst)),
    );
    for (chunk_idx, (start, codes, dst)) in chunks.enumerate() {
        let reference = params.references.get(chunk_idx);
        let scale = params.scales.get(chunk_idx);
        for (entry, &residual) in table.iter_mut().zip(&residuals) {
            *entry = residual.wrapping_mul(&scale).wrapping_add(&reference);
        }
        let slope = params.slopes.get(chunk_idx);
        if slope == 0 {
            for (o, &code) in dst.iter_mut().zip(codes) {
                o.write(table[code.as_()]);
            }
        } else {
            let (whole, frac) = split_slope(slope, params.shift);
            let mut whole_j = slope_term_split(whole, 0, start, 0);
            let mut frac_j = frac.wrapping_mul(start as u32);
            for (o, &code) in dst.iter_mut().zip(codes) {
                let term = whole_j.wrapping_add(i64::from(frac_j >> params.shift));
                o.write(table[code.as_()].wrapping_add(&term.as_()));
                whole_j = whole_j.wrapping_add(whole);
                frac_j = frac_j.wrapping_add(frac);
            }
        }
    }
    // SAFETY: the loop above initialized every value.
    unsafe { out.set_len(len) };
    Ok(PrimitiveArray::new(out.freeze(), validity))
}

/// View unpacked physical values as the array's logical type.
#[inline(always)]
fn physical_as_logical<T: PhysicalPType>(values: &mut [T::Physical]) -> &mut [T] {
    // SAFETY: `T::Physical` is `T` with the same size and alignment, and the model's wrapping
    // arithmetic is the same in two's complement whichever signedness `T` has.
    unsafe { mem::transmute::<&mut [T::Physical], &mut [T]>(values) }
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
    /// Whether chunk `chunk_idx` has a scale of one and a slope of zero, so it is plain FoR.
    #[inline]
    fn is_flat(&self, chunk_idx: usize) -> bool {
        self.scales.get(chunk_idx) == T::one() && self.slopes.get(chunk_idx) == 0
    }

    /// Turn the residuals of chunk `chunk_idx`, whose first value sits at position `start` within
    /// the chunk, into values in place.
    ///
    /// The loops are written to auto-vectorize; build with `-C target-cpu=native` (or a
    /// `x86-64-v4` target) to get AVX-512's native 64-bit multiplies.
    #[inline]
    fn apply_chunk(&self, chunk_idx: usize, start: usize, chunk: &mut [T]) {
        self.apply_chunk_impl(chunk_idx, start, chunk)
    }

    #[inline(always)]
    fn apply_chunk_impl(&self, chunk_idx: usize, start: usize, chunk: &mut [T]) {
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
        let (whole, frac) = split_slope(self.slopes.get(chunk_idx), self.shift);
        let shift = self.shift;
        // Running sums of `whole * j` and `frac * j`, equal to `slope_term_split` at every `j` by
        // the same wrapping arithmetic, keep the loop free of 64-bit multiplies.
        let mut whole_j = slope_term_split(whole, 0, start, 0);
        let mut frac_j = frac.wrapping_mul(start as u32);
        for value in chunk.iter_mut() {
            let mut v = *value;
            if SCALE {
                v = v.wrapping_mul(&scale);
            }
            v = v.wrapping_add(&reference);
            if SLOPE {
                let term = whole_j.wrapping_add(i64::from(frac_j >> shift));
                v = v.wrapping_add(&term.as_());
                whole_j = whole_j.wrapping_add(whole);
                frac_j = frac_j.wrapping_add(frac);
            }
            *value = v;
        }
    }

    /// Write the values of chunk `chunk_idx`, whose first value sits at position `start` within
    /// the chunk, from residuals `src` of the narrower type `U` into `dst`.
    #[inline]
    fn apply_narrow<U>(&self, chunk_idx: usize, start: usize, src: &[U], dst: &mut [MaybeUninit<T>])
    where
        T: AsPrimitive<u64>,
        u64: AsPrimitive<T>,
        U: Copy + AsPrimitive<T> + AsPrimitive<u64>,
    {
        self.apply_narrow_impl(chunk_idx, start, src, dst)
    }

    #[inline(always)]
    fn apply_narrow_impl<U>(&self, chunk_idx: usize, start: usize, src: &[U], dst: &mut [MaybeUninit<T>])
    where
        T: AsPrimitive<u64>,
        u64: AsPrimitive<T>,
        U: Copy + AsPrimitive<T> + AsPrimitive<u64>,
    {
        let scale = self.scales.get(chunk_idx);
        let flat = self.slopes.get(chunk_idx) == 0;
        // Residuals of at most 32 bits times a scale below 2^32 multiply as 32 x 32 -> 64 bits,
        // one instruction on every x86 target.
        let narrow_mul =
            size_of::<U>() <= 4 && <T as AsPrimitive<u64>>::as_(scale) <= u64::from(u32::MAX);
        match (scale == T::one(), flat, narrow_mul) {
            (true, true, _) => self.narrow::<U, false, false, false>(chunk_idx, start, src, dst),
            (true, false, _) => self.narrow::<U, false, true, false>(chunk_idx, start, src, dst),
            (false, true, true) => self.narrow::<U, true, false, true>(chunk_idx, start, src, dst),
            (false, false, true) => self.narrow::<U, true, true, true>(chunk_idx, start, src, dst),
            (false, true, false) => self.narrow::<U, true, false, false>(chunk_idx, start, src, dst),
            (false, false, false) => self.narrow::<U, true, true, false>(chunk_idx, start, src, dst),
        }
    }

    #[inline(always)]
    fn narrow<U, const SCALE: bool, const SLOPE: bool, const NARROW_MUL: bool>(
        &self,
        chunk_idx: usize,
        start: usize,
        src: &[U],
        dst: &mut [MaybeUninit<T>],
    ) where
        T: AsPrimitive<u64>,
        u64: AsPrimitive<T>,
        U: Copy + AsPrimitive<T> + AsPrimitive<u64>,
    {
        let reference = self.references.get(chunk_idx);
        let scale = self.scales.get(chunk_idx);
        let scale32 = u64::from(<T as AsPrimitive<u64>>::as_(scale) as u32);
        let (whole, frac) = split_slope(self.slopes.get(chunk_idx), self.shift);
        let shift = self.shift;
        let mut whole_j = slope_term_split(whole, 0, start, 0);
        let mut frac_j = frac.wrapping_mul(start as u32);
        for (out, &e) in dst.iter_mut().zip(src) {
            let mut v: T = if SCALE && NARROW_MUL {
                // Exact in u64 and truncated to `T`, which equals `T`'s wrapping multiply.
                let e = u64::from(<U as AsPrimitive<u64>>::as_(e) as u32);
                (e * scale32).as_()
            } else if SCALE {
                <U as AsPrimitive<T>>::as_(e).wrapping_mul(&scale)
            } else {
                e.as_()
            };
            v = v.wrapping_add(&reference);
            if SLOPE {
                let term = whole_j.wrapping_add(i64::from(frac_j >> shift));
                v = v.wrapping_add(&term.as_());
                whole_j = whole_j.wrapping_add(whole);
                frac_j = frac_j.wrapping_add(frac);
            }
            out.write(v);
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
