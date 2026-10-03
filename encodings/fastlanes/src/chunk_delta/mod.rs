// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lag-1 deltas that restart at every 1024-element chunk, less each chunk's smallest delta.
//!
//! This is Parquet's `DELTA_BINARY_PACKED` with 1024-value blocks: each chunk stores its first
//! value, its smallest delta, and every delta less that minimum, which is never negative and so
//! packs at the chunk's own width with [`VarBitPacked`], or at one width with
//! [`BitPacked`](crate::BitPacked). Unlike [`Delta`](crate::Delta), which keeps a base per SIMD lane
//! (16 per chunk for 64-bit values) and leaves negative deltas to later layers, it keeps one base
//! and one minimum per chunk. Unlike zigzag-encoding the deltas, subtracting the minimum never
//! needs more bits than the deltas' range.
//!
//! Element `i` sits at position `p = offset + i`, in chunk `c = p / 1024` at position
//! `j = p % 1024`, and decodes as
//!
//! ```text
//! value[p] = bases[c] + Σ_{k=0..=j} (deltas[c * 1024 + k] + mins[c])
//! ```
//!
//! with wrapping arithmetic. Each chunk's first delta is zero, so `bases[c]` is the chunk's first
//! value less its minimum. The `deltas` child covers whole chunks from the start of the first, so
//! it holds `offset + len` values.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::BitPacking;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::WrappingSub;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::slice::SliceArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::require_child;
use vortex_array::require_validity;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::VarBitPacked;
use crate::VarBitPackedArrayExt;
use crate::unpack_iter::for_each_packed_chunk;
use crate::varbitpacked::unpack_chunk;

mod plugin;
#[cfg(test)]
mod tests;

pub use plugin::ChunkDeltaPlugin;

/// The serialized and in-memory ID of [`ChunkDelta`].
pub fn chunk_delta_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.chunk_delta");
    *ID
}

#[array_slots(ChunkDelta)]
pub struct ChunkDeltaSlots {
    /// Lag-1 deltas less their chunk's minimum, as unsigned integers no wider than the array's
    /// type, `offset + len` of them, starting at the first chunk's start. Each chunk's first delta
    /// is zero.
    #[slot(0)]
    pub deltas: ArrayRef,
    /// Each chunk's first value less its minimum delta, of the array's type.
    #[slot(1)]
    pub bases: ArrayRef,
    /// Each chunk's smallest delta, of the array's type.
    #[slot(2)]
    pub mins: ArrayRef,
    /// The validity bitmap indicating which elements are non-null.
    #[slot(3)]
    pub validity_child: Option<ArrayRef>,
}

/// A [`ChunkDelta`]-encoded array.
pub type ChunkDeltaArray = Array<ChunkDelta>;

#[derive(Clone, Debug)]
pub struct ChunkDelta;

#[derive(Clone, Debug)]
pub struct ChunkDeltaData {
    /// The position of the first element within the first chunk.
    pub(crate) offset: u16,
}

impl Display for ChunkDeltaData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

impl ArrayHash for ChunkDeltaData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
    }
}

impl ArrayEq for ChunkDeltaData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset
    }
}

/// The number of chunks `offset + len` elements span.
fn num_chunks(offset: u16, len: usize) -> usize {
    (usize::from(offset) + len).div_ceil(FL_CHUNK_SIZE)
}

pub trait ChunkDeltaArrayExt: ChunkDeltaArraySlotsExt {
    /// The position of the first element within the first chunk.
    fn offset(&self) -> u16 {
        self.offset
    }

    fn validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.as_ref().dtype().nullability())
    }
}

impl<T: TypedArrayRef<ChunkDelta>> ChunkDeltaArrayExt for T {}

impl ChunkDelta {
    /// Construct from deltas less their chunk's minimum, and per-chunk bases and minimums.
    ///
    /// `deltas` must be a non-nullable unsigned array no wider than the array's type, holding
    /// `offset + len` values; `bases` and `mins` non-nullable arrays of the array's type with one
    /// value per chunk.
    pub fn try_new(
        deltas: ArrayRef,
        bases: ArrayRef,
        mins: ArrayRef,
        validity: Validity,
        len: usize,
        offset: u16,
    ) -> VortexResult<ChunkDeltaArray> {
        let dtype = bases.dtype().with_nullability(validity.nullability());
        let slots = smallvec![
            Some(deltas),
            Some(bases),
            Some(mins),
            validity_to_child(&validity, len)
        ];
        Array::try_from_parts(
            ArrayParts::new(ChunkDelta, dtype, len, ChunkDeltaData { offset }).with_slots(slots),
        )
    }

    /// Encode `array` as lag-1 deltas restarting at every chunk, less each chunk's minimum.
    ///
    /// The deltas are stored in the narrowest unsigned type that holds the largest of them, so
    /// they unpack into narrow lanes. Null positions repeat the previous value.
    pub fn encode(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ChunkDeltaArray> {
        let validity = array.validity()?;
        let mask = validity.execute_mask(array.len(), ctx)?;
        let ptype = array.ptype();
        let unsigned = array.reinterpret_cast(ptype.to_unsigned());
        let (deltas, bases, mins) = match_each_unsigned_integer_ptype!(unsigned.ptype(), |U| {
            let (deltas, bases, mins) = encode_deltas(unsigned.as_slice::<U>(), &mask);
            (
                narrow(&deltas),
                PrimitiveArray::new(bases, Validity::NonNullable).reinterpret_cast(ptype),
                PrimitiveArray::new(mins, Validity::NonNullable).reinterpret_cast(ptype),
            )
        });
        Self::try_new(
            deltas,
            bases.into_array(),
            mins.into_array(),
            validity,
            array.len(),
            0,
        )
    }
}

/// `deltas` as the narrowest unsigned array that holds the largest of them.
fn narrow<U: NativePType + PrimInt + AsPrimitive<u8> + AsPrimitive<u16> + AsPrimitive<u32>>(
    deltas: &[U],
) -> ArrayRef {
    let max = deltas.iter().fold(U::zero(), |m, &d| m | d);
    let bits = size_of::<U>() * 8 - max.leading_zeros() as usize;
    let cast = |f: &dyn Fn() -> PrimitiveArray| f().into_array();
    match bits {
        _ if bits.div_ceil(8).next_power_of_two() >= size_of::<U>() => cast(&|| {
            PrimitiveArray::new(Buffer::copy_from(deltas), Validity::NonNullable)
        }),
        0..=8 => cast(&|| {
            let d: Buffer<u8> = deltas.iter().map(|&d| d.as_()).collect();
            PrimitiveArray::new(d, Validity::NonNullable)
        }),
        9..=16 => cast(&|| {
            let d: Buffer<u16> = deltas.iter().map(|&d| d.as_()).collect();
            PrimitiveArray::new(d, Validity::NonNullable)
        }),
        _ => cast(&|| {
            let d: Buffer<u32> = deltas.iter().map(|&d| d.as_()).collect();
            PrimitiveArray::new(d, Validity::NonNullable)
        }),
    }
}

/// Per chunk, the deltas less the chunk's minimum, the first value less the minimum, and the
/// minimum, all as the unsigned bit patterns of the array's type.
fn encode_deltas<U>(values: &[U], mask: &vortex_mask::Mask) -> (Buffer<U>, Buffer<U>, Buffer<U>)
where
    U: NativePType + PrimInt + WrappingAdd + WrappingSub,
{
    // Comparing with the sign bit flipped orders the bit patterns as signed integers.
    let sign = U::one() << (size_of::<U>() * 8 - 1);
    let num_chunks = values.len().div_ceil(FL_CHUNK_SIZE);
    let mut deltas = BufferMut::<U>::with_capacity(values.len());
    let mut bases = BufferMut::<U>::with_capacity(num_chunks);
    let mut mins = BufferMut::<U>::with_capacity(num_chunks);
    let mut prev = U::zero();
    let mut chunk = [U::zero(); FL_CHUNK_SIZE];
    for (c, values) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        let start = c * FL_CHUNK_SIZE;
        for (j, &v) in values.iter().enumerate() {
            let v = if mask.value(start + j) { v } else { prev };
            chunk[j] = if j == 0 { U::zero() } else { v.wrapping_sub(&prev) };
            if j == 0 {
                bases.push(v);
            }
            prev = v;
        }
        let deltas_of_chunk = &chunk[..values.len()];
        let min = deltas_of_chunk[1.min(values.len())..]
            .iter()
            .map(|&d| d ^ sign)
            .min()
            .map_or(U::zero(), |m| m ^ sign);
        let first = bases.len() - 1;
        bases[first] = bases[first].wrapping_sub(&min);
        mins.push(min);
        deltas.push(U::zero());
        deltas.extend(deltas_of_chunk[1.min(values.len())..].iter().map(|&d| d.wrapping_sub(&min)));
    }
    (deltas.freeze(), bases.freeze(), mins.freeze())
}

/// Write the running sum of `base` and the deltas, widened from `D`, each plus `min`, into `dst`.
#[inline(always)]
fn prefix_sum<U: Scan<D>, D>(base: U, min: U, deltas: &[D], dst: &mut [U]) {
    U::scan(base, min, deltas, dst)
}

/// The running sum of `base` and the deltas of type `D`, each plus `min`, written to `dst`.
pub(crate) trait Scan<D>: Sized {
    fn scan(base: Self, min: Self, deltas: &[D], dst: &mut [Self]);
}

/// One dependent add per value.
#[inline(always)]
fn scan_scalar<U, D>(base: U, min: U, deltas: &[D], dst: &mut [U])
where
    U: PrimInt + WrappingAdd + 'static,
    D: AsPrimitive<U>,
{
    let mut acc = base;
    for (o, &d) in dst.iter_mut().zip(deltas) {
        acc = acc.wrapping_add(&d.as_().wrapping_add(&min));
        *o = acc;
    }
}

macro_rules! impl_scalar_scan {
    ($($u:ty => [$($d:ty),*]),*) => {$($(
        impl Scan<$d> for $u {
            #[inline(always)]
            fn scan(base: Self, min: Self, deltas: &[$d], dst: &mut [Self]) {
                scan_scalar(base, min, deltas, dst)
            }
        }
    )*)*};
}

impl_scalar_scan!(u8 => [u8], u16 => [u8, u16]);
// Deltas are never wider than the values, but `decompress` names every pair.
impl_scalar_scan!(u8 => [u16, u32, u64], u16 => [u32, u64], u32 => [u64]);

/// With AVX-512, each block of lanes is widened, offset by the minimum and scanned in registers in log
/// steps of lane shifts and adds, and only one add per block carries between blocks, against a
/// dependent add per value for the scalar loop. Each output value is stored once. Like the rest
/// of the crate, this relies on building for the CPU (`-C target-cpu=native` or `x86-64-v4`)
/// rather than runtime dispatch.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512 {
    use std::arch::x86_64::*;

    use super::Scan;
    use super::scan_scalar;

    /// Load one block of deltas, zero-extended to the lanes of the output.
    trait Widen: Copy {
        /// # Safety
        /// `ptr` must point at a whole block of values.
        unsafe fn load(ptr: *const Self) -> __m512i;
    }

    macro_rules! widen {
        ($t:ty, $ptr:ident => $load:expr) => {
            impl Widen for $t {
                #[inline(always)]
                unsafe fn load($ptr: *const Self) -> __m512i {
                    // SAFETY: the caller guarantees a whole block at `ptr`.
                    unsafe { $load }
                }
            }
        };
    }

    /// Deltas of a narrower type than the 64-bit output, eight per block.
    #[derive(Clone, Copy)]
    struct To64<D>(D);
    /// Deltas of a narrower type than the 32-bit output, sixteen per block.
    #[derive(Clone, Copy)]
    struct To32<D>(D);

    widen!(To64<u8>, p => _mm512_cvtepu8_epi64(_mm_loadl_epi64(p.cast())));
    widen!(To64<u16>, p => _mm512_cvtepu16_epi64(_mm_loadu_si128(p.cast())));
    widen!(To64<u32>, p => _mm512_cvtepu32_epi64(_mm256_loadu_si256(p.cast())));
    widen!(To64<u64>, p => _mm512_loadu_si512(p.cast()));
    widen!(To32<u8>, p => _mm512_cvtepu8_epi32(_mm_loadu_si128(p.cast())));
    widen!(To32<u16>, p => _mm512_cvtepu16_epi32(_mm256_loadu_si256(p.cast())));
    widen!(To32<u32>, p => _mm512_loadu_si512(p.cast()));

    #[inline(always)]
    fn scan_u64<D: Copy>(base: u64, min: u64, deltas: &[D], dst: &mut [u64])
    where
        To64<D>: Widen,
        D: num_traits::AsPrimitive<u64>,
    {
        let n = dst.len().min(deltas.len());
        let blocks = n / 8;
        // SAFETY: AVX-512F is enabled at compile time; block `b` reads deltas and writes values
        // `8b..8b + 8`, all within the first `n`.
        let carry = unsafe {
            let zero = _mm512_setzero_si512();
            let min_v = _mm512_set1_epi64(min as i64);
            let last = _mm512_set1_epi64(7);
            let mut carry = _mm512_set1_epi64(base as i64);
            for b in 0..blocks {
                let z = To64::<D>::load(deltas.as_ptr().add(8 * b).cast());
                let mut x = _mm512_add_epi64(z, min_v);
                // `alignr(x, zero, 8 - k)` shifts the lanes up by `k`, filling with zeros.
                x = _mm512_add_epi64(x, _mm512_alignr_epi64::<7>(x, zero));
                x = _mm512_add_epi64(x, _mm512_alignr_epi64::<6>(x, zero));
                x = _mm512_add_epi64(x, _mm512_alignr_epi64::<4>(x, zero));
                // The block's total is broadcast before the carry is added, so the carry chain
                // between blocks is a single add.
                let total = _mm512_permutexvar_epi64(last, x);
                _mm512_storeu_si512(dst.as_mut_ptr().add(8 * b).cast(), _mm512_add_epi64(x, carry));
                carry = _mm512_add_epi64(carry, total);
            }
            _mm_cvtsi128_si64(_mm512_castsi512_si128(carry)) as u64
        };
        scan_scalar(carry, min, &deltas[8 * blocks..n], &mut dst[8 * blocks..n]);
    }

    #[inline(always)]
    fn scan_u32<D: Copy>(base: u32, min: u32, deltas: &[D], dst: &mut [u32])
    where
        To32<D>: Widen,
        D: num_traits::AsPrimitive<u32>,
    {
        let n = dst.len().min(deltas.len());
        let blocks = n / 16;
        // SAFETY: as in `scan_u64`, with blocks of 16 values.
        let carry = unsafe {
            let zero = _mm512_setzero_si512();
            let min_v = _mm512_set1_epi32(min as i32);
            let last = _mm512_set1_epi32(15);
            let mut carry = _mm512_set1_epi32(base as i32);
            for b in 0..blocks {
                let z = To32::<D>::load(deltas.as_ptr().add(16 * b).cast());
                let mut x = _mm512_add_epi32(z, min_v);
                x = _mm512_add_epi32(x, _mm512_alignr_epi32::<15>(x, zero));
                x = _mm512_add_epi32(x, _mm512_alignr_epi32::<14>(x, zero));
                x = _mm512_add_epi32(x, _mm512_alignr_epi32::<12>(x, zero));
                x = _mm512_add_epi32(x, _mm512_alignr_epi32::<8>(x, zero));
                let total = _mm512_permutexvar_epi32(last, x);
                _mm512_storeu_si512(dst.as_mut_ptr().add(16 * b).cast(), _mm512_add_epi32(x, carry));
                carry = _mm512_add_epi32(carry, total);
            }
            _mm_cvtsi128_si32(_mm512_castsi512_si128(carry)) as u32
        };
        scan_scalar(carry, min, &deltas[16 * blocks..n], &mut dst[16 * blocks..n]);
    }

    macro_rules! impl_simd_scan {
        ($u:ty, $f:ident, [$($d:ty),*]) => {$(
            impl Scan<$d> for $u {
                #[inline(always)]
                fn scan(base: Self, min: Self, deltas: &[$d], dst: &mut [Self]) {
                    $f(base, min, deltas, dst)
                }
            }
        )*};
    }

    impl_simd_scan!(u32, scan_u32, [u8, u16, u32]);
    impl_simd_scan!(u64, scan_u64, [u8, u16, u32, u64]);
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
impl_scalar_scan!(u32 => [u8, u16, u32], u64 => [u8, u16, u32, u64]);

fn decompress(array: &ChunkDeltaArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let ptype = array.dtype().as_ptype();
    let unsigned = ptype.to_unsigned();
    let bases = array
        .bases()
        .as_::<Primitive>()
        .into_owned()
        .reinterpret_cast(unsigned);
    let mins = array
        .mins()
        .as_::<Primitive>()
        .into_owned()
        .reinterpret_cast(unsigned);
    let validity = ChunkDeltaArrayExt::validity(array);
    let deltas_ptype = array.deltas().dtype().as_ptype();
    let values = match_each_unsigned_integer_ptype!(unsigned, |U| {
        let (bases, mins) = (bases.as_slice::<U>(), mins.as_slice::<U>());
        let values = match deltas_ptype {
            PType::U8 => decompress_typed::<U, u8>(array, bases, mins, ctx)?,
            PType::U16 => decompress_typed::<U, u16>(array, bases, mins, ctx)?,
            PType::U32 => decompress_typed::<U, u32>(array, bases, mins, ctx)?,
            _ => decompress_typed::<U, u64>(array, bases, mins, ctx)?,
        };
        PrimitiveArray::new(values, validity)
    });
    Ok(values.reinterpret_cast(ptype))
}

/// Decode into the unsigned type `U` of the array's width, from deltas of the type `D`.
fn decompress_typed<U, D>(
    array: &ChunkDeltaArray,
    bases: &[U],
    mins: &[U],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<U>>
where
    U: NativePType + Scan<D>,
    D: PhysicalPType<Physical = D> + PrimInt + BitPacking,
{
    let len = array.len();
    let offset = usize::from(array.offset);
    let mut values = BufferMut::<U>::with_capacity_in(len, ctx.allocator().clone());
    let output: &mut [MaybeUninit<U>] = &mut values.spare_capacity_mut()[..len];
    let mut scratch = [D::zero(); FL_CHUNK_SIZE];
    let mut decoded = [U::zero(); FL_CHUNK_SIZE];
    // `range` counts from the start of the first chunk; the output starts at `offset`.
    let mut emit = |deltas: &[D], range: Range<usize>| {
        let chunk = range.start / FL_CHUNK_SIZE;
        let skip = offset.saturating_sub(range.start);
        let dst = &mut output[range.start + skip - offset..range.end - offset];
        if skip == 0 && dst.len() == FL_CHUNK_SIZE {
            // SAFETY: the prefix sum writes every value of `dst`.
            let dst = unsafe { std::mem::transmute::<&mut [MaybeUninit<U>], &mut [U]>(dst) };
            prefix_sum(bases[chunk], mins[chunk], &deltas[..FL_CHUNK_SIZE], dst);
        } else {
            let n = range.len();
            prefix_sum(bases[chunk], mins[chunk], &deltas[..n], &mut decoded[..n]);
            for (o, &v) in dst.iter_mut().zip(&decoded[skip..n]) {
                o.write(v);
            }
        }
    };

    let deltas = array.deltas();
    let end = offset + len;
    if let Some(vbp) = deltas.as_opt::<VarBitPacked>() {
        vbp.data().for_each_chunk::<D>(end, |packed, width, range| {
            unpack_chunk(packed, width, &mut scratch);
            emit(&scratch, range);
        });
    } else if let Some(bp) = deltas.as_opt::<BitPacked>() {
        let width = bp.bit_width() as usize;
        // Patch positions, counted from the start of the first chunk, and their values. Both are
        // sorted, so each chunk takes the next run of them.
        let (positions, patch_values) = match bp.patches() {
            Some(patches) => {
                let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
                let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
                let positions: Vec<usize> = match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
                    indices
                        .as_slice::<P>()
                        .iter()
                        .map(|&i| <P as AsPrimitive<usize>>::as_(i) - patches.offset())
                        .collect()
                });
                (positions, values.as_slice::<D>().to_vec())
            }
            None => (Vec::new(), Vec::new()),
        };
        let mut next = 0;
        for_each_packed_chunk::<D, _>(bp.packed_slice::<D>(), width, 0, end, |packed, range| {
            // SAFETY: `packed` holds one chunk at `width` and `scratch` is a whole chunk.
            unsafe { BitPacking::unchecked_unpack(width, packed, &mut scratch) };
            while next < positions.len() && positions[next] < range.end {
                scratch[positions[next] - range.start] = patch_values[next];
                next += 1;
            }
            emit(&scratch, range);
        })?;
    } else {
        let deltas = deltas.as_::<Primitive>();
        for (chunk, deltas) in deltas.as_slice::<D>().chunks(FL_CHUNK_SIZE).enumerate() {
            let start = chunk * FL_CHUNK_SIZE;
            emit(deltas, start..start + deltas.len());
        }
    }
    // SAFETY: every position was written above.
    unsafe { values.set_len(len) };
    Ok(values.freeze())
}

/// Whether the fused decode can read `deltas` directly: bit-packed at one width per chunk or one
/// width for all, aligned to the chunks.
fn is_fusable(deltas: &ArrayRef) -> bool {
    deltas
        .as_opt::<VarBitPacked>()
        .is_some_and(|vbp| vbp.offset() == 0)
        || deltas
            .as_opt::<BitPacked>()
            .is_some_and(|bp| bp.offset() == 0)
}

impl VTable for ChunkDelta {
    type TypedArrayData = ChunkDeltaData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        chunk_delta_id()
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(dtype.is_int(), "ChunkDelta requires an integer dtype, got {dtype}");
        vortex_ensure!(
            usize::from(data.offset) < FL_CHUNK_SIZE,
            "ChunkDelta offset must be less than {FL_CHUNK_SIZE}, got {}",
            data.offset
        );
        let slots = ChunkDeltaSlotsView::from_slots(slots);
        let deltas_dtype = slots.deltas.dtype();
        vortex_ensure!(
            deltas_dtype.is_unsigned_int()
                && !deltas_dtype.is_nullable()
                && deltas_dtype.as_ptype().byte_width() <= dtype.as_ptype().byte_width(),
            "ChunkDelta deltas must be non-nullable unsigned integers no wider than {dtype}, got \
             {deltas_dtype}"
        );
        vortex_ensure!(
            slots.deltas.len() == usize::from(data.offset) + len,
            "ChunkDelta expects {} deltas, got {}",
            usize::from(data.offset) + len,
            slots.deltas.len()
        );
        vortex_ensure!(
            slots.bases.dtype() == &dtype.as_nonnullable(),
            "ChunkDelta bases must be {}, got {}",
            dtype.as_nonnullable(),
            slots.bases.dtype()
        );
        vortex_ensure!(
            slots.mins.dtype() == &dtype.as_nonnullable(),
            "ChunkDelta mins must be {}, got {}",
            dtype.as_nonnullable(),
            slots.mins.dtype()
        );
        let chunks = num_chunks(data.offset, len);
        for (name, child) in [("bases", slots.bases), ("mins", slots.mins)] {
            vortex_ensure!(
                child.len() == chunks,
                "ChunkDelta expects {chunks} {name}, got {}",
                child.len()
            );
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ChunkDeltaArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ChunkDeltaSlots::NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("ChunkDelta serialization requires ChunkDeltaPlugin")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("ChunkDelta deserialization requires ChunkDeltaPlugin")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        require_validity!(array, ChunkDeltaSlots::VALIDITY_CHILD);
        let array = require_child!(array, array.bases(), ChunkDeltaSlots::BASES => Primitive);
        let array = require_child!(array, array.mins(), ChunkDeltaSlots::MINS => Primitive);
        // A slice of bit-packed deltas with patches stays lazy until executed. Step it to the
        // sliced bit-packed array, which starts at a chunk boundary, so the fused decode applies.
        let array = if array
            .deltas()
            .as_opt::<Slice>()
            .is_some_and(|slice| slice.child().is::<BitPacked>())
        {
            require_child!(array, array.deltas(), ChunkDeltaSlots::DELTAS => BitPacked)
        } else {
            array
        };
        let array = if is_fusable(array.deltas()) {
            array
        } else {
            require_child!(array, array.deltas(), ChunkDeltaSlots::DELTAS => Primitive)
        };
        Ok(ExecutionResult::done(decompress(&array, ctx)?.into_array()))
    }
}

const RULES: ParentRuleSet<ChunkDelta> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(ChunkDelta))]);

impl SliceReduce for ChunkDelta {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // Keep the deltas from the start of the first chunk the slice touches, since the values
        // before the slice within that chunk are needed to reach its first value.
        let offset = usize::from(array.offset);
        let start = offset + range.start;
        let end = offset + range.end;
        let first = start / FL_CHUNK_SIZE;
        let validity = ChunkDeltaArrayExt::validity(&array).slice(range.clone())?;
        let chunks = first..end.div_ceil(FL_CHUNK_SIZE).max(first);
        Ok(Some(
            ChunkDelta::try_new(
                array.deltas().slice(first * FL_CHUNK_SIZE..end)?,
                array.bases().slice(chunks.clone())?,
                array.mins().slice(chunks)?,
                validity,
                range.len(),
                u16::try_from(start % FL_CHUNK_SIZE)?,
            )?
            .into_array(),
        ))
    }
}

impl ValidityVTable<ChunkDelta> for ChunkDelta {
    fn validity(array: ArrayView<'_, ChunkDelta>) -> VortexResult<Validity> {
        Ok(ChunkDeltaArrayExt::validity(&array))
    }
}

impl OperationsVTable<ChunkDelta> for ChunkDelta {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, ChunkDelta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Decode the chunk up to `index`.
        let position = usize::from(array.offset) + index;
        let chunk = position / FL_CHUNK_SIZE;
        let chunk_start = chunk * FL_CHUNK_SIZE;
        let sliced = ChunkDelta::try_new(
            array.deltas().slice(chunk_start..position + 1)?,
            array.bases().slice(chunk..chunk + 1)?,
            array.mins().slice(chunk..chunk + 1)?,
            Validity::from(array.dtype().nullability()),
            position + 1 - chunk_start,
            0,
        )?;
        let decoded = decompress_unfused(sliced, ctx)?;
        let last = decoded.len() - 1;
        decoded.into_array().execute_scalar(last, ctx)
    }
}

/// Decode after executing the children, for arrays that did not come through `execute`.
fn decompress_unfused(array: ChunkDeltaArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let deltas = array.deltas().clone();
    let deltas = if is_fusable(&deltas) {
        deltas
    } else {
        deltas.execute::<PrimitiveArray>(ctx)?.into_array()
    };
    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?.into_array();
    let mins = array.mins().clone().execute::<PrimitiveArray>(ctx)?.into_array();
    let array = ChunkDelta::try_new(
        deltas,
        bases,
        mins,
        ChunkDeltaArrayExt::validity(&array),
        array.len(),
        array.offset,
    )?;
    decompress(&array, ctx)
}
