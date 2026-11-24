// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunk-pruned constant comparison kernel for [`Affine`].
//!
//! For each 1,024-element chunk, exact `i128` interval arithmetic verifies whether the affine
//! transform stays within `T`'s representable range `[T::MIN, T::MAX]` (no wraparound). When it
//! does:
//! - Chunks whose interval lies wholly on one side of `rhs`, or whose `scale` does not divide
//!   `rhs - ref` for `Eq`/`NotEq`, short-circuit to a constant bool without unpacking.
//! - Flat (`slope == 0`) candidate chunks rewrite the comparison via exact integer division to a
//!   direct residual threshold, using FastLanes' fused `BitPackingCompare::unchecked_unpack_cmp`
//!   straight on packed bits.
//! - Sloped or wrapping candidate chunks decode into an L1 scratch buffer and pack the boolean
//!   results directly into words.
//!
//! Any `BitPacked` patches overwrite their exact bits afterwards so every chunk benefits from
//! packed-range pruning even when patches are present.

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use fastlanes::transpose_bits;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::WrappingMul;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::binary::CompareKernel;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::pack_bools_into_words;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::resolve_aligned_bitpacked;
use crate::Affine;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::affine_decompress::Params;
use crate::affine::array::num_chunks;
use crate::affine::array::slope_term;
use crate::affine::array::split_slope;

const U64_BITS: usize = u64::BITS as usize;
const WORDS_PER_CHUNK: usize = FL_CHUNK_SIZE / U64_BITS;

/// Primitive comparison run directly on unpacked unsigned residuals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BaseCompare {
    Eq,
    Gt,
}

/// Per-chunk strategy for comparing an `Affine` chunk against a constant `rhs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkComparePlan {
    /// Every unpatched value in the chunk yields the same boolean answer.
    Constant(bool),
    /// Flat no-wrap chunk rewritten to `base(residual, threshold) ^ negate`.
    FusedPacked {
        base: BaseCompare,
        threshold: u64,
        negate: bool,
    },
    /// Sloped or wrapping chunk that must decode into scratch and compare in `T`.
    DecodeAndCompare,
}

impl CompareKernel for Affine {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(constant_prim) = constant.as_primitive_opt() else {
            return Ok(None);
        };
        let lhs_ptype = lhs.dtype().as_ptype();
        if constant_prim.ptype() != lhs_ptype {
            return Ok(None);
        }

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        if lhs.is_empty() {
            return Ok(Some(
                Canonical::empty(&DType::Bool(nullability)).into_array(),
            ));
        }
        if constant.is_null() {
            return Ok(Some(
                ConstantArray::new(Scalar::null(DType::Bool(nullability)), lhs.len()).into_array(),
            ));
        }
        let validity = lhs.validity()?.union_nullability(nullability);
        if matches!(validity, Validity::AllInvalid) {
            return Ok(Some(
                ConstantArray::new(Scalar::null(DType::Bool(nullability)), lhs.len()).into_array(),
            ));
        }

        match_each_integer_ptype!(lhs_ptype, |T| {
            let rhs_val: T = constant_prim
                .typed_value::<T>()
                .vortex_expect("non-null constant matches lhs ptype");
            compare_constant_typed::<T>(lhs, rhs_val, operator, validity, ctx)
        })
    }
}

fn compare_constant_typed<T>(
    lhs: ArrayView<'_, Affine>,
    rhs: T,
    operator: CompareOperator,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: PhysicalPType<
            Physical: BitPacking
                          + BitPackingCompare
                          + FastLanesComparable<Bitpacked = T::Physical>
                          + AsPrimitive<u64>,
        > + NativePType
        + PrimInt
        + WrappingAdd
        + WrappingMul
        + AsPrimitive<i128>
        + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    if let Some(bp_arr) = resolve_aligned_bitpacked(lhs, ctx)? {
        let bp = bp_arr.as_::<BitPacked>();
        let bit_width = bp.bit_width();
        let params = Params::<T>::execute(lhs, ctx)?;
        let encoded_ptype = bp.dtype().as_ptype();
        let result = match_each_unsigned_integer_ptype!(encoded_ptype.to_unsigned(), |U| {
            compare_bitpacked_typed::<T, U>(
                lhs.len(),
                bp,
                bit_width,
                &params,
                rhs,
                operator,
                validity,
                ctx,
            )?
        });
        return Ok(Some(result));
    }

    if let Some(prim) = lhs.encoded().as_opt::<Primitive>() {
        let params = Params::<T>::execute(lhs, ctx)?;
        let encoded_ptype = prim.ptype();
        let result = if encoded_ptype.byte_width() < T::PTYPE.byte_width() {
            match_each_unsigned_integer_ptype!(encoded_ptype, |U| {
                compare_primitive_narrow::<T, U>(
                    prim.as_slice::<U>(),
                    &params,
                    rhs,
                    operator,
                    validity,
                    ctx,
                )
            })
        } else {
            compare_primitive_same_width::<T>(
                prim.as_slice::<T>(),
                &params,
                rhs,
                operator,
                validity,
                ctx,
            )
        };
        return Ok(Some(result));
    }

    Ok(None)
}

#[expect(
    clippy::too_many_arguments,
    reason = "typed bitpacked compare execution parameters"
)]
fn compare_bitpacked_typed<T, U>(
    len: usize,
    bp: ArrayView<'_, BitPacked>,
    bit_width: u8,
    params: &Params<T>,
    rhs: T,
    operator: CompareOperator,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<i128> + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: PhysicalPType<Physical = U>
        + BitPacking
        + BitPackingCompare
        + FastLanesComparable<Bitpacked = U>
        + AsPrimitive<u64>,
    u64: AsPrimitive<U>,
{
    let offset = params.offset;
    let total_chunks = num_chunks(offset as u16, len);
    let max_packed: i128 = if bit_width == 64 {
        i128::from(u64::MAX)
    } else {
        (1i128 << bit_width) - 1
    };

    // Without patches, check if every chunk short-circuits to the same constant boolean.
    if bp.patches().is_none()
        && let Some(constant_answer) =
            all_chunks_constant(len, offset, total_chunks, params, max_packed, rhs, operator)
    {
        return Ok(make_constant_bool(
            constant_answer,
            len,
            validity,
            ctx.allocator().clone(),
        ));
    }

    let bit_width = usize::from(bit_width);
    let packed = bp.packed_slice::<U>();
    let packed_chunk_len = FL_CHUNK_SIZE * bit_width / (8 * size_of::<U>());

    let mut words: BufferMut<u64> =
        BufferMut::zeroed_in(total_chunks * WORDS_PER_CHUNK, ctx.allocator().clone());
    {
        let words_slice = words.as_mut_slice();
        let mut lane_major = [0u64; WORDS_PER_CHUNK];
        let mut scratch_u = [U::default(); FL_CHUNK_SIZE];
        let mut scratch_t = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];

        for chunk_idx in 0..total_chunks {
            let j_lo = if chunk_idx == 0 { offset } else { 0 };
            let j_hi = (offset + len - chunk_idx * FL_CHUNK_SIZE).min(FL_CHUNK_SIZE);
            let plan =
                classify_chunk::<T>(params, chunk_idx, max_packed, j_lo, j_hi - 1, rhs, operator);
            let out = words_slice[chunk_idx * WORDS_PER_CHUNK..]
                .first_chunk_mut::<WORDS_PER_CHUNK>()
                .vortex_expect("over-allocated buffer holds a full block per chunk");

            match plan {
                ChunkComparePlan::Constant(false) => {}
                ChunkComparePlan::Constant(true) => out.fill(u64::MAX),
                ChunkComparePlan::FusedPacked {
                    base,
                    threshold,
                    negate,
                } if bit_width > 0 => {
                    let packed_chunk = &packed[chunk_idx * packed_chunk_len..][..packed_chunk_len];
                    let threshold_u: U = threshold.as_();
                    match base {
                        BaseCompare::Eq => {
                            // SAFETY: `packed_chunk` holds one chunk at `bit_width` bits.
                            unsafe {
                                U::unchecked_unpack_cmp::<U, _>(
                                    bit_width,
                                    packed_chunk,
                                    &mut lane_major,
                                    U::is_eq,
                                    threshold_u,
                                );
                            }
                        }
                        BaseCompare::Gt => {
                            // SAFETY: `packed_chunk` holds one chunk at `bit_width` bits.
                            unsafe {
                                U::unchecked_unpack_cmp::<U, _>(
                                    bit_width,
                                    packed_chunk,
                                    &mut lane_major,
                                    U::is_gt,
                                    threshold_u,
                                );
                            }
                        }
                    }
                    transpose_bits::<U>(&lane_major, out);
                    if negate {
                        out.iter_mut().for_each(|word| *word = !*word);
                    }
                }
                ChunkComparePlan::FusedPacked { .. } | ChunkComparePlan::DecodeAndCompare => {
                    if bit_width == 0 {
                        scratch_u[j_lo..j_hi].fill(U::zero());
                    } else {
                        let packed_chunk =
                            &packed[chunk_idx * packed_chunk_len..][..packed_chunk_len];
                        // SAFETY: `packed_chunk` holds one chunk at `bit_width` bits.
                        unsafe {
                            BitPacking::unchecked_unpack(bit_width, packed_chunk, &mut scratch_u);
                        }
                    }
                    let count = j_hi - j_lo;
                    params.apply_narrow(
                        chunk_idx,
                        j_lo,
                        &scratch_u[j_lo..j_hi],
                        &mut scratch_t[..count],
                    );
                    // SAFETY: `apply_narrow` initialized `scratch_t[..count]`.
                    let decoded = unsafe { scratch_t[..count].assume_init_ref() };
                    pack_bools_into_words(
                        words_slice,
                        chunk_idx * FL_CHUNK_SIZE + j_lo,
                        count,
                        |i| eval_compare(decoded[i], rhs, operator),
                    );
                }
            }
        }
    }

    let mut bits = BitBufferMut::from_buffer(words.into_byte_buffer(), offset, len);

    if let Some(patches) = bp.patches() {
        apply_compare_patches::<T>(&mut bits, &patches, params, rhs, operator, ctx)?;
    }

    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

fn compare_primitive_narrow<T, U>(
    src: &[U],
    params: &Params<T>,
    rhs: T,
    operator: CompareOperator,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> ArrayRef
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<i128> + AsPrimitive<u64>,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    U: NativePType + PrimInt + AsPrimitive<u64>,
    u64: AsPrimitive<U>,
{
    let len = src.len();
    let offset = params.offset;
    let total_chunks = num_chunks(offset as u16, len);
    let max_e = i128::from(<U as AsPrimitive<u64>>::as_(U::max_value()));

    if let Some(constant_answer) =
        all_chunks_constant(len, offset, total_chunks, params, max_e, rhs, operator)
    {
        return make_constant_bool(constant_answer, len, validity, ctx.allocator().clone());
    }

    let mut words: BufferMut<u64> =
        BufferMut::zeroed_in(total_chunks * WORDS_PER_CHUNK, ctx.allocator().clone());
    {
        let words_slice = words.as_mut_slice();
        let mut scratch_t = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
        let mut src_pos = 0usize;

        for chunk_idx in 0..total_chunks {
            let j_lo = if chunk_idx == 0 { offset } else { 0 };
            let j_hi = (offset + len - chunk_idx * FL_CHUNK_SIZE).min(FL_CHUNK_SIZE);
            let count = j_hi - j_lo;
            let chunk_src = &src[src_pos..src_pos + count];
            src_pos += count;

            let plan = classify_chunk::<T>(params, chunk_idx, max_e, j_lo, j_hi - 1, rhs, operator);
            match plan {
                ChunkComparePlan::Constant(false) => {}
                ChunkComparePlan::Constant(true) => {
                    words_slice[chunk_idx * WORDS_PER_CHUNK..][..WORDS_PER_CHUNK].fill(u64::MAX);
                }
                ChunkComparePlan::FusedPacked {
                    base,
                    threshold,
                    negate,
                } => {
                    let threshold_u: U = threshold.as_();
                    let bit_start = chunk_idx * FL_CHUNK_SIZE + j_lo;
                    match (base, negate) {
                        (BaseCompare::Eq, false) => {
                            pack_bools_into_words(words_slice, bit_start, count, |i| {
                                chunk_src[i] == threshold_u
                            });
                        }
                        (BaseCompare::Eq, true) => {
                            pack_bools_into_words(words_slice, bit_start, count, |i| {
                                chunk_src[i] != threshold_u
                            });
                        }
                        (BaseCompare::Gt, false) => {
                            pack_bools_into_words(words_slice, bit_start, count, |i| {
                                chunk_src[i] > threshold_u
                            });
                        }
                        (BaseCompare::Gt, true) => {
                            pack_bools_into_words(words_slice, bit_start, count, |i| {
                                chunk_src[i] <= threshold_u
                            });
                        }
                    }
                }
                ChunkComparePlan::DecodeAndCompare => {
                    params.apply_narrow(chunk_idx, j_lo, chunk_src, &mut scratch_t[..count]);
                    // SAFETY: `apply_narrow` initialized `scratch_t[..count]`.
                    let decoded = unsafe { scratch_t[..count].assume_init_ref() };
                    pack_bools_into_words(
                        words_slice,
                        chunk_idx * FL_CHUNK_SIZE + j_lo,
                        count,
                        |i| eval_compare(decoded[i], rhs, operator),
                    );
                }
            }
        }
    }

    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), offset, len);
    BoolArray::new(bits.freeze(), validity).into_array()
}

fn compare_primitive_same_width<T>(
    src: &[T],
    params: &Params<T>,
    rhs: T,
    operator: CompareOperator,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> ArrayRef
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul + AsPrimitive<i128>,
    i64: AsPrimitive<T>,
{
    let len = src.len();
    let offset = params.offset;
    let total_chunks = num_chunks(offset as u16, len);
    let unsigned_max_e = T::PTYPE.is_unsigned_int().then(|| T::max_value().as_());

    let mut words: BufferMut<u64> =
        BufferMut::zeroed_in(total_chunks * WORDS_PER_CHUNK, ctx.allocator().clone());
    {
        let words_slice = words.as_mut_slice();
        let mut scratch_t = [T::zero(); FL_CHUNK_SIZE];
        let mut src_pos = 0usize;

        for chunk_idx in 0..total_chunks {
            let j_lo = if chunk_idx == 0 { offset } else { 0 };
            let j_hi = (offset + len - chunk_idx * FL_CHUNK_SIZE).min(FL_CHUNK_SIZE);
            let count = j_hi - j_lo;
            let chunk_src = &src[src_pos..src_pos + count];
            src_pos += count;

            let plan = if let Some(max_e) = unsigned_max_e {
                classify_chunk::<T>(params, chunk_idx, max_e, j_lo, j_hi - 1, rhs, operator)
            } else if params.scales.get(chunk_idx) == T::zero() && params.slopes.get(chunk_idx) == 0
            {
                ChunkComparePlan::Constant(eval_compare(
                    params.references.get(chunk_idx),
                    rhs,
                    operator,
                ))
            } else {
                ChunkComparePlan::DecodeAndCompare
            };

            match plan {
                ChunkComparePlan::Constant(false) => {}
                ChunkComparePlan::Constant(true) => {
                    words_slice[chunk_idx * WORDS_PER_CHUNK..][..WORDS_PER_CHUNK].fill(u64::MAX);
                }
                ChunkComparePlan::FusedPacked { .. } | ChunkComparePlan::DecodeAndCompare => {
                    scratch_t[..count].copy_from_slice(chunk_src);
                    params.apply_chunk(chunk_idx, j_lo, &mut scratch_t[..count]);
                    pack_bools_into_words(
                        words_slice,
                        chunk_idx * FL_CHUNK_SIZE + j_lo,
                        count,
                        |i| eval_compare(scratch_t[i], rhs, operator),
                    );
                }
            }
        }
    }

    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), offset, len);
    BoolArray::new(bits.freeze(), validity).into_array()
}

fn apply_compare_patches<T>(
    bits: &mut BitBufferMut,
    patches: &Patches,
    params: &Params<T>,
    rhs: T,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let patch_indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let patch_values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let patch_offset = patches.offset();

    match_each_unsigned_integer_ptype!(patch_indices.ptype(), |P| {
        apply_compare_patches_indices::<T, P>(
            bits,
            patch_indices.as_slice::<P>(),
            patch_offset,
            &patch_values,
            params,
            rhs,
            operator,
        )
    });
    Ok(())
}

fn apply_compare_patches_indices<T, P>(
    bits: &mut BitBufferMut,
    patch_indices: &[P],
    patch_offset: usize,
    patch_values: &PrimitiveArray,
    params: &Params<T>,
    rhs: T,
    operator: CompareOperator,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    P: Copy + AsPrimitive<usize>,
{
    match_each_integer_ptype!(patch_values.ptype(), |V| {
        apply_compare_patches_typed::<T, P, V>(
            bits,
            patch_indices,
            patch_offset,
            patch_values.as_slice::<V>(),
            params,
            rhs,
            operator,
        )
    });
}

fn apply_compare_patches_typed<T, P, V>(
    bits: &mut BitBufferMut,
    patch_indices: &[P],
    patch_offset: usize,
    patch_values: &[V],
    params: &Params<T>,
    rhs: T,
    operator: CompareOperator,
) where
    T: NativePType + PrimInt + WrappingAdd + WrappingMul,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    P: Copy + AsPrimitive<usize>,
    V: NativePType + AsPrimitive<u64>,
{
    for (&p_idx, &p_val) in patch_indices.iter().zip(patch_values) {
        let local_idx = p_idx.as_() - patch_offset;
        let pos = params.offset + local_idx;
        let residual: T = <u64 as AsPrimitive<T>>::as_(p_val.as_());
        let val = params.decode(pos / FL_CHUNK_SIZE, pos % FL_CHUNK_SIZE, residual);
        bits.set_to(local_idx, eval_compare(val, rhs, operator));
    }
}

fn all_chunks_constant<T>(
    len: usize,
    offset: usize,
    total_chunks: usize,
    params: &Params<T>,
    max_e: i128,
    rhs: T,
    operator: CompareOperator,
) -> Option<bool>
where
    T: NativePType + PrimInt + AsPrimitive<i128>,
{
    let mut answer: Option<bool> = None;
    for chunk_idx in 0..total_chunks {
        let j_lo = if chunk_idx == 0 { offset } else { 0 };
        let j_hi = (offset + len - chunk_idx * FL_CHUNK_SIZE).min(FL_CHUNK_SIZE);
        match classify_chunk::<T>(params, chunk_idx, max_e, j_lo, j_hi - 1, rhs, operator) {
            ChunkComparePlan::Constant(b) => match answer {
                None => answer = Some(b),
                Some(prev) if prev == b => {}
                Some(_) => return None,
            },
            ChunkComparePlan::FusedPacked { .. } | ChunkComparePlan::DecodeAndCompare => {
                return None;
            }
        }
    }
    answer
}

fn make_constant_bool(
    value: bool,
    len: usize,
    validity: Validity,
    allocator: vortex_buffer::BufferAllocatorRef,
) -> ArrayRef {
    if validity.definitely_no_nulls() {
        let scalar = Scalar::bool(value, validity.nullability());
        ConstantArray::new(scalar, len).into_array()
    } else {
        BoolArray::new(BitBuffer::full_in(value, len, allocator), validity).into_array()
    }
}

#[inline]
fn eval_compare<T: NativePType>(lhs: T, rhs: T, operator: CompareOperator) -> bool {
    match operator {
        CompareOperator::Eq => lhs.is_eq(rhs),
        CompareOperator::NotEq => !lhs.is_eq(rhs),
        CompareOperator::Lt => lhs.is_lt(rhs),
        CompareOperator::Lte => lhs.is_le(rhs),
        CompareOperator::Gt => lhs.is_gt(rhs),
        CompareOperator::Gte => lhs.is_ge(rhs),
    }
}

/// Classifies chunk `chunk_idx` assuming unpatched residuals lie in `0..=max_e` and active
/// in-chunk positions lie in `j_lo..=j_hi`.
///
/// Exact `i128` interval arithmetic verifies that every `(e, j)` pair produces a value inside
/// `[T::MIN, T::MAX]`, guaranteeing modular wrapping in `T` agrees with mathematical integers.
fn classify_chunk<T>(
    params: &Params<T>,
    chunk_idx: usize,
    max_e: i128,
    j_lo: usize,
    j_hi: usize,
    rhs: T,
    operator: CompareOperator,
) -> ChunkComparePlan
where
    T: NativePType + PrimInt + AsPrimitive<i128>,
{
    let t_min: i128 = T::min_value().as_();
    let t_max: i128 = T::max_value().as_();
    let r: i128 = params.references.get(chunk_idx).as_();
    let s: i128 = params.scales.get(chunk_idx).as_();
    let slope = params.slopes.get(chunk_idx);
    let rhs_i: i128 = rhs.as_();

    let Some(end_val) = s.checked_mul(max_e).and_then(|prod| r.checked_add(prod)) else {
        return ChunkComparePlan::DecodeAndCompare;
    };
    let (base_lo, base_hi) = if r <= end_val {
        (r, end_val)
    } else {
        (end_val, r)
    };

    if slope != 0 {
        return classify_sloped_chunk(
            base_lo,
            base_hi,
            slope,
            params.shift,
            j_lo,
            j_hi,
            t_min,
            t_max,
            rhs_i,
            operator,
        );
    }

    classify_flat_chunk(r, s, base_lo, base_hi, t_min, t_max, rhs_i, operator)
}

#[expect(
    clippy::too_many_arguments,
    reason = "interval bounds classification helper"
)]
fn classify_sloped_chunk(
    base_lo: i128,
    base_hi: i128,
    slope: i64,
    shift: u8,
    j_lo: usize,
    j_hi: usize,
    t_min: i128,
    t_max: i128,
    rhs_i: i128,
    operator: CompareOperator,
) -> ChunkComparePlan {
    let (whole, _frac) = split_slope(slope, shift);
    // Ensure `whole * j + ((frac * j) >> shift)` never wraps in `i64` for `j < FL_CHUNK_SIZE`.
    if i128::from(whole).abs() * (FL_CHUNK_SIZE as i128) + (FL_CHUNK_SIZE as i128)
        >= i128::from(i64::MAX)
    {
        return ChunkComparePlan::DecodeAndCompare;
    }
    let t0 = i128::from(slope_term(slope, j_lo, shift));
    let t1 = i128::from(slope_term(slope, j_hi, shift));
    let (t_lo, t_hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
    let (Some(lo), Some(hi)) = (base_lo.checked_add(t_lo), base_hi.checked_add(t_hi)) else {
        return ChunkComparePlan::DecodeAndCompare;
    };
    if lo < t_min || hi > t_max {
        return ChunkComparePlan::DecodeAndCompare;
    }
    classify_interval_only(lo, hi, rhs_i, operator).map_or(
        ChunkComparePlan::DecodeAndCompare,
        ChunkComparePlan::Constant,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "flat interval classification helper"
)]
fn classify_flat_chunk(
    r: i128,
    s: i128,
    lo: i128,
    hi: i128,
    t_min: i128,
    t_max: i128,
    rhs_i: i128,
    operator: CompareOperator,
) -> ChunkComparePlan {
    if lo < t_min || hi > t_max {
        return ChunkComparePlan::DecodeAndCompare;
    }
    if let Some(constant) = classify_interval_only(lo, hi, rhs_i, operator) {
        return ChunkComparePlan::Constant(constant);
    }

    // `lo < hi`, so `s != 0` and `rhs_i` falls inside `[lo, hi]`.
    match operator {
        CompareOperator::Eq | CompareOperator::NotEq => {
            let negate = matches!(operator, CompareOperator::NotEq);
            let diff = rhs_i - r;
            if diff % s != 0 {
                return ChunkComparePlan::Constant(negate);
            }
            ChunkComparePlan::FusedPacked {
                base: BaseCompare::Eq,
                threshold: (diff / s) as u64,
                negate,
            }
        }
        CompareOperator::Lt | CompareOperator::Gte => {
            let is_lt = matches!(operator, CompareOperator::Lt);
            let (threshold, negate) = if s > 0 {
                (((rhs_i - r - 1) / s) as u64, is_lt)
            } else {
                (((r - rhs_i) / (-s)) as u64, !is_lt)
            };
            ChunkComparePlan::FusedPacked {
                base: BaseCompare::Gt,
                threshold,
                negate,
            }
        }
        CompareOperator::Lte | CompareOperator::Gt => {
            let is_lte = matches!(operator, CompareOperator::Lte);
            let (threshold, negate) = if s > 0 {
                (((rhs_i - r) / s) as u64, is_lte)
            } else {
                (((r - rhs_i - 1) / (-s)) as u64, !is_lte)
            };
            ChunkComparePlan::FusedPacked {
                base: BaseCompare::Gt,
                threshold,
                negate,
            }
        }
    }
}

/// Returns `Some(answer)` if the closed value interval `[lo, hi]` decides `val <op> rhs_i` for
/// every element in the chunk.
#[inline]
fn classify_interval_only(
    lo: i128,
    hi: i128,
    rhs_i: i128,
    operator: CompareOperator,
) -> Option<bool> {
    match operator {
        CompareOperator::Eq if rhs_i < lo || rhs_i > hi => Some(false),
        CompareOperator::Eq if lo == hi => Some(true),
        CompareOperator::NotEq if rhs_i < lo || rhs_i > hi => Some(true),
        CompareOperator::NotEq if lo == hi => Some(false),
        CompareOperator::Lt if hi < rhs_i => Some(true),
        CompareOperator::Lt if lo >= rhs_i => Some(false),
        CompareOperator::Lte if hi <= rhs_i => Some(true),
        CompareOperator::Lte if lo > rhs_i => Some(false),
        CompareOperator::Gt if lo > rhs_i => Some(true),
        CompareOperator::Gt if hi <= rhs_i => Some(false),
        CompareOperator::Gte if lo >= rhs_i => Some(true),
        CompareOperator::Gte if hi < rhs_i => Some(false),
        _ => None,
    }
}
