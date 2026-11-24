// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use fastlanes::transpose_bits;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use num_traits::WrappingSub;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::compare::CompareKernel;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_functions::between::BetweenOptions;
use vortex_functions::between::StrictComparison;
use vortex_functions::builtins::ArrayFunctions;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::FoRReferences;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;
use crate::unpack_iter::BitPacked as BitPackedIter;
use crate::unpack_iter::for_each_packed_chunk;

const U64_BITS: usize = u64::BITS as usize;
const WORDS_PER_CHUNK: usize = FL_CHUNK_SIZE / U64_BITS;

impl CompareKernel for FoR {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(constant) = rhs.as_constant()
            && let Some(constant) = constant.as_primitive_opt()
            && constant.ptype() == lhs.dtype().as_ptype()
        {
            match_each_integer_ptype!(constant.ptype(), |T| {
                return compare_constant(
                    lhs,
                    constant
                        .typed_value::<T>()
                        .vortex_expect("null scalar handled in adaptor"),
                    rhs.dtype().nullability(),
                    operator,
                    ctx,
                );
            })
        }

        Ok(None)
    }
}

/// Compares the array to a constant through its encoded values.
///
/// No valid value is below the reference, so a constant below it answers every row. Otherwise
/// the constant is a distance from the reference. Equality survives the wrapping subtraction.
/// Ordering compares unsigned distances.
fn compare_constant<T>(
    lhs: ArrayView<'_, FoR>,
    rhs: T,
    nullability: Nullability,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: IntegerPType
        + PhysicalPType
        + BitPackedIter
        + PrimInt
        + WrappingAdd
        + WrappingSub
        + AsPrimitive<T>
        + AsPrimitive<T::Physical>,
    T::Physical: BitPacking
        + BitPackingCompare
        + FastLanesComparable<Bitpacked = T::Physical>
        + NativePType
        + Zero
        + AsPrimitive<T>
        + AsPrimitive<u64>,
    u64: AsPrimitive<T::Physical>,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    T: TryFrom<PValue, Error = VortexError>,
    PValue: From<T>,
{
    let reference = match lhs.references() {
        FoRReferences::Global(reference) => reference,
        FoRReferences::Blocked(references) => {
            return compare_blocked_constant(lhs, references, rhs, nullability, operator, ctx)
                .map(Some);
        }
    };
    let reference = reference
        .as_primitive()
        .typed_value::<T>()
        .vortex_expect("FoR reference is non-null and matches the array ptype");

    if rhs < reference {
        let answer = matches!(
            operator,
            CompareOperator::NotEq | CompareOperator::Gt | CompareOperator::Gte
        );
        return constant_answer(lhs, answer, nullability).map(Some);
    }

    let distance = rhs.wrapping_sub(&reference);
    let is_equality = matches!(operator, CompareOperator::Eq | CompareOperator::NotEq);

    if is_equality || T::PTYPE.is_unsigned_int() {
        compare_unsigned_encoded(lhs, distance, nullability, operator).map(Some)
    } else {
        compare_signed_encoded(lhs, distance, nullability, operator).map(Some)
    }
}

#[inline]
fn eval_cmp<T: Ord>(lhs: T, rhs: T, operator: CompareOperator) -> bool {
    match operator {
        CompareOperator::Eq => lhs == rhs,
        CompareOperator::NotEq => lhs != rhs,
        CompareOperator::Lt => lhs < rhs,
        CompareOperator::Lte => lhs <= rhs,
        CompareOperator::Gt => lhs > rhs,
        CompareOperator::Gte => lhs >= rhs,
    }
}

/// Compares a blocked FoR array to a constant without full-array canonicalization.
///
/// For aligned `BitPacked` children, each 1024-element block adjusts `rhs` by its block reference:
/// blocks whose `[ref, ref + 2^bw - 1]` range excludes `rhs` short-circuit with zero unpacking,
/// and remaining candidate blocks use FastLanes fused `unchecked_unpack_cmp`.
fn compare_blocked_constant<T>(
    lhs: ArrayView<'_, FoR>,
    references: &ArrayRef,
    rhs: T,
    rhs_nullability: Nullability,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: IntegerPType
        + PhysicalPType
        + BitPackedIter
        + PrimInt
        + WrappingAdd
        + WrappingSub
        + AsPrimitive<T>
        + AsPrimitive<T::Physical>,
    T::Physical: BitPacking
        + BitPackingCompare
        + FastLanesComparable<Bitpacked = T::Physical>
        + NativePType
        + Zero
        + AsPrimitive<T>
        + AsPrimitive<u64>,
    u64: AsPrimitive<T::Physical>,
    u8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let len = lhs.len();
    let result_nullability = lhs.dtype().nullability() | rhs_nullability;
    let validity = lhs.validity()?.union_nullability(result_nullability);

    if len == 0 {
        return Ok(BoolArray::new(BitBuffer::empty(), validity).into_array());
    }
    if matches!(validity, Validity::AllInvalid) {
        return Ok(
            ConstantArray::new(Scalar::null(DType::Bool(result_nullability)), len).into_array(),
        );
    }

    let refs_prim = references.clone().execute::<PrimitiveArray>(ctx)?;
    let refs = refs_prim.as_slice::<T>();

    if let Some(bp) = lhs.encoded().as_opt::<BitPacked>()
        && bp.offset() == lhs.offset()
    {
        if bp.dtype().as_ptype() == T::PTYPE {
            return compare_blocked_bitpacked::<T, T>(lhs, bp, refs, rhs, validity, operator, ctx);
        }
        return match_each_unsigned_integer_ptype!(bp.dtype().as_ptype(), |U| {
            compare_blocked_bitpacked::<T, U>(lhs, bp, refs, rhs, validity, operator, ctx)
        });
    }

    let encoded = lhs.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    if encoded.ptype() == T::PTYPE {
        Ok(compare_blocked_primitive::<T, T>(
            encoded.as_slice::<T>(),
            usize::from(lhs.offset()),
            refs,
            rhs,
            validity,
            operator,
        ))
    } else {
        Ok(match_each_unsigned_integer_ptype!(encoded.ptype(), |U| {
            compare_blocked_primitive::<T, U>(
                encoded.as_slice::<U>(),
                usize::from(lhs.offset()),
                refs,
                rhs,
                validity,
                operator,
            )
        }))
    }
}

fn compare_blocked_bitpacked<T, U>(
    lhs: ArrayView<'_, FoR>,
    bp: ArrayView<'_, BitPacked>,
    refs: &[T],
    rhs: T,
    validity: Validity,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: IntegerPType
        + PhysicalPType
        + PrimInt
        + WrappingAdd
        + WrappingSub
        + AsPrimitive<T::Physical>,
    T::Physical: AsPrimitive<u64>,
    U: PhysicalPType + BitPackedIter + AsPrimitive<T>,
    U::Physical: BitPacking
        + BitPackingCompare
        + FastLanesComparable<Bitpacked = U::Physical>
        + NativePType
        + Zero
        + AsPrimitive<T>
        + AsPrimitive<u64>,
    u64: AsPrimitive<U::Physical>,
{
    let len = lhs.len();
    let offset = usize::from(lhs.offset());
    let bit_width = usize::from(bp.bit_width());
    let max_packed: u64 = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    let num_chunks = (offset + len).div_ceil(FL_CHUNK_SIZE);
    let mut words: BufferMut<u64> = BufferMut::zeroed(num_chunks * WORDS_PER_CHUNK);

    {
        let words = words.as_mut_slice();
        let mut lane_major = [0u64; WORDS_PER_CHUNK];
        let mut scratch = [U::Physical::zero(); FL_CHUNK_SIZE];
        for_each_packed_chunk::<U, _>(
            bp.packed_slice::<U::Physical>(),
            bit_width,
            offset,
            len,
            |packed_chunk, range| {
                let chunk_idx = range.start / FL_CHUNK_SIZE;
                let ref_b = refs[chunk_idx];
                let out = words[range.start / U64_BITS..]
                    .first_chunk_mut::<WORDS_PER_CHUNK>()
                    .vortex_expect("over-allocated buffer holds a full block per chunk");

                let max_nonwrap: u64 = <T::Physical as AsPrimitive<u64>>::as_(
                    T::max_value().wrapping_sub(&ref_b).as_(),
                );

                if max_packed <= max_nonwrap {
                    if rhs < ref_b {
                        let ans = matches!(
                            operator,
                            CompareOperator::NotEq | CompareOperator::Gt | CompareOperator::Gte
                        );
                        out.fill(if ans { u64::MAX } else { 0 });
                    } else {
                        let distance: u64 =
                            <T::Physical as AsPrimitive<u64>>::as_(rhs.wrapping_sub(&ref_b).as_());
                        if distance > max_packed {
                            let ans = matches!(
                                operator,
                                CompareOperator::NotEq | CompareOperator::Lt | CompareOperator::Lte
                            );
                            out.fill(if ans { u64::MAX } else { 0 });
                        } else if bit_width == 0 {
                            let ans = matches!(
                                operator,
                                CompareOperator::Eq | CompareOperator::Lte | CompareOperator::Gte
                            );
                            out.fill(if ans { u64::MAX } else { 0 });
                        } else {
                            let dist_phys: U::Physical = distance.as_();
                            unpack_cmp_block::<U::Physical>(
                                bit_width,
                                packed_chunk,
                                dist_phys,
                                operator,
                                &mut lane_major,
                                out,
                            );
                        }
                    }
                } else {
                    if bit_width == 0 {
                        scratch.fill(U::Physical::zero());
                    } else {
                        // SAFETY: `packed_chunk` holds one chunk at `bit_width` and `scratch`
                        // holds `FL_CHUNK_SIZE` values.
                        unsafe {
                            U::Physical::unchecked_unpack(bit_width, packed_chunk, &mut scratch)
                        };
                    }
                    for (word_idx, word) in out.iter_mut().enumerate() {
                        let base = word_idx * U64_BITS;
                        let mut bits_word = 0u64;
                        for bit in 0..U64_BITS {
                            let decoded = ref_b.wrapping_add(&scratch[base + bit].as_());
                            if eval_cmp(decoded, rhs, operator) {
                                bits_word |= 1u64 << bit;
                            }
                        }
                        *word = bits_word;
                    }
                }
            },
        )?;
    }

    let mut bits = BitBufferMut::from_buffer(words.into_byte_buffer(), offset, len);

    if let Some(patches) = bp.patches() {
        let p_idx = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let p_val = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let p_off = patches.offset();
        match_each_unsigned_integer_ptype!(p_idx.ptype(), |I| {
            apply_compare_patches::<T, U, I>(
                &mut bits,
                p_idx.as_slice::<I>(),
                p_val.as_slice::<U>(),
                p_off,
                offset,
                refs,
                rhs,
                operator,
            )
        });
    }

    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

#[inline]
fn unpack_cmp_block<P>(
    bit_width: usize,
    packed_chunk: &[P],
    dist: P,
    operator: CompareOperator,
    lane_major: &mut [u64; WORDS_PER_CHUNK],
    out: &mut [u64; WORDS_PER_CHUNK],
) where
    P: BitPacking + BitPackingCompare + FastLanesComparable<Bitpacked = P> + PartialOrd + Eq + Copy,
{
    let negate = match operator {
        CompareOperator::Eq => {
            // SAFETY: caller ensures `packed_chunk` holds one chunk at `bit_width`.
            unsafe {
                P::unchecked_unpack_cmp(bit_width, packed_chunk, lane_major, |a, b| a == b, dist)
            };
            false
        }
        CompareOperator::NotEq => {
            // SAFETY: caller ensures `packed_chunk` holds one chunk at `bit_width`.
            unsafe {
                P::unchecked_unpack_cmp(bit_width, packed_chunk, lane_major, |a, b| a == b, dist)
            };
            true
        }
        CompareOperator::Lt => {
            // SAFETY: caller ensures `packed_chunk` holds one chunk at `bit_width`.
            unsafe {
                P::unchecked_unpack_cmp(bit_width, packed_chunk, lane_major, |a, b| a < b, dist)
            };
            false
        }
        CompareOperator::Gte => {
            // SAFETY: caller ensures `packed_chunk` holds one chunk at `bit_width`.
            unsafe {
                P::unchecked_unpack_cmp(bit_width, packed_chunk, lane_major, |a, b| a < b, dist)
            };
            true
        }
        CompareOperator::Gt => {
            // SAFETY: caller ensures `packed_chunk` holds one chunk at `bit_width`.
            unsafe {
                P::unchecked_unpack_cmp(bit_width, packed_chunk, lane_major, |a, b| a > b, dist)
            };
            false
        }
        CompareOperator::Lte => {
            // SAFETY: caller ensures `packed_chunk` holds one chunk at `bit_width`.
            unsafe {
                P::unchecked_unpack_cmp(bit_width, packed_chunk, lane_major, |a, b| a > b, dist)
            };
            true
        }
    };
    transpose_bits::<P>(lane_major, out);
    if negate {
        out.iter_mut().for_each(|w| *w = !*w);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "typed blocked compare patch fixup parameters"
)]
fn apply_compare_patches<T, U, I>(
    bits: &mut BitBufferMut,
    indices: &[I],
    values: &[U],
    patch_offset: usize,
    for_offset: usize,
    refs: &[T],
    rhs: T,
    operator: CompareOperator,
) where
    T: NativePType + WrappingAdd + Ord,
    U: Copy + AsPrimitive<T>,
    I: AsPrimitive<usize>,
{
    let bit_len = bits.len();
    for (&idx, &patch_val) in indices.iter().zip(values) {
        let Some(pos) = idx.as_().checked_sub(patch_offset) else {
            continue;
        };
        if pos >= bit_len {
            continue;
        }
        let ref_b = refs[(for_offset + pos) / FL_CHUNK_SIZE];
        let decoded = ref_b.wrapping_add(&patch_val.as_());
        bits.set_to(pos, eval_cmp(decoded, rhs, operator));
    }
}

fn compare_blocked_primitive<T, U>(
    encoded: &[U],
    offset: usize,
    refs: &[T],
    rhs: T,
    validity: Validity,
    operator: CompareOperator,
) -> ArrayRef
where
    T: NativePType + WrappingAdd + Ord,
    U: Copy + AsPrimitive<T>,
{
    let len = encoded.len();
    let mut bits = BitBufferMut::with_capacity(len);
    let first_len = (FL_CHUNK_SIZE - offset).min(len);
    let (first, rest) = encoded.split_at(first_len);
    for (chunk, &ref_b) in iter::once(first)
        .chain(rest.chunks(FL_CHUNK_SIZE))
        .zip(refs)
    {
        for &val in chunk {
            let decoded = ref_b.wrapping_add(&val.as_());
            bits.append(eval_cmp(decoded, rhs, operator));
        }
    }
    BoolArray::new(bits.freeze(), validity).into_array()
}

/// Compares the encoded values to `distance` with the operator unchanged.
///
/// This is correct for equality on any integer type, and for ordering on unsigned types, where
/// the encoded values and `distance` order as unsigned distances from the reference.
fn compare_unsigned_encoded<T: IntegerPType + Into<PValue>>(
    lhs: ArrayView<'_, FoR>,
    distance: T,
    nullability: Nullability,
    operator: CompareOperator,
) -> VortexResult<ArrayRef> {
    lhs.encoded().compare(
        ConstantArray::new(Scalar::primitive(distance, nullability), lhs.len()).into_array(),
        operator,
    )
}

/// Orders signed encoded values as unsigned distances.
///
/// Distances above `T::MAX` read as negative, so each operator is one signed range or its
/// complement: `[0, distance)` for a non-negative `distance`, `[distance, -1]` otherwise.
fn compare_signed_encoded<T>(
    lhs: ArrayView<'_, FoR>,
    distance: T,
    nullability: Nullability,
    operator: CompareOperator,
) -> VortexResult<ArrayRef>
where
    T: IntegerPType + Into<PValue>,
{
    let strictness = |strict: bool| {
        if strict {
            StrictComparison::Strict
        } else {
            StrictComparison::NonStrict
        }
    };

    let below = matches!(operator, CompareOperator::Lt | CompareOperator::Lte);
    let (lower, upper, options, negate) = if distance >= T::zero() {
        let upper_strict = matches!(operator, CompareOperator::Lt | CompareOperator::Gte);
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: strictness(upper_strict),
        };
        (T::zero(), distance, options, !below)
    } else {
        let lower_strict = matches!(operator, CompareOperator::Lte | CompareOperator::Gt);
        let options = BetweenOptions {
            lower_strict: strictness(lower_strict),
            upper_strict: StrictComparison::NonStrict,
        };
        (distance, !T::zero(), options, below)
    };

    let bound = |value: T| {
        ConstantArray::new(Scalar::primitive(value, nullability), lhs.len()).into_array()
    };
    let in_range = lhs
        .encoded()
        .clone()
        .between(bound(lower), bound(upper), options)?;

    if negate { in_range.not() } else { Ok(in_range) }
}

/// A constant answer for every row that still carries the array's validity.
fn constant_answer(
    lhs: ArrayView<'_, FoR>,
    value: bool,
    rhs_nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let nullability = lhs.dtype().nullability() | rhs_nullability;

    Ok(match lhs.validity()?.union_nullability(nullability) {
        Validity::NonNullable | Validity::AllValid => {
            ConstantArray::new(Scalar::bool(value, nullability), len).into_array()
        }
        Validity::AllInvalid => {
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array()
        }
        Validity::Array(validity) => ConstantArray::new(Scalar::bool(value, nullability), len)
            .into_array()
            .mask(validity)?,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedData;
    use crate::FoR;
    use crate::FoRArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn for_arr(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoR::try_new(encoded, reference).vortex_expect("FoR array construction should succeed")
    }

    #[test]
    fn test_compare_constant() {
        let mut ctx = SESSION.create_execution_ctx();
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::AllValid).into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            30i32,
            Nullability::NonNullable,
            CompareOperator::Eq,
            &mut ctx,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, true, false].map(Some)),
            &mut ctx
        );

        let result = compare_constant(
            lhs.as_view(),
            12i32,
            Nullability::NonNullable,
            CompareOperator::NotEq,
            &mut ctx,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, true, false].map(Some)),
            &mut ctx
        );

        for (op, expected) in [
            (CompareOperator::Lt, [true, false, true]),
            (CompareOperator::Lte, [true, true, true]),
            (CompareOperator::Gt, [false, false, false]),
            (CompareOperator::Gte, [false, true, false]),
        ] {
            let result =
                compare_constant(lhs.as_view(), 30i32, Nullability::NonNullable, op, &mut ctx)
                    .unwrap()
                    .unwrap();
            assert_arrays_eq!(result, BoolArray::from_iter(expected.map(Some)), &mut ctx);
        }
    }

    #[test]
    fn test_compare_nullable_constant() {
        let mut ctx = SESSION.create_execution_ctx();
        let reference = Scalar::from(0);
        // 10, 30, 12
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::NonNullable).into_array(),
            reference,
        );

        assert_eq!(
            compare_constant(
                lhs.as_view(),
                30i32,
                Nullability::Nullable,
                CompareOperator::Eq,
                &mut ctx,
            )
            .unwrap()
            .unwrap()
            .dtype(),
            &DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            compare_constant(
                lhs.as_view(),
                30i32,
                Nullability::NonNullable,
                CompareOperator::Eq,
                &mut ctx,
            )
            .unwrap()
            .unwrap()
            .dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
    }

    /// A constant below the reference answers every row, and a nullable array keeps the result
    /// nullable even when the constant is not.
    #[rstest]
    #[case::all_valid(Validity::AllValid, [Some(true); 3])]
    #[case::all_invalid(Validity::AllInvalid, [None; 3])]
    #[case::array(Validity::from_iter([true, false, true]), [Some(true), None, Some(true)])]
    fn compare_below_reference_nullable_array(
        #[case] validity: Validity,
        #[case] expected: [Option<bool>; 3],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 20, 2), validity).into_array(),
            Scalar::from(10),
        );

        let result = compare_constant(
            lhs.as_view(),
            5i32,
            Nullability::NonNullable,
            CompareOperator::Gt,
            &mut ctx,
        )?
        .vortex_expect("constant below the reference has an answer");

        assert_eq!(result.dtype(), &DType::Bool(Nullability::Nullable));
        assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);

        Ok(())
    }

    #[test]
    fn compare_non_encodable_constant() {
        let mut ctx = SESSION.create_execution_ctx();
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 10, 1), Validity::AllValid).into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            -1i32,
            Nullability::NonNullable,
            CompareOperator::Eq,
            &mut ctx,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, false].map(Some)),
            &mut ctx
        );

        let result = compare_constant(
            lhs.as_view(),
            -1i32,
            Nullability::NonNullable,
            CompareOperator::NotEq,
            &mut ctx,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, true, true].map(Some)),
            &mut ctx
        );
    }

    #[test]
    fn compare_large_constant() {
        let mut ctx = SESSION.create_execution_ctx();
        let reference = Scalar::from(-9219218377546224477i64);
        let lhs = for_arr(
            PrimitiveArray::new(
                buffer![0i64, 9654309310445864926u64 as i64],
                Validity::AllValid,
            )
            .into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            435090932899640449i64,
            Nullability::Nullable,
            CompareOperator::Eq,
            &mut ctx,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(false), Some(true)]),
            &mut ctx
        );

        let result = compare_constant(
            lhs.as_view(),
            435090932899640449i64,
            Nullability::Nullable,
            CompareOperator::NotEq,
            &mut ctx,
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), Some(false)]),
            &mut ctx
        );
    }

    const ALL_OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    /// Encodes the values against their minimum.
    fn encode_with_min<T>(
        values: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(T, PrimitiveArray)>
    where
        T: NativePType + WrappingSub,
        T: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    {
        let reference = values
            .clone()
            .into_array()
            .statistics()
            .compute_min::<T>(ctx)
            .vortex_expect("min");
        let encoded = PrimitiveArray::new(
            values
                .as_slice::<T>()
                .iter()
                .map(|v| v.wrapping_sub(&reference))
                .collect::<vortex_buffer::Buffer<T>>(),
            values.validity()?,
        );
        Ok((reference, encoded))
    }

    /// Asserts the kernel engages and matches the primitive comparison.
    fn assert_matches_primitive<T: Into<Scalar>>(
        array: &FoRArray,
        values: &PrimitiveArray,
        constants: impl IntoIterator<Item = T>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for constant in constants {
            let rhs = ConstantArray::new(constant, values.len()).into_array();
            for operator in ALL_OPERATORS {
                let actual = <FoR as CompareKernel>::compare(array.as_view(), &rhs, operator, ctx)?
                    .unwrap_or_else(|| panic!("FoR compare must engage for {operator:?}"));
                let expected = values.clone().into_array().compare(rhs.clone(), operator)?;
                assert_arrays_eq!(actual, expected, ctx);
            }
        }
        Ok(())
    }

    /// Asserts FoR over a bit-packed child matches the primitive comparison.
    fn assert_bit_packed_matches_primitive<T>(
        values: PrimitiveArray,
        bit_width: u8,
        constants: impl IntoIterator<Item = T>,
    ) -> VortexResult<()>
    where
        T: NativePType + WrappingSub + Into<PValue> + Into<Scalar>,
        T: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let (reference, encoded) = encode_with_min::<T>(&values, &mut ctx)?;
        let packed = BitPackedData::encode(&encoded.into_array(), bit_width, &mut ctx)?;
        let array = FoR::try_new(packed.into_array(), reference.into())?;

        assert_matches_primitive(&array, &values, constants, &mut ctx)
    }

    /// Distances above `i8::MAX` read as negative encoded values.
    #[test]
    fn signed_distances_above_max_match_primitive() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter(
            (i8::MIN..=i8::MAX).map(|v| (v % 7 != 3).then_some(v)),
        );
        let (reference, encoded) = encode_with_min::<i8>(&values, &mut ctx)?;
        let array = FoR::try_new(encoded.into_array(), reference.into())?;

        assert_matches_primitive(&array, &values, i8::MIN..=i8::MAX, &mut ctx)
    }

    #[test]
    fn signed_bit_packed_matches_primitive() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter((0..2000i32).map(|i| {
            if i % 97 == 0 {
                -1_000 + i * 50
            } else {
                -1_000 + i % 40
            }
        }));
        let constants = [
            i32::MIN,
            -1_001,
            -1_000,
            -999,
            -980,
            0,
            98_900,
            98_901,
            i32::MAX,
        ];
        assert_bit_packed_matches_primitive(values, 6, constants)
    }

    #[test]
    fn unsigned_bit_packed_matches_primitive() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter((0..1500u16).map(|i| 1_000 + i % 70));
        let constants = [0u16, 999, 1_000, 1_001, 1_035, 1_069, 1_070, u16::MAX];
        assert_bit_packed_matches_primitive(values, 7, constants)
    }

    #[test]
    fn nullable_bit_packed_matches_primitive() -> VortexResult<()> {
        let values = PrimitiveArray::from_option_iter(
            (0..1200i64).map(|i| (i % 5 != 0).then_some(-50 + i % 30)),
        );
        let constants = [i64::MIN, -51, -50, -40, -21, -20, i64::MAX];
        assert_bit_packed_matches_primitive(values, 5, constants)
    }

    #[test]
    fn blocked_signed_bitpacked_boundaries_and_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 3 chunks with negative, boundary, and extreme refs + outlier patches.
        let values = PrimitiveArray::from_option_iter((0..3000usize).map(|i| {
            if i % 19 == 0 {
                return None;
            }
            let val = match i / 1024 {
                0 => {
                    if i % 97 == 0 {
                        i64::MIN + 5_000
                    } else {
                        i64::MIN + (i % 30) as i64
                    }
                }
                1 => {
                    if i % 83 == 0 {
                        -10 + 2_000
                    } else {
                        -10 + (i % 25) as i64
                    }
                }
                _ => i64::MAX - 40 + (i % 41) as i64,
            };
            Some(val)
        }));
        let for_arr = FoR::encode_chunked(values.clone(), &mut ctx)?;
        let bp = BitPackedData::encode(for_arr.encoded(), 5, &mut ctx)?;
        let refs = for_arr.blocked_references().unwrap().clone();
        let blocked = FoR::try_new_chunked(bp.into_array(), refs, 0)?;

        let constants = [
            i64::MIN,
            i64::MIN + 15,
            i64::MIN + 29,
            i64::MIN + 30,
            i64::MIN + 5_000,
            -11,
            -10,
            0,
            14,
            1_990,
            i64::MAX - 41,
            i64::MAX - 20,
            i64::MAX,
        ];
        assert_matches_primitive(&blocked, &values, constants, &mut ctx)?;

        // Also test non-zero offset slice across chunk boundaries (unpatched narrow u16).
        let unpatched_raw = PrimitiveArray::from_iter((0..3500i64).map(|i| match i / 1024 {
            0 => -1_000_000 + (i % 100),
            1 => i % 100,
            2 => 5_000_000 + (i % 100),
            _ => 10_000_000 + (i % 100),
        }));
        let unpatched_for = FoR::encode_chunked(unpatched_raw.clone(), &mut ctx)?;
        let unpatched_bp = BitPackedData::encode(unpatched_for.encoded(), 7, &mut ctx)?;
        let unpatched_blocked = FoR::try_new_chunked(
            unpatched_bp.into_array(),
            unpatched_for.blocked_references().unwrap().clone(),
            0,
        )?;
        let sliced = unpatched_blocked
            .into_array()
            .slice(300..2600)?
            .optimize_ctx(&SESSION)?;
        let sliced_expected = unpatched_raw
            .into_array()
            .slice(300..2600)?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_matches_primitive(
            &sliced.as_::<FoR>().into_owned(),
            &sliced_expected,
            [
                -1_000_001i64,
                -1_000_000,
                -999_950,
                0,
                50,
                5_000_050,
                10_000_000,
            ],
            &mut ctx,
        )
    }

    #[test]
    fn blocked_unsigned_and_bw0_and_narrowed() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Bit-width 0 chunks (constant per chunk, varying across chunks) + narrowed u8.
        let values = PrimitiveArray::from_iter((0..3000u64).map(|i| match i / 1024 {
            0 => 10u64,
            1 => 10_000_000,
            _ => u64::MAX,
        }));
        let for_arr = FoR::encode_chunked(values.clone(), &mut ctx)?;
        let bp = BitPackedData::encode(for_arr.encoded(), 0, &mut ctx)?;
        let blocked = FoR::try_new_chunked(
            bp.into_array(),
            for_arr.blocked_references().unwrap().clone(),
            0,
        )?;
        assert_matches_primitive(
            &blocked,
            &values,
            [0u64, 9, 10, 11, 10_000_000, u64::MAX - 1, u64::MAX],
            &mut ctx,
        )?;

        // Primitive (non-BitPacked) blocked child with wrapping reference.
        let raw_prim = PrimitiveArray::from_iter((0..2500i16).map(|i| i % 7));
        let refs = PrimitiveArray::from_iter([i16::MIN, -3, i16::MAX]).into_array();
        let wrapping = FoR::try_new_chunked(raw_prim.into_array(), refs, 0)?;
        let expected = wrapping
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_matches_primitive(
            &wrapping,
            &expected,
            [i16::MIN, i16::MIN + 3, -3, 0, 3, i16::MAX],
            &mut ctx,
        )
    }
}
