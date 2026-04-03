// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;

impl CompareKernel for BitPacked {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        if !constant.dtype().is_int() {
            return Ok(None);
        }

        match_each_integer_ptype!(lhs.dtype().as_ptype(), |T| {
            let value = T::try_from(&constant)?;
            compare_constant::<T>(lhs, value, rhs.dtype().nullability(), operator, ctx).map(Some)
        })
    }
}

impl BetweenKernel for BitPacked {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        if !lower.dtype().is_int() || !upper.dtype().is_int() {
            return Ok(None);
        }

        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            let lower = T::try_from(&lower)?;
            let upper = T::try_from(&upper)?;
            between_constant::<T>(array, lower, upper, nullability, options, ctx).map(Some)
        })
    }
}

fn compare_constant<T>(
    array: ArrayView<'_, BitPacked>,
    value: T,
    nullability: Nullability,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + FastLanesComparable,
    <T as FastLanesComparable>::Bitpacked: UnsignedPType + BitPacking + BitPackingCompare,
{
    compare_constant_typed::<<T as FastLanesComparable>::Bitpacked, T>(
        array,
        value,
        nullability,
        operator,
        ctx,
    )
}

fn compare_constant_typed<U, T>(
    array: ArrayView<'_, BitPacked>,
    value: T,
    nullability: Nullability,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + FastLanesComparable<Bitpacked = U>,
    U: UnsignedPType + BitPacking + BitPackingCompare,
{
    match operator {
        CompareOperator::Eq => {
            compare_constant_with::<U, T, _>(array, value, nullability, ctx, T::is_eq)
        }
        CompareOperator::NotEq => {
            compare_constant_with::<U, T, _>(array, value, nullability, ctx, is_ne::<T>)
        }
        CompareOperator::Gt => {
            compare_constant_with::<U, T, _>(array, value, nullability, ctx, T::is_gt)
        }
        CompareOperator::Gte => {
            compare_constant_with::<U, T, _>(array, value, nullability, ctx, T::is_ge)
        }
        CompareOperator::Lt => {
            compare_constant_with::<U, T, _>(array, value, nullability, ctx, T::is_lt)
        }
        CompareOperator::Lte => {
            compare_constant_with::<U, T, _>(array, value, nullability, ctx, T::is_le)
        }
    }
}

fn compare_constant_with<U, T, C>(
    array: ArrayView<'_, BitPacked>,
    value: T,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
    compare: C,
) -> VortexResult<ArrayRef>
where
    T: NativePType + FastLanesComparable<Bitpacked = U>,
    U: UnsignedPType + BitPacking + BitPackingCompare,
    C: Fn(T, T) -> bool + Copy,
{
    let mut bits = collect_chunk_masks::<U>(
        array.data(),
        array.len(),
        array.offset(),
        |bit_width, packed_chunk, chunk_matches| unsafe {
            U::unchecked_unpack_cmp(bit_width, packed_chunk, chunk_matches, compare, value);
        },
    );

    if let Some(patches) = array.patches() {
        apply_patch_predicate::<T>(&mut bits, &patches, ctx, |patched| compare(patched, value))?;
    }

    Ok(BoolArray::new(
        bits.freeze(),
        array.validity()?.union_nullability(nullability),
    )
    .into_array())
}

fn between_constant<T>(
    array: ArrayView<'_, BitPacked>,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + FastLanesComparable,
    <T as FastLanesComparable>::Bitpacked: UnsignedPType + BitPacking + BitPackingCompare,
{
    between_constant_typed::<<T as FastLanesComparable>::Bitpacked, T>(
        array,
        lower,
        upper,
        nullability,
        options,
        ctx,
    )
}

fn between_constant_typed<U, T>(
    array: ArrayView<'_, BitPacked>,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + FastLanesComparable<Bitpacked = U>,
    U: UnsignedPType + BitPacking + BitPackingCompare,
{
    let mut bits = match (options.lower_strict, options.upper_strict) {
        (StrictComparison::Strict, StrictComparison::Strict) => {
            collect_between_masks::<U, T, _, _>(
                array.data(),
                array.len(),
                array.offset(),
                lower,
                upper,
                NativePType::is_lt,
                NativePType::is_lt,
            )
        }
        (StrictComparison::Strict, StrictComparison::NonStrict) => {
            collect_between_masks::<U, T, _, _>(
                array.data(),
                array.len(),
                array.offset(),
                lower,
                upper,
                NativePType::is_lt,
                NativePType::is_le,
            )
        }
        (StrictComparison::NonStrict, StrictComparison::Strict) => {
            collect_between_masks::<U, T, _, _>(
                array.data(),
                array.len(),
                array.offset(),
                lower,
                upper,
                NativePType::is_le,
                NativePType::is_lt,
            )
        }
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => {
            collect_between_masks::<U, T, _, _>(
                array.data(),
                array.len(),
                array.offset(),
                lower,
                upper,
                NativePType::is_le,
                NativePType::is_le,
            )
        }
    };

    if let Some(patches) = array.patches() {
        apply_patch_predicate::<T>(&mut bits, &patches, ctx, |patched| {
            lower_matches_bound(lower, patched, options.lower_strict)
                && upper_matches_bound(patched, upper, options.upper_strict)
        })?;
    }

    Ok(BoolArray::new(
        bits.freeze(),
        array.validity()?.union_nullability(nullability),
    )
    .into_array())
}

fn collect_between_masks<U, T, LF, UF>(
    array: &BitPackedData,
    len: usize,
    offset: u16,
    lower: T,
    upper: T,
    lower_matches: LF,
    upper_matches: UF,
) -> BitBufferMut
where
    T: NativePType + FastLanesComparable<Bitpacked = U>,
    U: UnsignedPType + BitPacking,
    LF: Fn(T, T) -> bool + Copy,
    UF: Fn(T, T) -> bool + Copy,
{
    collect_unpacked_chunk_masks::<U>(array, len, offset, |unpacked, chunk_matches| {
        fill_between_chunk::<U, T, LF, UF>(
            unpacked,
            chunk_matches,
            lower,
            upper,
            lower_matches,
            upper_matches,
        );
    })
}

fn collect_chunk_masks<U>(
    array: &BitPackedData,
    len: usize,
    offset: u16,
    mut fill_chunk: impl FnMut(usize, &[U], &mut [u64; 16]),
) -> BitBufferMut
where
    U: UnsignedPType + BitPacking,
{
    if len == 0 {
        return BitBufferMut::empty();
    }

    let bit_width = array.bit_width() as usize;
    let packed = array.packed_slice::<U>();
    let elems_per_chunk = 128 * bit_width / size_of::<U>();
    let num_chunks = (offset as usize + len).div_ceil(1024);
    let mut output = BufferMut::<u64>::with_capacity(num_chunks * 16);

    for chunk_idx in 0..num_chunks {
        let packed_chunk = &packed[chunk_idx * elems_per_chunk..][..elems_per_chunk];
        append_chunk_matches(&mut output, |chunk_matches| {
            fill_chunk(bit_width, packed_chunk, chunk_matches);
        });
    }

    let total_len = num_chunks * 1024;
    let mut output = BitBufferMut::from_buffer(output.into_byte_buffer(), 0, total_len);

    if offset == 0 {
        output.truncate(len);
        return output;
    }

    BitBufferMut::copy_from(
        &output
            .freeze()
            .slice(offset as usize..offset as usize + len),
    )
}

fn collect_unpacked_chunk_masks<U>(
    array: &BitPackedData,
    len: usize,
    offset: u16,
    mut fill_chunk: impl FnMut(&[U; 1024], &mut [u64; 16]),
) -> BitBufferMut
where
    U: UnsignedPType + BitPacking,
{
    if len == 0 {
        return BitBufferMut::empty();
    }

    let bit_width = array.bit_width() as usize;
    let packed = array.packed_slice::<U>();
    let elems_per_chunk = 128 * bit_width / size_of::<U>();
    let num_chunks = (offset as usize + len).div_ceil(1024);
    let mut output = BufferMut::<u64>::with_capacity(num_chunks * 16);
    let mut unpacked = [U::default(); 1024];

    for chunk_idx in 0..num_chunks {
        let packed_chunk = &packed[chunk_idx * elems_per_chunk..][..elems_per_chunk];

        unsafe {
            U::unchecked_unpack(bit_width, packed_chunk, &mut unpacked);
        }

        append_chunk_matches(&mut output, |chunk_matches| {
            fill_chunk(&unpacked, chunk_matches);
        });
    }

    let total_len = num_chunks * 1024;
    let mut output = BitBufferMut::from_buffer(output.into_byte_buffer(), 0, total_len);

    if offset == 0 {
        output.truncate(len);
        return output;
    }

    BitBufferMut::copy_from(
        &output
            .freeze()
            .slice(offset as usize..offset as usize + len),
    )
}

#[inline]
fn append_chunk_matches(output: &mut BufferMut<u64>, fill_chunk: impl FnOnce(&mut [u64; 16])) {
    let base_len = output.len();

    let spare = output.spare_capacity_mut();
    debug_assert!(spare.len() >= 16);
    let chunk_matches = unsafe { &mut *(spare.as_mut_ptr().cast::<[u64; 16]>()) };

    fill_chunk(chunk_matches);

    // SAFETY: `fill_chunk` initializes all 16 words before we expose them via `set_len`.
    unsafe {
        output.set_len(base_len + 16);
    }
}

#[inline]
fn fill_between_chunk<U, T, LF, UF>(
    unpacked: &[U; 1024],
    chunk_matches: &mut [u64; 16],
    lower: T,
    upper: T,
    lower_matches: LF,
    upper_matches: UF,
) where
    T: NativePType + FastLanesComparable<Bitpacked = U>,
    U: UnsignedPType,
    LF: Fn(T, T) -> bool,
    UF: Fn(T, T) -> bool,
{
    for (word_idx, word) in chunk_matches.iter_mut().enumerate() {
        let start = word_idx * 64;
        let mut mask = 0u64;

        for bit_idx in 0..64 {
            let value = T::as_unpacked(unpacked[start + bit_idx]);
            mask |=
                u64::from(lower_matches(lower, value) && upper_matches(value, upper)) << bit_idx;
        }

        *word = mask;
    }
}

fn apply_patch_predicate<T>(
    bits: &mut BitBufferMut,
    patches: &Patches,
    ctx: &mut ExecutionCtx,
    mut predicate: impl FnMut(T) -> bool,
) -> VortexResult<()>
where
    T: NativePType,
{
    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let values = values.as_slice::<T>();
    let offset = patches.offset();

    match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
        for (&index, &value) in indices.as_slice::<I>().iter().zip(values) {
            let absolute_index: usize = index.as_();
            if absolute_index < offset {
                continue;
            }

            let bit_index = absolute_index - offset;
            if bit_index >= bits.len() {
                break;
            }

            bits.set_to(bit_index, predicate(value));
        }
        Ok(())
    })
}

#[inline]
fn is_ne<T: NativePType>(lhs: T, rhs: T) -> bool {
    !lhs.is_eq(rhs)
}

#[inline]
fn lower_matches_bound<T: NativePType>(lower: T, value: T, strict: StrictComparison) -> bool {
    match strict {
        StrictComparison::Strict => lower.is_lt(value),
        StrictComparison::NonStrict => lower.is_le(value),
    }
}

#[inline]
fn upper_matches_bound<T: NativePType>(value: T, upper: T, strict: StrictComparison) -> bool {
    match strict {
        StrictComparison::Strict => value.is_lt(upper),
        StrictComparison::NonStrict => value.is_le(upper),
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::validity::Validity;

    use crate::BitPacked;
    use crate::BitPackedArrayExt;
    use crate::bitpack_compress::bitpack_encode;

    fn bp(array: &PrimitiveArray, bit_width: u8) -> crate::BitPackedArray {
        bitpack_encode(
            array,
            bit_width,
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
    }

    #[test]
    fn compare_unsigned_constant() {
        let array = bp(&PrimitiveArray::from_iter([1u32, 2, 3, 4, 5]), 3);
        let rhs = ConstantArray::new(3u32, array.len()).into_array();

        let result = <BitPacked as CompareKernel>::compare(
            array.as_view(),
            &rhs,
            CompareOperator::Gt,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, false, true, true])
        );
    }

    #[test]
    fn compare_signed_constant() {
        let array = bp(&PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]), 3);
        let rhs = ConstantArray::new(2i32, array.len()).into_array();

        let result = <BitPacked as CompareKernel>::compare(
            array.as_view(),
            &rhs,
            CompareOperator::Gte,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, true, true, true, true])
        );
    }

    #[test]
    fn compare_with_patches() {
        let array = bp(&PrimitiveArray::from_iter(0u32..257), 8);
        assert!(array.patches().is_some());

        let rhs = ConstantArray::new(256u32, array.len()).into_array();
        let result = <BitPacked as CompareKernel>::compare(
            array.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();

        assert_arrays_eq!(
            result,
            BoolArray::from_indices(array.len(), [256usize], Validity::NonNullable,)
        );
    }

    #[test]
    fn compare_nullable() {
        let array = bp(
            &PrimitiveArray::from_option_iter([Some(1u16), None, Some(3), Some(4), None]),
            3,
        );
        let rhs = ConstantArray::new(3u16, array.len()).into_array();

        let result = <BitPacked as CompareKernel>::compare(
            array.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(false), None, Some(true), Some(false), None])
        );
    }

    #[test]
    fn binary_compare_pushdown_executes() {
        let array = bp(&PrimitiveArray::from_iter([1u32, 2, 3, 4, 5]), 3).into_array();
        let rhs = ConstantArray::new(4u32, array.len()).into_array();

        let result = array
            .binary(rhs, CompareOperator::Lt.into())
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .into_array();

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, true, true, false, false])
        );
    }

    #[test]
    fn between_executes_in_encoded_space() {
        let array = bp(&PrimitiveArray::from_iter(0u32..257), 8).into_array();
        let len = array.len();

        let result = array
            .between(
                ConstantArray::new(255u32, len).into_array(),
                ConstantArray::new(256u32, len).into_array(),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::NonStrict,
                },
            )
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .into_array();

        assert_arrays_eq!(
            result,
            BoolArray::from_indices(len, [255usize, 256], Validity::NonNullable,)
        );
    }

    #[test]
    fn between_strict_upper() {
        let array = bp(&PrimitiveArray::from_iter([10i32, 11, 12, 13]), 4).into_array();
        let len = array.len();

        let result = array
            .between(
                ConstantArray::new(10i32, len).into_array(),
                ConstantArray::new(12i32, len).into_array(),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                },
            )
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .into_array();

        assert_arrays_eq!(result, BoolArray::from_iter([true, true, false, false]));
    }
}
