// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::SumV2;
use super::sum_v2_partial_fields;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupRanges;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::fns::sum::sum_float_all;
use crate::aggregate_fn::fns::sum::sum_signed_all;
use crate::aggregate_fn::fns::sum::sum_unsigned_all;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFn;
use crate::arrays::StructArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::matcher::Matcher;
use crate::scalar_fn::fns::cast::Cast;
use crate::validity::Validity;

/// Encoding-specific grouped [`SumV2`] kernel for primitive element arrays.
#[derive(Debug)]
pub(crate) struct PrimitiveGroupedSumV2EncodingKernel;

impl DynGroupedAggregateKernel for PrimitiveGroupedSumV2EncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(options) = aggregate_fn.as_opt::<SumV2>() else {
            return Ok(None);
        };
        try_grouped_sum(groups, ctx, options.skip_nans)
    }
}

/// Encoding-specific grouped [`SumV2`] kernel for boolean element arrays.
///
/// The sum of a group of booleans is the number of valid `true` values, so each group is a
/// popcount over its range of bits.
#[derive(Debug)]
pub(crate) struct BoolGroupedSumV2EncodingKernel;

impl DynGroupedAggregateKernel for BoolGroupedSumV2EncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if aggregate_fn.as_opt::<SumV2>().is_none() || !groups.elements().is::<Bool>() {
            return Ok(None);
        }
        let elements = groups.elements().clone().downcast::<Bool>();
        let group_ranges = groups.group_ranges(ctx)?;
        let group_validity = groups.group_validity(ctx)?;
        let elem_mask = elements
            .as_ref()
            .validity()?
            .execute_mask(elements.as_ref().len(), ctx)?;

        // Only valid `true` values count, and a group with no valid values sums to null.
        let bits = elements.to_bit_buffer();
        let (valid_true, valid) = match elem_mask.bit_buffer() {
            AllOr::All => (bits, None),
            AllOr::None => (
                BitBuffer::new_unset(bits.len()),
                Some(BitBuffer::new_unset(bits.len())),
            ),
            AllOr::Some(validity) => (&bits & validity, Some(validity.clone())),
        };

        let group_count = group_ranges.len();
        let mut is_empty = BitBufferMut::new_unset(group_count);
        let sums = group_ranges.iter().enumerate().map(|(i, (offset, size))| {
            if !group_validity.value(i) {
                return 0u64;
            }
            let any_valid = match &valid {
                None => size > 0,
                Some(valid) => valid.slice(offset..offset + size).true_count() > 0,
            };
            if !any_valid {
                // SAFETY: `i` comes from enumerating `group_ranges`, and the bitmap has one bit
                // per group.
                unsafe { is_empty.set_unchecked(i) };
            }
            valid_true.slice(offset..offset + size).true_count() as u64
        });
        let sums = PrimitiveArray::from_iter(sums);
        let partial_fields = sum_v2_partial_fields(sums.dtype().clone());

        // SAFETY: all three children have one value per group and match `partial_fields`; the
        // struct validity is derived from the same group count.
        Ok(Some(
            unsafe {
                StructArray::new_unchecked(
                    vec![
                        sums.into_array(),
                        BoolArray::new(BitBuffer::new_unset(group_count), Validity::NonNullable)
                            .into_array(),
                        BoolArray::new(is_empty.freeze(), Validity::NonNullable).into_array(),
                    ],
                    partial_fields,
                    group_count,
                    Validity::from_mask(group_validity, Nullability::Nullable),
                )
            }
            .into_array(),
        ))
    }
}

/// Encoding-specific grouped [`SumV2`] kernel for integers cast to `f64`.
///
/// Summing `cast(ints as f64)` converts every element and then sums the floats in order, which is
/// how engines such as DataFusion sum integer lists. This kernel performs the same per-element
/// conversion inside the summation loop instead of first materializing the cast elements.
#[derive(Debug)]
pub(crate) struct CastGroupedSumV2EncodingKernel;

impl DynGroupedAggregateKernel for CastGroupedSumV2EncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if aggregate_fn.as_opt::<SumV2>().is_none() {
            return Ok(None);
        }
        let Some(cast) = ExactScalarFn::<Cast>::try_match(groups.elements()) else {
            return Ok(None);
        };
        if !matches!(cast.options, DType::Primitive(PType::F64, _)) {
            return Ok(None);
        }
        let Some(scalar_fn) = groups.elements().as_opt::<ScalarFn>() else {
            return Ok(None);
        };
        let input = scalar_fn.child_at(0);
        if !matches!(input.dtype(), DType::Primitive(ptype, _) if ptype.is_int()) {
            return Ok(None);
        }

        let group_ranges = groups.group_ranges(ctx)?;
        let group_validity = groups.group_validity(ctx)?;
        if !input.is::<Primitive>()
            && let Some(sums) = chunked_grouped_sum(
                input,
                &group_ranges,
                &group_validity,
                SumMode::CastToF64,
                ctx,
            )?
        {
            return Ok(Some(sums));
        }

        let input = input.clone().execute::<PrimitiveArray>(ctx)?;
        let elem_mask = input
            .as_ref()
            .validity()?
            .execute_mask(input.as_ref().len(), ctx)?;
        let all_valid = elem_mask.all_true();

        // An integer always converts to `f64`, so the sum never overflows.
        let (sums, is_overflow, is_empty) = match_each_integer_ptype!(input.ptype(), |T| {
            collect_sums::<T, f64>(
                input.as_slice::<T>(),
                &group_ranges,
                &group_validity,
                &elem_mask,
                all_valid,
                |acc, slice| {
                    sum_float_all(acc, slice, false);
                    false
                },
            )
        });
        let partial_fields = sum_v2_partial_fields(sums.dtype().clone());

        // SAFETY: all three children have one value per group and match `partial_fields`; the
        // struct validity is derived from the same group count.
        Ok(Some(
            unsafe {
                StructArray::new_unchecked(
                    vec![
                        sums.into_array(),
                        BoolArray::new(is_overflow, Validity::NonNullable).into_array(),
                        BoolArray::new(is_empty, Validity::NonNullable).into_array(),
                    ],
                    partial_fields,
                    group_validity.len(),
                    Validity::from_mask(group_validity, Nullability::Nullable),
                )
            }
            .into_array(),
        ))
    }
}

/// Grouped [`SumV2`] kernel for encoded primitive elements whose groups are laid out in order.
///
/// Executing the whole element array first decodes it into one large buffer, which for lightly
/// compressed data such as bit-packed lists costs more than summing it. This kernel decodes the
/// elements in cache-sized slices and carries each group's running sum across slice boundaries.
#[derive(Debug)]
pub(crate) struct ChunkedGroupedSumV2Kernel;

impl DynGroupedAggregateKernel for ChunkedGroupedSumV2Kernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(options) = aggregate_fn.as_opt::<SumV2>() else {
            return Ok(None);
        };
        let elements = groups.elements();
        if elements.is::<Primitive>() || !matches!(elements.dtype(), DType::Primitive(..)) {
            return Ok(None);
        }
        let group_ranges = groups.group_ranges(ctx)?;
        let group_validity = groups.group_validity(ctx)?;
        chunked_grouped_sum(
            elements,
            &group_ranges,
            &group_validity,
            SumMode::Native {
                skip_nans: options.skip_nans,
            },
            ctx,
        )
    }
}

/// How [`chunked_grouped_sum`] sums its input values.
#[derive(Clone, Copy)]
enum SumMode {
    /// Sum the values in their own type, as [`grouped_sum`] does.
    Native { skip_nans: bool },
    /// Convert integers to `f64` and sum the floats, as [`CastGroupedSumV2EncodingKernel`] does.
    CastToF64,
}

/// Number of elements decoded at a time, sized so a decoded slice of 64-bit values stays in cache.
const SUM_CHUNK_LEN: usize = 8 * 1024;

/// Sum groups whose ranges are ordered and disjoint by decoding `input` one slice at a time.
///
/// Returns `None` when the group ranges are not ordered, so the caller can decode the whole input.
fn chunked_grouped_sum(
    input: &ArrayRef,
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    mode: SumMode,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let mut prev_end = 0;
    for (offset, size) in group_ranges.iter() {
        if offset < prev_end {
            return Ok(None);
        }
        prev_end = offset + size;
    }
    let DType::Primitive(ptype, _) = input.dtype() else {
        return Ok(None);
    };
    let ptype = *ptype;

    let (sums, is_overflow, is_empty) = match mode {
        SumMode::CastToF64 => {
            if !ptype.is_int() {
                return Ok(None);
            }
            match_each_integer_ptype!(ptype, |T| {
                chunked_sums::<T, f64>(input, group_ranges, group_validity, ctx, |acc, slice| {
                    sum_float_all(acc, slice, false);
                    false
                })?
            })
        }
        SumMode::Native { skip_nans } => match_each_native_ptype!(ptype,
            unsigned: |T| {
                chunked_sums::<T, u64>(input, group_ranges, group_validity, ctx, sum_unsigned_all)?
            },
            signed: |T| {
                chunked_sums::<T, i64>(input, group_ranges, group_validity, ctx, sum_signed_all)?
            },
            floating: |T| {
                chunked_sums::<T, f64>(input, group_ranges, group_validity, ctx, |acc, slice| {
                    sum_float_all(acc, slice, skip_nans);
                    false
                })?
            }
        ),
    };

    let partial_fields = sum_v2_partial_fields(sums.dtype().clone());

    // SAFETY: all three children have one value per group and match `partial_fields`; the struct
    // validity is derived from the same group count.
    Ok(Some(
        unsafe {
            StructArray::new_unchecked(
                vec![
                    sums.into_array(),
                    BoolArray::new(is_overflow, Validity::NonNullable).into_array(),
                    BoolArray::new(is_empty, Validity::NonNullable).into_array(),
                ],
                partial_fields,
                group_validity.len(),
                Validity::from_mask(group_validity.clone(), Nullability::Nullable),
            )
        }
        .into_array(),
    ))
}

/// The running sum of one group.
#[derive(Default)]
struct GroupSum<A> {
    acc: A,
    overflow: bool,
    any_valid: bool,
}

/// Sum ordered, disjoint groups of `input` while decoding it one slice at a time.
fn chunked_sums<T: NativePType, A: NativePType + Default>(
    input: &ArrayRef,
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    ctx: &mut ExecutionCtx,
    sum_run: impl Fn(&mut A, &[T]) -> bool,
) -> VortexResult<(PrimitiveArray, BitBuffer, BitBuffer)> {
    let group_count = group_ranges.len();
    let mut sums = BufferMut::<A>::with_capacity(group_count);
    let mut is_overflow = BitBufferMut::new_unset(group_count);
    let mut is_empty = BitBufferMut::new_unset(group_count);

    // The decoded slice `[chunk_start, chunk_start + chunk.len())` of the input.
    let mut chunk_start = 0;
    let mut chunk: Option<(PrimitiveArray, Mask)> = None;

    for (group, (offset, size)) in group_ranges.iter().enumerate() {
        let mut state = GroupSum::<A>::default();
        if group_validity.value(group) {
            let end = offset + size;
            let mut pos = offset;
            while pos < end && !state.overflow {
                let loaded = chunk.as_ref().is_some_and(|(values, _)| {
                    pos >= chunk_start && pos < chunk_start + values.len()
                });
                if !loaded {
                    // Start on a block boundary, since encodings such as bit-packing decode whole
                    // blocks of 1024 values.
                    chunk_start = pos - pos % 1024;
                    let chunk_end = (chunk_start + SUM_CHUNK_LEN).min(input.len());
                    let values = input
                        .slice(chunk_start..chunk_end)?
                        .execute::<PrimitiveArray>(ctx)?;
                    let mask = values
                        .as_ref()
                        .validity()?
                        .execute_mask(values.as_ref().len(), ctx)?;
                    chunk = Some((values, mask));
                }
                let (values, mask) = chunk.as_ref().vortex_expect("chunk was just loaded");
                let local_start = pos - chunk_start;
                let local_end = (end - chunk_start).min(values.len());
                let (overflow, any_valid) = sum_masked_group(
                    &mut state.acc,
                    values.as_slice::<T>(),
                    local_start,
                    local_end - local_start,
                    mask,
                    &sum_run,
                );
                state.overflow |= overflow;
                state.any_valid |= any_valid;
                pos = chunk_start + local_end;
            }
        }
        if state.overflow {
            // SAFETY: `group` comes from enumerating `group_ranges`, and the bitmap has one bit
            // per group.
            unsafe { is_overflow.set_unchecked(group) };
        }
        if !state.any_valid {
            // SAFETY: as above.
            unsafe { is_empty.set_unchecked(group) };
        }
        sums.push(state.acc);
    }

    Ok((
        PrimitiveArray::new(sums.freeze(), Validity::NonNullable),
        is_overflow.freeze(),
        is_empty.freeze(),
    ))
}

/// Grouped [`SumV2`] implementation for canonical primitive elements.
///
/// Reuses the scalar primitive-sum reductions ([`sum_unsigned_all`]/[`sum_signed_all`]/
/// [`sum_float_all`]) so the per-group semantics match scalar `sum_v2` exactly (overflow saturates
/// to a null sum, NaNs are skipped). The element validity mask is materialized once and sliced
/// per group, rather than the per-group accumulator setup of the generic fallback path.
pub(super) fn try_grouped_sum(
    groups: &GroupedArray,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
) -> VortexResult<Option<ArrayRef>> {
    if !groups.elements().is::<Primitive>() {
        return Ok(None);
    }
    let elements = groups.elements().clone().downcast::<Primitive>();
    let group_ranges = groups.group_ranges(ctx)?;
    let group_validity = groups.group_validity(ctx)?;

    Ok(Some(grouped_sum(
        &elements,
        &group_ranges,
        &group_validity,
        ctx,
        skip_nans,
    )?))
}

/// Sum each group described by `group_ranges` (element `(offset, size)` pairs), one sum per group.
fn grouped_sum(
    elements: &PrimitiveArray,
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
) -> VortexResult<ArrayRef> {
    let elem_mask = elements
        .as_ref()
        .validity()?
        .execute_mask(elements.as_ref().len(), ctx)?;
    let all_valid = elem_mask.all_true();

    let (sums, is_overflow, is_empty) = match_each_native_ptype!(elements.ptype(),
        unsigned: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, u64>(
                values, group_ranges, group_validity, &elem_mask, all_valid, sum_unsigned_all)
        },
        signed: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, i64>(
                values, group_ranges, group_validity, &elem_mask, all_valid, sum_signed_all)
        },
        floating: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, f64>(
                values, group_ranges, group_validity, &elem_mask, all_valid,
                |acc, slice| { sum_float_all(acc, slice, skip_nans); false })
        }
    );

    let partial_fields = sum_v2_partial_fields(sums.dtype().clone());

    // SAFETY: all three children have one value per group and match `partial_fields`; the struct
    // validity is derived from the same group count.
    Ok(unsafe {
        StructArray::new_unchecked(
            vec![
                sums.into_array(),
                BoolArray::new(is_overflow, Validity::NonNullable).into_array(),
                BoolArray::new(is_empty, Validity::NonNullable).into_array(),
            ],
            partial_fields,
            group_validity.len(),
            Validity::from_mask(group_validity.clone(), Nullability::Nullable),
        )
    }
    .into_array())
}

/// Reduce each group's element slice into a non-null sum, overflow bitmap, and empty bitmap.
fn collect_sums<T: NativePType, A: NativePType + Default>(
    values: &[T],
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    elem_mask: &Mask,
    all_valid: bool,
    sum_run: impl Fn(&mut A, &[T]) -> bool,
) -> (PrimitiveArray, BitBuffer, BitBuffer) {
    let group_count = group_ranges.len();
    let mut is_overflow = BitBufferMut::new_unset(group_count);
    let mut is_empty = BitBufferMut::new_unset(group_count);
    let sums = group_ranges.iter().enumerate().map(|(i, (offset, size))| {
        if !group_validity.value(i) {
            return A::default();
        }
        let mut acc = A::default();
        let (overflow, any_valid) = if all_valid {
            (sum_run(&mut acc, &values[offset..offset + size]), size > 0)
        } else {
            sum_masked_group(&mut acc, values, offset, size, elem_mask, &sum_run)
        };
        if overflow {
            // SAFETY: `i` comes from enumerating `group_ranges`, and the bitmap has one bit per
            // group.
            unsafe { is_overflow.set_unchecked(i) };
        }
        if !any_valid {
            // SAFETY: `i` comes from enumerating `group_ranges`, and the bitmap has one bit per
            // group.
            unsafe { is_empty.set_unchecked(i) };
        }
        acc
    });
    let sums = PrimitiveArray::from_iter(sums);
    (sums, is_overflow.freeze(), is_empty.freeze())
}

/// Sum valid runs in one group, returning `(overflow, any_valid)`.
fn sum_masked_group<T: NativePType, A>(
    acc: &mut A,
    values: &[T],
    offset: usize,
    size: usize,
    elem_mask: &Mask,
    sum_run: &impl Fn(&mut A, &[T]) -> bool,
) -> (bool, bool) {
    match elem_mask {
        Mask::AllTrue(_) => (sum_run(acc, &values[offset..offset + size]), size > 0),
        Mask::AllFalse(_) => (false, false),
        Mask::Values(mask_values) => {
            let validity = mask_values
                .bit_buffer()
                .as_view()
                .slice(offset..offset + size);
            let mut any_valid = false;
            for (start, end) in validity.set_slices() {
                any_valid = true;
                if sum_run(acc, &values[offset + start..offset + end]) {
                    return (true, true);
                }
            }
            (false, any_valid)
        }
    }
}
