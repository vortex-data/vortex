// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::SumV2;
use super::sum_v2_partial_fields;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynGroupedAccumulator;
use crate::aggregate_fn::GroupRanges;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::sum::sum_float_all;
use crate::aggregate_fn::fns::sum::sum_signed_all;
use crate::aggregate_fn::fns::sum::sum_unsigned_all;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Chunked;
use crate::arrays::ListViewArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
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

    Ok(grouped_partials(
        sums.into_array(),
        is_overflow,
        is_empty,
        group_validity,
    ))
}

/// The grouped [`SumV2`] partials of non-null widened `sums`, one per group.
pub(crate) fn grouped_partials(
    sums: ArrayRef,
    is_overflow: BitBuffer,
    is_empty: BitBuffer,
    group_validity: &Mask,
) -> ArrayRef {
    let partial_fields = sum_v2_partial_fields(sums.dtype().clone());

    // SAFETY: all three children have one value per group and match `partial_fields`; the struct
    // validity is derived from the same group count.
    unsafe {
        StructArray::new_unchecked(
            vec![
                sums,
                BoolArray::new(is_overflow, Validity::NonNullable).into_array(),
                BoolArray::new(is_empty, Validity::NonNullable).into_array(),
            ],
            partial_fields,
            group_validity.len(),
            Validity::from_mask(group_validity.clone(), Nullability::Nullable),
        )
    }
    .into_array()
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

/// Grouped [`SumV2`] kernel for bool elements: a group's sum is the count of its valid `true`
/// elements.
#[derive(Debug)]
pub(crate) struct BoolGroupedSumV2EncodingKernel;

impl DynGroupedAggregateKernel for BoolGroupedSumV2EncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !aggregate_fn.is::<SumV2>() || !groups.elements().is::<Bool>() {
            return Ok(None);
        }
        let elements = groups.elements().clone().downcast::<Bool>();
        let len = elements.as_ref().len();
        let validity = elements.as_ref().validity()?.execute_mask(len, ctx)?;
        let values = elements.to_bit_buffer();
        let counted = match validity.bit_buffer() {
            AllOr::All => values,
            AllOr::None => BitBuffer::new_unset(len),
            AllOr::Some(valid) => &values & valid,
        };
        let group_ranges = groups.group_ranges(ctx)?;
        let group_validity = groups.group_validity(ctx)?;

        let mut is_empty = BitBufferMut::new_unset(group_ranges.len());
        let sums: Buffer<u64> = group_ranges
            .iter()
            .enumerate()
            .map(|(group, (offset, size))| {
                if !group_validity.value(group) {
                    return 0;
                }
                let end = offset + size;
                let any_valid = match validity.bit_buffer() {
                    AllOr::All => size > 0,
                    AllOr::None => false,
                    AllOr::Some(valid) => valid.count_range(offset, end) > 0,
                };
                if !any_valid {
                    // SAFETY: `group` comes from enumerating `group_ranges`, and the bitmap has
                    // one bit per group.
                    unsafe { is_empty.set_unchecked(group) };
                }
                counted.count_range(offset, end) as u64
            })
            .collect();
        Ok(Some(grouped_partials(
            sums.into_array(),
            BitBuffer::new_unset(group_ranges.len()),
            is_empty.freeze(),
            &group_validity,
        )))
    }
}

/// Grouped [`SumV2`] kernel for chunked elements.
///
/// Each chunk sums the parts of the groups it holds with its own grouped kernel, and a group that
/// spans chunks adds up its parts. Executing the chunked elements instead would decode and copy
/// every chunk into one array first.
#[derive(Debug)]
pub(crate) struct ChunkedGroupedSumV2EncodingKernel;

/// The parts of groups one chunk holds: which group, and where in the chunk.
#[derive(Default)]
struct ChunkParts {
    groups: Vec<usize>,
    offsets: BufferMut<u64>,
    sizes: BufferMut<u64>,
}

impl DynGroupedAggregateKernel for ChunkedGroupedSumV2EncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(options) = aggregate_fn.as_opt::<SumV2>() else {
            return Ok(None);
        };
        if !groups.elements().is::<Chunked>() {
            return Ok(None);
        }
        let elements = groups.elements().clone().downcast::<Chunked>();
        let Some(mut partials) = SumV2GroupPartials::new(options, elements.dtype(), groups.len())
        else {
            return Ok(None);
        };
        let group_ranges = groups.group_ranges(ctx)?;
        let group_validity = groups.group_validity(ctx)?;

        let bounds = elements.chunk_offset_values();
        let mut parts: Vec<ChunkParts> = (0..elements.nchunks())
            .map(|_| ChunkParts::default())
            .collect();
        for (group, (offset, size)) in group_ranges.iter().enumerate() {
            if size == 0 || !group_validity.value(group) {
                continue;
            }
            // The groups of a list view can come in any order.
            let mut chunk = bounds.partition_point(|&bound| bound <= offset) - 1;
            let (mut start, end) = (offset, offset + size);
            while start < end {
                let chunk_end = bounds[chunk + 1];
                if start < chunk_end {
                    let part_end = end.min(chunk_end);
                    let part = &mut parts[chunk];
                    part.groups.push(group);
                    part.offsets.push((start - bounds[chunk]) as u64);
                    part.sizes.push((part_end - start) as u64);
                    start = part_end;
                }
                chunk += 1;
            }
        }

        for (chunk, part) in parts.into_iter().enumerate() {
            if part.groups.is_empty() {
                continue;
            }
            // SAFETY: each part lies within its chunk, and the offsets and sizes are one per part.
            let part_groups = unsafe {
                ListViewArray::new_unchecked(
                    elements.chunk(chunk).clone(),
                    part.offsets.into_array(),
                    part.sizes.into_array(),
                    Validity::NonNullable,
                )
            };
            let mut accumulator =
                GroupedAccumulator::try_new(SumV2, *options, elements.dtype().clone())?;
            accumulator.accumulate_list(&part_groups.into_array(), ctx)?;
            partials.merge(&part.groups, &accumulator.flush()?, ctx)?;
        }
        Ok(Some(partials.finish(&group_validity)))
    }
}

/// [`SumV2`] partials for groups whose parts are summed separately, such as a group spanning
/// chunks, or a sparse array's fill and patches. Merging adds a group's parts like
/// [`AggregateFnVTable::merge_partials`](crate::aggregate_fn::AggregateFnVTable::merge_partials).
pub struct SumV2GroupPartials {
    sums: GroupSums,
    is_overflow: BitBufferMut,
    is_empty: BitBufferMut,
}

enum GroupSums {
    Unsigned(BufferMut<u64>),
    Signed(BufferMut<i64>),
    Float {
        sums: BufferMut<f64>,
        skip_nans: bool,
    },
}

impl SumV2GroupPartials {
    /// Empty partials for `groups` groups of `input_dtype` values, or `None` when the sum is not
    /// a widened primitive.
    pub fn new(
        options: &NumericalAggregateOpts,
        input_dtype: &DType,
        groups: usize,
    ) -> Option<Self> {
        let sums = match SumV2.return_dtype(options, input_dtype)? {
            DType::Primitive(PType::U64, _) => GroupSums::Unsigned(BufferMut::zeroed(groups)),
            DType::Primitive(PType::I64, _) => GroupSums::Signed(BufferMut::zeroed(groups)),
            DType::Primitive(PType::F64, _) => GroupSums::Float {
                sums: BufferMut::zeroed(groups),
                skip_nans: options.skip_nans,
            },
            _ => return None,
        };
        Some(Self {
            sums,
            is_overflow: BitBufferMut::new_unset(groups),
            is_empty: BitBufferMut::new_set(groups),
        })
    }

    /// Add `partials`, grouped [`SumV2`] partials over parts of the groups `groups`, one group per
    /// partial.
    pub fn merge(
        &mut self,
        groups: &[usize],
        partials: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let partials = partials.clone().execute::<StructArray>(ctx)?;
        let sums = partials
            .unmasked_field(0)
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let is_overflow = partials
            .unmasked_field(1)
            .clone()
            .execute::<BoolArray>(ctx)?
            .into_bit_buffer();
        let is_empty = partials
            .unmasked_field(2)
            .clone()
            .execute::<BoolArray>(ctx)?
            .into_bit_buffer();
        match &mut self.sums {
            GroupSums::Unsigned(totals) => merge_sums(
                totals,
                &mut self.is_overflow,
                &mut self.is_empty,
                groups,
                sums.as_slice::<u64>(),
                &is_overflow,
                &is_empty,
                u64::checked_add,
            ),
            GroupSums::Signed(totals) => merge_sums(
                totals,
                &mut self.is_overflow,
                &mut self.is_empty,
                groups,
                sums.as_slice::<i64>(),
                &is_overflow,
                &is_empty,
                i64::checked_add,
            ),
            GroupSums::Float { sums: totals, .. } => merge_sums(
                totals,
                &mut self.is_overflow,
                &mut self.is_empty,
                groups,
                sums.as_slice::<f64>(),
                &is_overflow,
                &is_empty,
                |total, sum| Some(total + sum),
            ),
        }
        Ok(())
    }

    /// Add `counts[group]` copies of `value` to each group.
    pub fn add_repeated(&mut self, value: &Scalar, counts: &[u64]) -> VortexResult<()> {
        if value.is_null() {
            return Ok(());
        }
        match &mut self.sums {
            GroupSums::Unsigned(totals) => {
                let value = match value.as_bool_opt() {
                    Some(value) => u64::from(value.value().unwrap_or(false)),
                    None => value
                        .as_primitive()
                        .as_::<u64>()
                        .ok_or_else(|| vortex_err!("{value} is not a u64 sum"))?,
                };
                add_repeated(
                    totals,
                    &mut self.is_overflow,
                    &mut self.is_empty,
                    counts,
                    |total, count| value.checked_mul(count)?.checked_add(total),
                );
            }
            GroupSums::Signed(totals) => {
                let value = value
                    .as_primitive()
                    .as_::<i64>()
                    .ok_or_else(|| vortex_err!("{value} is not an i64 sum"))?;
                add_repeated(
                    totals,
                    &mut self.is_overflow,
                    &mut self.is_empty,
                    counts,
                    |total, count| {
                        value
                            .checked_mul(i64::try_from(count).ok()?)?
                            .checked_add(total)
                    },
                );
            }
            GroupSums::Float {
                sums: totals,
                skip_nans,
            } => {
                let value = value
                    .as_primitive()
                    .as_::<f64>()
                    .ok_or_else(|| vortex_err!("{value} is not an f64 sum"))?;
                let skip = value.is_nan() && *skip_nans;
                add_repeated(
                    totals,
                    &mut self.is_overflow,
                    &mut self.is_empty,
                    counts,
                    |total, count| {
                        Some(if skip {
                            total
                        } else {
                            total + value * count as f64
                        })
                    },
                );
            }
        }
        Ok(())
    }

    /// The partials of the groups, null where `group_validity` is.
    pub fn finish(self, group_validity: &Mask) -> ArrayRef {
        let sums = match self.sums {
            GroupSums::Unsigned(sums) => sums.into_array(),
            GroupSums::Signed(sums) => sums.into_array(),
            GroupSums::Float { sums, .. } => sums.into_array(),
        };
        grouped_partials(
            sums,
            self.is_overflow.freeze(),
            self.is_empty.freeze(),
            group_validity,
        )
    }
}

/// Merge one partial per group of `groups` into the totals, as [`SumV2`] merges partials.
#[expect(
    clippy::too_many_arguments,
    reason = "the totals and the partials are three columns each"
)]
fn merge_sums<T: Copy>(
    totals: &mut [T],
    total_overflow: &mut BitBufferMut,
    total_empty: &mut BitBufferMut,
    groups: &[usize],
    sums: &[T],
    is_overflow: &BitBuffer,
    is_empty: &BitBuffer,
    add: impl Fn(T, T) -> Option<T>,
) {
    for (part, &group) in groups.iter().enumerate() {
        if total_overflow.value(group) {
            continue;
        }
        if is_overflow.value(part) {
            total_overflow.set(group);
            total_empty.unset(group);
            continue;
        }
        if is_empty.value(part) {
            continue;
        }
        match add(totals[group], sums[part]) {
            Some(total) => totals[group] = total,
            None => total_overflow.set(group),
        }
        total_empty.unset(group);
    }
}

/// Add each group's repeated value, `add(total, count)`, to the totals of groups with a count.
fn add_repeated<T: Copy>(
    totals: &mut [T],
    total_overflow: &mut BitBufferMut,
    total_empty: &mut BitBufferMut,
    counts: &[u64],
    add: impl Fn(T, u64) -> Option<T>,
) {
    for (group, &count) in counts.iter().enumerate() {
        if count == 0 || total_overflow.value(group) {
            continue;
        }
        match add(totals[group], count) {
            Some(total) => totals[group] = total,
            None => total_overflow.set(group),
        }
        total_empty.unset(group);
    }
}
