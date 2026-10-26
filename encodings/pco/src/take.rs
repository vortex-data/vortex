// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Batched selection using compact request descriptors and one operation-owned page cache.

use std::collections::BTreeSet;
use std::ops::Range;

use pco::data_types::Number;
use pco::data_types::NumberType;
use pco::match_number_enum;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PiecewiseSequence;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::half;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Pco;
use crate::PcoArrayExt;
use crate::access::AccessState;
use crate::array::number_type_from_ptype;
use crate::decoder::PageDecoder;
use crate::page_index::ChunkCursor;
use crate::page_index::IndexedView;
use crate::page_index::PageIndex;

/// Compact descriptor; null indices have no start. Zero-length runs consume no source rows.
struct Run {
    start: Option<usize>,
    length: usize,
    stride: usize,
}

impl TakeExecute for Pco {
    fn take(
        array: ArrayView<'_, Pco>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if array.metadata.logical_row_prefix.is_none() {
            return Ok(None);
        }
        if indices.is_empty() {
            let dtype = array
                .dtype()
                .clone()
                .union_nullability(indices.dtype().nullability());
            return Ok(Some(Canonical::empty(&dtype).into_array()));
        }
        // Compact range execution is introduced separately; preserve the normal fallback.
        if indices.is::<PiecewiseSequence>() {
            return Ok(None);
        }
        let grouped;
        let runs = {
            let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
            let mask = indices.validity()?.execute_mask(indices.len(), ctx)?;
            // Reuse native logical-null-aware extrema and their immutable-array stats cache.
            // Numeric extrema prove every defined index is representable and in bounds.
            let extrema = min_max(indices.as_array(), ctx, NumericalAggregateOpts::default())?;
            let summary = if let Some(extrema) = extrema {
                match_each_integer_ptype!(indices.ptype(), |I| {
                    validate_extrema(indices.as_slice::<I>(), &mask, &extrema, array.len())?
                })
            } else {
                PlainSummary {
                    defined: 0,
                    contiguous: false,
                }
            };
            if summary.defined == 0 {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                        .into_array(),
                ));
            }
            if let Some(result) = zero_page_take(array, indices.len(), ctx)? {
                return Ok(Some(result));
            }
            grouped = (!summary.contiguous).then_some(summary.defined);
            if grouped.is_some() && descriptor_bytes(summary.defined) >= canonical_bytes(array) {
                return Ok(None);
            }
            if grouped.is_some()
                && match_each_integer_ptype!(indices.ptype(), |I| {
                    prefer_canonical_primitive(
                        array,
                        view,
                        indices.as_slice::<I>(),
                        &mask,
                        summary.defined,
                    )?
                })
            {
                return Ok(None);
            }
            match_each_integer_ptype!(indices.ptype(), |I| {
                primitive_runs(indices.as_slice::<I>(), &mask)?
            })
        };
        let nullable = array.dtype().nullability() | indices.dtype().nullability();
        let result = match_number_enum!(number_type_from_ptype(array.dtype().as_ptype()), NumberType<T> => {
            gather::<T>(array, view, &runs, indices.len(), nullable, grouped, ctx)?
        });
        Ok(Some(result.into_array()))
    }
}

struct PlainSummary {
    defined: usize,
    contiguous: bool,
}

fn validate_extrema<I: NativePType>(
    indices: &[I],
    mask: &Mask,
    extrema: &MinMaxResult,
    source_len: usize,
) -> VortexResult<PlainSummary>
where
    usize: TryFrom<I>,
{
    let min = extrema
        .min
        .as_primitive()
        .try_typed_value::<I>()?
        .ok_or_else(|| vortex_err!("Pco index minimum is null"))?;
    let max = extrema
        .max
        .as_primitive()
        .try_typed_value::<I>()?
        .ok_or_else(|| vortex_err!("Pco index maximum is null"))?;
    checked_index(min)?;
    let max = checked_index(max)?;
    vortex_ensure!(
        max < source_len,
        "Pco take index {max} exceeds array length {source_len}"
    );
    // Bounds/error category is guaranteed above. No contract chooses the first invalid row.
    // Contiguous detection exits at the first mismatch; null physical values are never inspected.
    let contiguous = match mask {
        Mask::AllTrue(_) => indices.windows(2).all(|pair| {
            checked_index(pair[0])
                .ok()
                .and_then(|row| row.checked_add(1))
                == checked_index(pair[1]).ok()
        }),
        _ => false,
    };
    Ok(PlainSummary {
        defined: mask.true_count(),
        contiguous,
    })
}

fn zero_page_take(
    array: ArrayView<'_, Pco>,
    len: usize,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !array.pages.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        ConstantArray::new(Scalar::null(array.dtype().as_nullable()), len).into_array(),
    ))
}

fn canonical_bytes(array: ArrayView<'_, Pco>) -> usize {
    // Fallback canonicalizes the visible array, not its potentially much larger owner.
    array
        .len()
        .saturating_mul(array.dtype().as_ptype().byte_width())
}

fn descriptor_bytes(requests: usize) -> usize {
    // Request-planning cost heuristic, not a heap bound: Run/sorting/resolved descriptors.
    // Null runs, vector capacity, tree nodes, masks, ranks and decoder state are excluded.
    // Output storage is common to both strategies.
    requests.saturating_mul(
        size_of::<Run>() + size_of::<(usize, usize)>() + size_of::<(usize, usize, Range<usize>)>(),
    )
}

/// Borrow validated primitive inputs during planning. Broad requests stop once selected
/// physical values plus prospective descriptors exceed the visible canonical values buffer.
/// Only the chosen sparse strategy allocates Runs/sorting descriptors; no source-sized bitmap.
fn prefer_canonical_primitive<I: Copy>(
    array: ArrayView<'_, Pco>,
    view: IndexedView<'_>,
    indices: &[I],
    mask: &Mask,
    requests: usize,
) -> VortexResult<bool>
where
    usize: TryFrom<I>,
{
    let mut chunks = ChunkCursor::default();
    let full_bytes = canonical_bytes(array);
    let setup = descriptor_bytes(requests);
    let width = array.dtype().as_ptype().byte_width();
    let mut touched = BTreeSet::new();
    let mut selected = 0usize;
    for (&row, valid) in indices.iter().zip(mask.iter()) {
        if !valid {
            continue;
        }
        let row = array.slice_start() + checked_index(row)?;
        let page = view.find_row(row);
        if touched.insert(page) {
            selected = selected.saturating_add(view.physical(page, &mut chunks)?.n_values);
            if setup.saturating_add(selected.saturating_mul(width)) >= full_bytes {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn checked_index<I>(value: I) -> VortexResult<usize>
where
    usize: TryFrom<I>,
{
    usize::try_from(value).map_err(|_| vortex_err!("Pco take index is negative or exceeds usize"))
}

fn primitive_runs<I: Copy>(indices: &[I], mask: &Mask) -> VortexResult<Vec<Run>>
where
    usize: TryFrom<I>,
{
    let mut runs: Vec<Run> = Vec::with_capacity(indices.len().min(64));
    for (&value, valid) in indices.iter().zip(mask.iter()) {
        let start = if valid {
            Some(checked_index(value)?)
        } else {
            None
        };
        if let Some(previous) = runs.last_mut() {
            let contiguous = match (previous.start, start) {
                (Some(first), Some(next)) => {
                    previous.stride == 1 && first.checked_add(previous.length) == Some(next)
                }
                (None, None) => true,
                _ => false,
            };
            if contiguous {
                previous.length += 1;
                continue;
            }
        }
        runs.push(Run {
            start,
            length: 1,
            stride: 1,
        });
    }
    Ok(runs)
}

fn gather<T: Number + NativePType>(
    array: ArrayView<'_, Pco>,
    view: IndexedView<'_>,
    runs: &[Run],
    output_len: usize,
    nullability: Nullability,
    grouped: Option<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    if let [
        Run {
            start: Some(start),
            length,
            stride: 1,
        },
    ] = runs
    {
        return stream_range::<T>(array, view, *start..*start + *length, nullability, ctx);
    }
    let defined = grouped.unwrap_or(output_len);
    if nullability.is_nullable() {
        gather_grouped::<T, true>(array, view, runs, output_len, defined, ctx)
    } else {
        gather_grouped::<T, false>(array, view, runs, output_len, defined, ctx)
    }
}

/// Sort requests once, resolve each repeated logical position once, then invoke the native
/// primitive gather once per physical page. Only the final scatter visits output positions;
/// it does no page-index search, validity rank, decoder lookup, or cache lookup.
fn gather_grouped<T: Number + NativePType, const NULLABLE: bool>(
    array: ArrayView<'_, Pco>,
    view: IndexedView<'_>,
    runs: &[Run],
    output_len: usize,
    defined: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let mut requests = Vec::with_capacity(defined);
    let mut output = 0usize;
    for run in runs {
        if let Some(start) = run.start {
            for offset in 0..run.length {
                requests.push((
                    array.slice_start() + start + offset * run.stride,
                    output + offset,
                ));
            }
        }
        output += run.length;
    }
    requests.sort_unstable();
    let mut state = AccessState::default();
    let mut resolved = Vec::with_capacity(requests.len().min(64));
    let mut cursor = 0;
    while cursor < requests.len() {
        let row = requests[cursor].0;
        let end = cursor + requests[cursor..].partition_point(|request| request.0 == row);
        if let Some((page, value)) = state.resolve::<true>(view, array, row, ctx)? {
            resolved.push((page, value, cursor..end));
        }
        cursor = end;
    }
    let mut values = BufferMut::<T>::zeroed_in(output_len, ctx.allocator().clone());
    let mut valid = BitBufferMut::new_unset(if NULLABLE { output_len } else { 0 });
    // Requests are already in page order. Release all validity/rank preparation before
    // decoding, and keep only the current decoded page while scattering each group.
    drop(state);
    if !resolved.is_empty() {
        let mut decoder = PageDecoder::<T>::new(array.metadata.header.as_slice())?;
        let mut chunks = ChunkCursor::default();
        let mut cursor = 0;
        while cursor < resolved.len() {
            let page = resolved[cursor].0;
            let end = cursor + resolved[cursor..].partition_point(|request| request.0 == page);
            let PageIndex::IndexedFast(index) = array.page_index()? else {
                return Ok(None);
            };
            let view = index.view();
            let compact = PrimitiveArray::from_iter(
                resolved[cursor..end]
                    .iter()
                    .map(|request| {
                        u64::try_from(request.1)
                            .map_err(|_| vortex_err!("Pco compact index exceeds u64"))
                    })
                    .collect::<VortexResult<Vec<_>>>()?,
            )
            .into_array();
            let info = view.physical(page, &mut chunks)?;
            let buffer = array
                .pages
                .get(page)
                .ok_or_else(|| vortex_err!("Pco compressed page is missing"))?;
            let chunk = array
                .chunk_metas
                .get(info.chunk)
                .ok_or_else(|| vortex_err!("Pco chunk metadata is missing"))?;
            let decoded = decoder.read(
                info.chunk,
                chunk.as_slice(),
                info.n_values,
                buffer.as_slice(),
                ctx,
            )?;
            let selected = decoded
                .into_array()
                .take(compact)?
                .execute::<PrimitiveArray>(ctx)?;
            for (request, &value) in resolved[cursor..end].iter().zip(selected.as_slice::<T>()) {
                for &(_, output) in &requests[request.2.clone()] {
                    values[output] = value;
                    if NULLABLE {
                        valid.set(output);
                    }
                }
            }
            cursor = end;
        }
    }
    let validity = if NULLABLE {
        Validity::from(valid.freeze())
    } else {
        Validity::NonNullable
    };
    Ok(PrimitiveArray::new(values.freeze(), validity))
}

/// One initialized final output buffer. Nullable outputs leave null positions at the
/// allocator's safe zero initialization instead of eagerly touching every null gap.
/// Contiguous selections stream pages into the final initialized values buffer and retain
/// exact sliced source validity. No output validity copy or retained decoded-page cache.
fn stream_range<T: Number + NativePType>(
    array: ArrayView<'_, Pco>,
    view: IndexedView<'_>,
    range: Range<usize>,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let global_start = array.slice_start() + range.start;
    let mut values = BufferMut::<T>::zeroed_in(range.len(), ctx.allocator().clone());
    let mut scratch = BufferMut::<T>::with_capacity_in(0, ctx.allocator().clone());
    if view.page_count() == 0 {
        let validity = array
            .unsliced_validity()
            .slice(global_start..global_start + range.len())?
            .union_nullability(nullability);
        return Ok(PrimitiveArray::new(values.freeze(), validity));
    }
    let mut state = AccessState::default();
    let mut decoder = PageDecoder::<T>::new(array.metadata.header.as_slice())?;
    let mut chunks = ChunkCursor::default();
    let stop = array.slice_start() + range.end;
    let mut row = global_start;
    // Descriptor-count iteration bounds work even when optional index contents are bad.
    // Valid files visit at most one span per physical page; an all-null span needs one turn.
    for _ in 0..=view.page_count() {
        if row >= stop {
            break;
        }
        let span = state.span(view, array, row, stop, ctx)?;
        let start = row - global_start;
        let output = values
            .get_mut(start..start + span.mask.len())
            .ok_or_else(|| vortex_err!("Pco contiguous output exceeds visible range"))?;
        if !span.mask.all_false() {
            let page = view.physical(span.page, &mut chunks)?;
            let buffer = array
                .pages
                .get(span.page)
                .ok_or_else(|| vortex_err!("Pco compressed page is missing"))?;
            let chunk = array
                .chunk_metas
                .get(page.chunk)
                .ok_or_else(|| vortex_err!("Pco chunk metadata is missing"))?;
            let n = page.n_values;
            if span.mask.all_true() && span.values == (0..n) {
                decoder.read_into(page.chunk, chunk.as_slice(), n, buffer.as_slice(), output)?;
            } else {
                if scratch.len() < n {
                    scratch.push_n(T::default(), n - scratch.len());
                }
                decoder.read_into(
                    page.chunk,
                    chunk.as_slice(),
                    n,
                    buffer.as_slice(),
                    &mut scratch[..n],
                )?;
                let source = scratch
                    .get(span.values)
                    .ok_or_else(|| vortex_err!("Pco compact range exceeds page output"))?;
                match &span.mask {
                    Mask::AllTrue(_) => output.copy_from_slice(source),
                    Mask::AllFalse(_) => {}
                    Mask::Values(mask) => {
                        let mut compact = 0;
                        for (start, end) in mask.bit_buffer().set_slices() {
                            let len = end - start;
                            output[start..end].copy_from_slice(&source[compact..compact + len]);
                            compact += len;
                        }
                    }
                }
            }
        }
        row = span.stop;
    }
    let validity = array
        .unsliced_validity()
        .slice(global_start..stop)?
        .union_nullability(nullability);
    Ok(PrimitiveArray::new(values.freeze(), validity))
}
