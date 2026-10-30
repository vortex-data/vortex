// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selective decoding for compact range takes and slices.

use std::iter::repeat_n;
use std::ops::Range;

use pco::data_types::Number;
use pco::data_types::NumberType;
use pco::match_number_enum;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PiecewiseSequence;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::piecewise_sequence::array::PiecewiseSequenceArraySlotsExt;
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
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Pco;
use crate::PcoArrayExt;
use crate::access::AccessState;
use crate::array::number_type_from_ptype;
use crate::decoder::DecodedPages;
use crate::decoder::PageDecoder;
use crate::decoder::PageLookup;
use crate::page_index::LogicalPageIndex;
use crate::page_index::PageIndex;

/// Compact descriptor; null indices have no start. Zero-length runs consume no source rows.
struct Run {
    start: Option<usize>,
    length: usize,
    stride: usize,
}

pub(crate) fn take(
    array: ArrayView<'_, Pco>,
    index: &LogicalPageIndex,
    sequence: ArrayView<'_, PiecewiseSequence>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let starts = sequence.starts().clone().execute::<PrimitiveArray>(ctx)?;
    let lengths = sequence.lengths().clone().execute::<PrimitiveArray>(ctx)?;
    let multipliers = sequence
        .multipliers()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let starts = primitive_usizes(&starts)?;
    let lengths = primitive_usizes(&lengths)?;
    let multipliers = primitive_usizes(&multipliers)?;
    let mut runs = Vec::with_capacity(starts.len());
    for ((start, length), stride) in starts.into_iter().zip(lengths).zip(multipliers) {
        runs.push(Run {
            start: Some(start),
            length,
            stride,
        });
    }
    validate_runs(&runs, array.len(), sequence.len())?;
    if array.pages.is_empty() {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), sequence.len())
                .into_array(),
        ));
    }
    let random = runs
        .iter()
        .filter(|run| run.stride > 1)
        .map(|run| run.length)
        .sum();
    if descriptor_bytes(random) >= canonical_bytes(array) && random != 0 {
        return Ok(None);
    }
    let nullable = array.dtype().nullability() | sequence.dtype().nullability();
    let result = match_number_enum!(number_type_from_ptype(array.dtype().as_ptype()), NumberType<T> => {
        gather::<T>(array, index, &runs, sequence.len(), nullable, ctx)?
    });
    Ok(Some(result.into_array()))
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

fn checked_index<I>(value: I) -> VortexResult<usize>
where
    usize: TryFrom<I>,
{
    usize::try_from(value).map_err(|_| vortex_err!("Pco range descriptor exceeds usize"))
}

fn primitive_usizes(array: &PrimitiveArray) -> VortexResult<Vec<usize>> {
    match_each_integer_ptype!(array.ptype(), |I| {
        array
            .as_slice::<I>()
            .iter()
            .map(|&value| checked_index(value))
            .collect()
    })
}

fn validate_runs(runs: &[Run], source_len: usize, output_len: usize) -> VortexResult<()> {
    let mut total = 0usize;
    for run in runs {
        total = total
            .checked_add(run.length)
            .ok_or_else(|| vortex_err!("Pco take output length overflows usize"))?;
        if let Some(start) = run.start
            && run.length != 0
        {
            let last = (run.length - 1)
                .checked_mul(run.stride)
                .and_then(|delta| start.checked_add(delta))
                .ok_or_else(|| vortex_err!("Pco take range overflows usize"))?;
            vortex_ensure!(
                last < source_len,
                "Pco take index {last} exceeds array length {source_len}"
            );
        }
    }
    vortex_ensure_eq!(
        total,
        output_len,
        "Pco take expanded length does not match declared length"
    );
    Ok(())
}

fn gather<T: Number + NativePType>(
    array: ArrayView<'_, Pco>,
    view: &LogicalPageIndex,
    runs: &[Run],
    output_len: usize,
    nullability: Nullability,
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
    if nullability.is_nullable() {
        gather_typed::<T, true>(array, view, runs, output_len, ctx)
    } else {
        gather_typed::<T, false>(array, view, runs, output_len, ctx)
    }
}

struct Output<T: NativePType, const NULLABLE: bool> {
    values: BufferMut<T>,
    len: usize,
}

impl<T: NativePType, const NULLABLE: bool> Output<T, NULLABLE> {
    fn new(len: usize, ctx: &ExecutionCtx) -> Self {
        let values = if NULLABLE {
            BufferMut::zeroed_in(len, ctx.allocator().clone())
        } else {
            BufferMut::with_capacity_in(len, ctx.allocator().clone())
        };
        Self { values, len: 0 }
    }

    fn push(&mut self, value: T) {
        if NULLABLE {
            self.values[self.len] = value;
        } else {
            self.values.push(value);
        }
        self.len += 1;
    }

    fn extend_from_slice(&mut self, values: &[T]) {
        if NULLABLE {
            self.values[self.len..self.len + values.len()].copy_from_slice(values);
        } else {
            self.values.extend_from_slice(values);
        }
        self.len += values.len();
    }

    fn repeat(&mut self, value: T, len: usize) {
        if NULLABLE {
            self.values[self.len..self.len + len].fill(value);
        } else {
            self.values.extend(repeat_n(value, len));
        }
        self.len += len;
    }

    fn nulls(&mut self, len: usize) {
        if NULLABLE {
            self.len += len;
        } else {
            self.repeat(T::default(), len);
        }
    }
}

fn gather_typed<T: Number + NativePType, const NULLABLE: bool>(
    array: ArrayView<'_, Pco>,
    view: &LogicalPageIndex,
    runs: &[Run],
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let mut state = AccessState::default();
    let mut decoded = DecodedPages::<T>::new(array)?;
    let mut values = Output::<T, NULLABLE>::new(output_len, ctx);
    // The non-nullable monomorphization has no bitmap allocation or append work.
    let mut valid = BitBufferMut::with_capacity(if NULLABLE { output_len } else { 0 });
    for run in runs {
        let Some(start) = run.start else {
            if NULLABLE {
                valid.append_n(false, run.length);
            }
            values.nulls(run.length);
            continue;
        };
        if run.stride == 0 && run.length != 0 {
            if let Some((page, value)) =
                state.resolve::<true>(view, array, array.slice_start() + start, ctx)?
            {
                let value = decoded.values(view, array, page, ctx)?.as_slice::<T>()[value];
                values.repeat(value, run.length);
                if NULLABLE {
                    valid.append_n(true, run.length);
                }
            } else {
                if NULLABLE {
                    valid.append_n(false, run.length);
                }
                values.nulls(run.length);
            }
            continue;
        }
        if run.stride == 1 && run.length != 0 {
            let mut row = array.slice_start() + start;
            let stop = row + run.length;
            // Descriptor-count iteration bounds work even when optional index contents are bad.
            // Valid files visit at most one span per physical page; an all-null span needs one turn.
            for _ in 0..=view.page_count() {
                if row >= stop {
                    break;
                }
                let span = state.span(view, array, row, stop, ctx)?;
                match &span.mask {
                    Mask::AllTrue(n) => {
                        values.extend_from_slice(
                            &decoded.values(view, array, span.page_index, ctx)?.as_slice::<T>()
                                [span.values],
                        );
                        if NULLABLE {
                            valid.append_n(true, *n);
                        }
                    }
                    Mask::AllFalse(n) => {
                        values.nulls(*n);
                        if NULLABLE {
                            valid.append_n(false, *n);
                        }
                    }
                    Mask::Values(mask) => {
                        let source = decoded
                            .values(view, array, span.page_index, ctx)?
                            .as_slice::<T>();
                        let source = &source[span.values];
                        let mut logical = 0usize;
                        let mut compact = 0usize;
                        // Copy defined runs straight into the final output. Null gaps are
                        // initialized once; no compact staging output is expanded afterward.
                        for (start, stop) in mask.bit_buffer().set_slices() {
                            values.nulls(start - logical);
                            let count = stop - start;
                            values.extend_from_slice(&source[compact..compact + count]);
                            compact += count;
                            logical = stop;
                        }
                        values.nulls(mask.len() - logical);
                        if NULLABLE {
                            valid.append_buffer(mask.bit_buffer());
                        }
                    }
                }
                row = span.stop;
            }
            continue;
        }
        for offset in 0..run.length {
            // The complete descriptor, including its output length and last element, was
            // checked before allocating output. The source slice itself is already validated.
            let row = array.slice_start() + start + offset * run.stride;
            if let Some((page, value)) = state.resolve::<true>(view, array, row, ctx)? {
                values.push(decoded.values(view, array, page, ctx)?.as_slice::<T>()[value]);
                if NULLABLE {
                    valid.append_true();
                }
            } else {
                values.nulls(1);
                if NULLABLE {
                    valid.append_false();
                }
            }
        }
    }
    let validity = if NULLABLE {
        Validity::from(valid.freeze())
    } else {
        Validity::NonNullable
    };
    vortex_ensure_eq!(values.len, output_len, "Pco range output is incomplete");
    Ok(PrimitiveArray::new(values.values.freeze(), validity))
}

/// One initialized final output buffer. Nullable outputs leave null positions at the
/// allocator's safe zero initialization instead of eagerly touching every null gap.
/// Contiguous selections stream pages into the final initialized values buffer and retain
/// exact sliced source validity. No output validity copy or retained decoded-page cache.
fn stream_range<T: Number + NativePType>(
    array: ArrayView<'_, Pco>,
    view: &LogicalPageIndex,
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
    let mut lookup = PageLookup::default();
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
            let (chunk_id, n_values) = lookup.resolve(view, array, span.page_index)?;
            let buffer = array
                .pages
                .get(span.page_index)
                .ok_or_else(|| vortex_err!("Pco compressed page is missing"))?;
            let chunk = array
                .chunk_metas
                .get(chunk_id)
                .ok_or_else(|| vortex_err!("Pco chunk metadata is missing"))?;
            let n = n_values;
            if span.mask.all_true() && span.values == (0..n) {
                decoder.read_into(chunk_id, chunk.as_slice(), n, buffer.as_slice(), output)?;
            } else {
                if scratch.len() < n {
                    scratch.reserve(n - scratch.len());
                    unsafe {
                        scratch.set_len(n);
                    }
                }
                decoder.read_into(
                    chunk_id,
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
    vortex_ensure_eq!(row, stop, "Pco slice output is incomplete");
    let validity = array
        .unsliced_validity()
        .slice(global_start..stop)?
        .union_nullability(nullability);
    Ok(PrimitiveArray::new(values.freeze(), validity))
}

pub(crate) fn slice(
    array: ArrayView<'_, Pco>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let PageIndex::IndexedFast(index) = array.page_index()? else {
        return array.decompress(&array.unsliced_validity(), ctx);
    };
    match_number_enum!(number_type_from_ptype(array.dtype().as_ptype()), NumberType<T> => {
        stream_range::<T>(array, index.as_ref(), 0..array.len(), array.dtype().nullability(), ctx)
    })
}
