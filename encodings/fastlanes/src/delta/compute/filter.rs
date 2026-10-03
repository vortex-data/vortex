// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::Delta as FastLanesDelta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::Delta;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::delta_decompress::decode_chunk;

/// When a mask touches at least this fraction of the chunks and selects at least
/// [`FALLBACK_MIN_DENSITY`] of the values, decoding everything and filtering the result with the
/// generic bulk filter is faster than decoding chunk by chunk and gathering. Both thresholds come
/// from benchmarking run, cluster, strided and uniform random masks over 32- and 64-bit values.
const FALLBACK_MIN_TOUCHED: f64 = 0.9;
const FALLBACK_MIN_DENSITY: f64 = 0.15;

/// Gathers the selected values one chunk at a time, decoding only the chunks that hold a selected
/// value and never materializing the full decoded array. Masks that select a large share of
/// values from nearly every chunk keep the decode-then-filter path, and contiguous masks keep the
/// slice path.
impl FilterKernel for Delta {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Mask::Values(values) = mask else {
            return Ok(None);
        };
        let bits = values.bit_buffer();
        // A contiguous mask executes as a slice, which decodes only the selected range.
        if is_contiguous(values) {
            return Ok(None);
        }
        let offset = array.offset();
        let total_chunks = (offset + array.len()).div_ceil(1024);
        let touched = touched_chunks(bits, offset);
        if touched as f64 >= FALLBACK_MIN_TOUCHED * total_chunks as f64
            && values.density() >= FALLBACK_MIN_DENSITY
        {
            return Ok(None);
        }

        let ptype = array.dtype().as_ptype();
        let validity = array.validity()?.filter(mask)?;
        let filtered = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
            const LANES: usize = U::LANES;
            let buffer = gather::<U, LANES>(array, bits, values.true_count(), ctx)?;
            PrimitiveArray::new(buffer, validity)
        });
        Ok(Some(filtered.reinterpret_cast(ptype).into_array()))
    }
}

/// Mirrors the filter executor's contiguity probe so declining stays cheap: cached slices or
/// indices answer directly, and otherwise only the candidate run or the tail after it is scanned.
fn is_contiguous(values: &MaskValues) -> bool {
    if let Some(slices) = values.cached_slices() {
        return slices.len() <= 1;
    }
    let true_count = values.true_count();
    if let Some(indices) = values.cached_indices() {
        return match (indices.first(), indices.last()) {
            (Some(first), Some(last)) => last - first + 1 == true_count,
            _ => true,
        };
    }
    let bits = values.bit_buffer();
    let Some(start) = bits.set_indices().next() else {
        return true;
    };
    let end = start + true_count;
    if end > bits.len() {
        return false;
    }
    if end - start <= bits.len() - end {
        bits.count_range(start, end) == true_count
    } else {
        bits.last_set_index() == Some(end - 1)
    }
}

/// Count the chunks that hold at least one selected value. `offset` is the array's position in
/// its first physical chunk.
fn touched_chunks(bits: &BitBuffer, offset: usize) -> usize {
    let mut count = 0;
    let mut last_counted = None;
    for (index, word) in bits.chunks().iter_padded().enumerate() {
        if word == 0 {
            continue;
        }
        let base = offset + index * 64;
        let first = (base + word.trailing_zeros() as usize) / 1024;
        let last = (base + 63 - word.leading_zeros() as usize) / 1024;
        for chunk in first..=last {
            if last_counted.is_none_or(|counted| chunk > counted) {
                count += 1;
                last_counted = Some(chunk);
            }
        }
    }
    count
}

/// Walk the mask 64 bits at a time: skip empty words, copy full words in bulk, and pick out the
/// set bits of partial words. Each touched chunk is decoded once.
fn gather<U, const LANES: usize>(
    array: ArrayView<'_, Delta>,
    bits: &BitBuffer,
    true_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<U>>
where
    U: NativePType + FastLanesDelta + Transpose,
{
    let bases = array
        .bases()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(U::PTYPE);
    let deltas = array
        .deltas()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(U::PTYPE);
    let mut chunks = DecodedChunk::<U, LANES>::new(bases.as_slice::<U>(), deltas.as_slice::<U>());

    let offset = array.offset();
    let mut output = BufferMut::<U>::with_capacity(true_count);
    for (index, mut word) in bits.chunks().iter_padded().enumerate() {
        if word == 0 {
            continue;
        }
        let base = offset + index * 64;
        if word == u64::MAX {
            // A full word may straddle a chunk boundary, so copy it in up to two parts.
            let (mut position, end) = (base, base + 64);
            while position < end {
                let chunk = position / 1024;
                let chunk_end = end.min((chunk + 1) * 1024);
                output.extend_from_slice(
                    &chunks.get(chunk)[position % 1024..chunk_end - chunk * 1024],
                );
                position = chunk_end;
            }
        } else {
            while word != 0 {
                let position = base + word.trailing_zeros() as usize;
                output.push(chunks.get(position / 1024)[position % 1024]);
                word &= word - 1;
            }
        }
    }
    Ok(output.freeze())
}

/// The most recently decoded chunk, so consecutive reads from one chunk decode it once.
struct DecodedChunk<'a, U, const LANES: usize> {
    bases: &'a [U],
    deltas: &'a [U],
    chunk: Option<usize>,
    transposed: [U; 1024],
    values: [U; 1024],
}

impl<'a, U, const LANES: usize> DecodedChunk<'a, U, LANES>
where
    U: NativePType + FastLanesDelta + Transpose,
{
    fn new(bases: &'a [U], deltas: &'a [U]) -> Self {
        Self {
            bases,
            deltas,
            chunk: None,
            transposed: [U::default(); 1024],
            values: [U::default(); 1024],
        }
    }

    #[inline]
    fn get(&mut self, chunk: usize) -> &[U; 1024] {
        if self.chunk != Some(chunk) {
            decode_chunk::<U, LANES>(
                self.bases,
                self.deltas,
                chunk,
                &mut self.transposed,
                &mut self.values,
            );
            self.chunk = Some(chunk);
        }
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::Delta;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::sparse_scattered(Mask::from_indices(3000, (0..3000).step_by(97)))]
    #[case::one_run_slices(Mask::from_slices(3000, vec![(1020, 1100)]))]
    #[case::last_chunk(Mask::from_indices(3000, [2047, 2048, 2999]))]
    #[case::full_words_across_chunks(Mask::from_slices(3000, vec![(960, 1150), (2000, 2100)]))]
    #[case::dense_falls_back(Mask::from_indices(3000, (0..3000).step_by(2)))]
    fn filter_matches_decoded(#[case] mask: Mask) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_option_iter(
            (0..3000i64).map(|v| (v % 11 != 0).then_some(v * 7 - 9_000)),
        );
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?.into_array();

        let actual = delta
            .filter(mask.clone())?
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected = primitive
            .into_array()
            .filter(mask)?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn filter_on_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter((0..3000u32).map(|v| v * 3));
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?
            .into_array()
            .slice(700..2900)?;
        let mask = Mask::from_indices(delta.len(), (0..delta.len()).step_by(13));

        let actual = delta
            .filter(mask.clone())?
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected = primitive
            .into_array()
            .slice(700..2900)?
            .filter(mask)?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }
}
