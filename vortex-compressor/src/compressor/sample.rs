// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sampling utilities for compression ratio estimation.

use std::mem::size_of;

use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::CascadingCompressor;
use crate::scheme::CompressorContext;
use crate::scheme::EstimateScore;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::trace;

/// The size of each sampled run.
pub const SAMPLE_SIZE: u32 = 64;

/// The number of sampled runs.
///
/// # Warning
///
/// The product of `SAMPLE_SIZE` and `SAMPLE_COUNT` should be (roughly) a multiple of 1024 so that
/// fastlanes bitpacking of sampled vectors does not introduce (large amounts of) padding.
pub const SAMPLE_COUNT: u32 = 16;

/// Fixed seed for the sampling RNG, ensuring deterministic compression output.
const SAMPLE_SEED: u64 = 1234567890;

/// Samples approximately 1% of the input array for compression ratio estimation.
pub(crate) fn sample(input: &ArrayRef, sample_size: u32, sample_count: u32) -> ArrayRef {
    if input.len() <= (sample_size as usize) * (sample_count as usize) {
        return input.clone();
    }

    let slices = sample_slices(input.len(), sample_size, sample_count);

    // For every slice, grab the relevant slice and repack into a new PrimitiveArray.
    let chunks: Vec<_> = slices
        .into_iter()
        .map(|(start, end)| {
            input
                .slice(start..end)
                .vortex_expect("slice should succeed")
        })
        .collect();
    // SAFETY: all chunks are slices of `input`, so they share its dtype.
    unsafe { ChunkedArray::new_unchecked(chunks, input.dtype().clone()) }.into_array()
}

/// Returns deterministic stratified sample ranges for an array length.
pub(crate) fn sample_slices(
    length: usize,
    sample_size: u32,
    sample_count: u32,
) -> Vec<(usize, usize)> {
    stratified_slices(
        length,
        sample_size,
        sample_count,
        &mut StdRng::seed_from_u64(SAMPLE_SEED),
    )
}

/// Computes the number of sample chunks to cover approximately 1% of `len` elements,
/// with a minimum of `SAMPLE_SIZE * SAMPLE_COUNT` (1024) values.
pub(crate) fn sample_count_approx_one_percent(len: usize) -> u32 {
    let approximately_one_percent =
        (len / 100) / usize::try_from(SAMPLE_SIZE).vortex_expect("SAMPLE_SIZE must fit in usize");
    u32::max(
        u32::next_multiple_of(
            approximately_one_percent
                .try_into()
                .vortex_expect("sample count must fit in u32"),
            16,
        ),
        SAMPLE_COUNT,
    )
}

/// Divides an array into `sample_count` equal partitions and picks one random contiguous
/// slice of `sample_size` elements from each partition.
///
/// This is a stratified sampling strategy: instead of drawing all samples from one region,
/// it spreads them evenly across the array so that every part of the data is represented.
/// Each returned `(start, end)` pair is a half-open range into the original array.
///
/// If the total number of requested samples (`sample_size * sample_count`) is greater than or
/// equal to `length`, a single slice spanning the whole array is returned.
fn stratified_slices(
    length: usize,
    sample_size: u32,
    sample_count: u32,
    rng: &mut StdRng,
) -> Vec<(usize, usize)> {
    let total_num_samples: usize = (sample_count as usize) * (sample_size as usize);
    if total_num_samples >= length {
        return vec![(0usize, length)];
    }

    let partitions = partition_indices(length, sample_count);
    let num_samples_per_partition: Vec<usize> = partition_indices(total_num_samples, sample_count)
        .into_iter()
        .map(|(start, stop)| stop - start)
        .collect();

    partitions
        .into_iter()
        .zip(num_samples_per_partition)
        .map(|((start, stop), size)| {
            assert!(
                stop - start >= size,
                "Slices must be bigger than their sampled size"
            );
            let random_start = rng.random_range(start..=(stop - size));
            (random_start, random_start + size)
        })
        .collect()
}

/// Splits `[0, length)` into `num_partitions` contiguous, non-overlapping slices of
/// approximately equal size.
///
/// If `length` is not evenly divisible by `num_partitions`, the first
/// `length % num_partitions` slices get one extra element. Each returned `(start, end)` pair
/// is a half-open range.
fn partition_indices(length: usize, num_partitions: u32) -> Vec<(usize, usize)> {
    let num_long_parts = length % num_partitions as usize;
    let short_step = length / num_partitions as usize;
    let long_step = short_step + 1;
    let long_stop = num_long_parts * long_step;

    (0..long_stop)
        .step_by(long_step)
        .map(|off| (off, off + long_step))
        .chain(
            (long_stop..length)
                .step_by(short_step)
                .map(|off| (off, off + short_step)),
        )
        .collect()
}

/// Estimates compression ratio by compressing a ~1% sample of the data.
///
/// Creates a new [`ArrayAndStats`] for the sample so that stats are generated from the sample, not
/// the full array.
///
/// # Errors
///
/// Returns an error if sample compression fails.
pub(crate) fn estimate_compression_ratio_with_sampling<S: Scheme + ?Sized>(
    compressor: &CascadingCompressor,
    scheme: &S,
    array: &ArrayRef,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateScore> {
    let sample_array = if compress_ctx.is_sample() {
        array.clone()
    } else {
        let sample_count = sample_count_approx_one_percent(array.len());
        // `ArrayAndStats` expects a canonical array (so that it can easily compute lazy stats).
        let canonical: Canonical = sample(array, SAMPLE_SIZE, sample_count).execute(exec_ctx)?;
        canonical.into_array()
    };

    let sample_data = ArrayAndStats::new(sample_array, scheme.stats_options());
    let error_ctx = trace::enabled_error_context(&compress_ctx);
    let sample_ctx = compress_ctx.with_sampling();

    let compressed = match scheme.compress(compressor, &sample_data, sample_ctx, exec_ctx) {
        Ok(compressed) => compressed,
        Err(err) => {
            trace::sample_compress_failed(scheme.id(), error_ctx.as_ref(), &err);
            return Err(err);
        }
    };

    let after = compressed.nbytes();
    let before = canonical_visible_nbytes(sample_data.array(), exec_ctx)?;

    let score = EstimateScore::from_sample_sizes(before, after);

    if matches!(score, EstimateScore::ZeroBytes) {
        trace::zero_byte_sample_result(scheme.id(), before);
    }

    Ok(score)
}

/// Returns the physical canonical size without retained, unreferenced string payload bytes.
pub(crate) fn canonical_visible_nbytes(
    array: &ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let Some(array) = array.as_opt::<VarBinView>() else {
        return Ok(array.nbytes());
    };
    let validity = array.validity()?.execute_mask(array.len(), exec_ctx)?;
    let mut referenced_ranges = HashMap::<(usize, usize), Vec<(usize, usize)>>::new();
    let mut record_view = |view: &BinaryView| {
        if view.is_inlined() {
            return;
        }
        let reference = view.as_view();
        let buffer = array.buffer(reference.buffer_index as usize);
        referenced_ranges
            .entry((buffer.as_ptr().addr(), buffer.len()))
            .or_default()
            .push((
                reference.offset as usize,
                reference.offset as usize + reference.size as usize,
            ));
    };
    match &validity {
        Mask::AllTrue(_) => array.views().iter().for_each(&mut record_view),
        Mask::AllFalse(_) => {}
        Mask::Values(values) => array
            .views()
            .iter()
            .zip(values.bit_buffer().iter())
            .filter(|(_, is_valid)| *is_valid)
            .for_each(|(view, _)| record_view(view)),
    }
    let outlined_bytes = referenced_ranges.into_values().try_fold(
        0u64,
        |total, mut ranges| -> VortexResult<u64> {
            ranges.sort_unstable();
            let mut range_total = 0u64;
            let mut current = None::<(usize, usize)>;
            for (start, end) in ranges {
                match current {
                    Some((current_start, current_end)) if start <= current_end => {
                        current = Some((current_start, current_end.max(end)));
                    }
                    Some((current_start, current_end)) => {
                        range_total = range_total
                            .checked_add(u64::try_from(current_end - current_start)?)
                            .ok_or_else(|| {
                                vortex_error::vortex_err!("sample byte size overflowed u64")
                            })?;
                        current = Some((start, end));
                    }
                    None => current = Some((start, end)),
                }
            }
            if let Some((start, end)) = current {
                range_total = range_total
                    .checked_add(u64::try_from(end - start)?)
                    .ok_or_else(|| vortex_error::vortex_err!("sample byte size overflowed u64"))?;
            }
            total
                .checked_add(range_total)
                .ok_or_else(|| vortex_error::vortex_err!("sample byte size overflowed u64"))
        },
    )?;
    let views_bytes = u64::try_from(array.len())?
        .checked_mul(u64::try_from(size_of::<BinaryView>())?)
        .ok_or_else(|| vortex_error::vortex_err!("sample byte size overflowed u64"))?;
    let validity_bytes = match validity {
        Mask::Values(values) => u64::try_from(values.len().div_ceil(u8::BITS as usize))?,
        Mask::AllTrue(_) | Mask::AllFalse(_) => 0,
    };

    views_bytes
        .checked_add(outlined_bytes)
        .and_then(|bytes| bytes.checked_add(validity_bytes))
        .ok_or_else(|| vortex_error::vortex_err!("sample byte size overflowed u64"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn sample_is_deterministic() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Create a deterministic array with linear-with-noise pattern
        let values: Vec<i64> = (0i64..100_000).map(|i| i + (i * 7 + 3) % 11).collect();

        let array =
            PrimitiveArray::new(Buffer::from_iter(values), Validity::NonNullable).into_array();

        let first = sample(&array, SAMPLE_SIZE, SAMPLE_COUNT);
        for _ in 0..10 {
            let again = sample(&array, SAMPLE_SIZE, SAMPLE_COUNT);
            assert_eq!(first.nbytes(), again.nbytes());
            assert_arrays_eq!(&first, &again, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn visible_size_ignores_unreferenced_string_payload() -> VortexResult<()> {
        let values = (0..128)
            .map(|index| format!("outlined-string-value-{index:08x}"))
            .collect::<Vec<_>>();
        let source = VarBinViewArray::from_iter_str(&values).into_array();
        let slice = source.slice(17..19)?;
        let expected = u64::try_from(2 * size_of::<BinaryView>())?
            + u64::try_from(values[17].len() + values[18].len())?;
        let mut exec_ctx = array_session().create_execution_ctx();

        assert!(slice.nbytes() > expected);
        assert_eq!(canonical_visible_nbytes(&slice, &mut exec_ctx)?, expected);
        Ok(())
    }

    #[test]
    fn visible_size_counts_valid_outlined_values_and_validity() -> VortexResult<()> {
        let outlined = "an-outlined-string-value";
        let array = VarBinViewArray::from_iter(
            [Some("inline"), Some(outlined), None],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let expected = u64::try_from(3 * size_of::<BinaryView>() + outlined.len() + 1)?;
        let mut exec_ctx = array_session().create_execution_ctx();

        assert_eq!(canonical_visible_nbytes(&array, &mut exec_ctx)?, expected);
        Ok(())
    }

    #[test]
    fn visible_size_counts_shared_payload_once() -> VortexResult<()> {
        let value = b"one shared outlined value";
        let view = BinaryView::make_view(value, 0, 0);
        let array = VarBinViewArray::try_new(
            Buffer::copy_from([view, view]),
            Arc::from([ByteBuffer::copy_from(value)]),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
            &mut array_session().create_execution_ctx(),
        )?
        .into_array();
        let expected = u64::try_from(2 * size_of::<BinaryView>() + value.len())?;

        assert_eq!(
            canonical_visible_nbytes(&array, &mut array_session().create_execution_ctx())?,
            expected
        );
        Ok(())
    }
}
