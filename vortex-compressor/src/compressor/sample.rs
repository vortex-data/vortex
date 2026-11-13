// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sampling utilities for compression ratio estimation.

use std::sync::Arc;

use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::cascade::exact_uncompressed_size;
use crate::CascadingCompressor;
use crate::scheme::CompressorContext;
use crate::scheme::EstimateScore;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
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
    let chunks = slices.into_iter().map(|(start, end)| {
        input
            .slice(start..end)
            .vortex_expect("slice should succeed")
    });
    // SAFETY: all chunks are slices of `input`, so they share its dtype.
    unsafe { ChunkedArray::new_unchecked(chunks, input.dtype().clone()) }.into_array()
}

/// Samples `chunk_count` whole chunks of `chunk_len` rows that start at multiples of
/// `chunk_len`, one from each of `chunk_count` equal partitions of the array's full chunks.
///
/// Use this instead of the default compression sample for schemes that encode each chunk
/// independently, such as the FastLanes encodings with `chunk_len = 1024`. The default sample
/// joins short runs from across the array, so a chunk of it mixes rows that were never adjacent;
/// an aligned chunk here holds exactly the rows the scheme would encode together.
///
/// Returns `None` when the array has no more than `chunk_count` full chunks, in which case the
/// caller should use the whole array. The trailing partial chunk is never sampled. Sampling is
/// deterministic.
///
/// # Panics
///
/// Panics if `chunk_len` or `chunk_count` is zero.
pub fn sample_aligned_chunks(
    input: &ArrayRef,
    chunk_len: usize,
    chunk_count: u32,
) -> Option<ArrayRef> {
    let slices = aligned_chunk_slices(input.len(), chunk_len, chunk_count)?;

    let chunks = slices.into_iter().map(|(start, end)| {
        input
            .slice(start..end)
            .vortex_expect("slice should succeed")
    });
    // SAFETY: all chunks are slices of `input`, so they share its dtype.
    Some(unsafe { ChunkedArray::new_unchecked(chunks, input.dtype().clone()) }.into_array())
}

/// Returns the deterministic ranges [`sample_aligned_chunks`] slices, or `None` when the array
/// has no more than `chunk_count` full chunks.
fn aligned_chunk_slices(
    length: usize,
    chunk_len: usize,
    chunk_count: u32,
) -> Option<Vec<(usize, usize)>> {
    assert!(chunk_len > 0, "chunk_len must be positive");
    assert!(chunk_count > 0, "chunk_count must be positive");

    let full_chunks = length / chunk_len;
    if full_chunks <= chunk_count as usize {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(SAMPLE_SEED);
    Some(
        partition_indices(full_chunks, chunk_count)
            .into_iter()
            .map(|(start, stop)| {
                let chunk = rng.random_range(start..stop);
                (chunk * chunk_len, (chunk + 1) * chunk_len)
            })
            .collect(),
    )
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

/// The sample every sampled scheme compresses to estimate its ratio on one array.
///
/// It is built once per array and cached in the array's [`ArrayAndStats`], so the rows are copied,
/// their stats generated and their size measured once, however many schemes sample them.
struct CompressionSample {
    /// The sampled rows and their stats cache.
    data: ArrayAndStats,
    /// The bytes the sampled rows reference.
    before: u64,
    /// Whether every sampled row is null.
    all_invalid: bool,
}

/// Returns the compression sample of `data`, building and caching it on first use.
///
/// The compressor builds `data` with the merged stats options of every scheme it considers, so
/// the cached sample serves them all. A caller whose scheme needs more stats gets an uncached
/// sample.
fn compression_sample(
    data: &ArrayAndStats,
    scheme_opts: GenerateStatsOptions,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Arc<CompressionSample>> {
    let opts = data.stats_options();
    // Inside sample compression the sample is `data` itself, which shares its stats cache.
    // Caching it there would make the cache own a handle to itself.
    if compress_ctx.is_sample() || opts.merge(scheme_opts) != opts {
        return build_sample(data, opts.merge(scheme_opts), compress_ctx, exec_ctx).map(Arc::new);
    }
    if let Some(sample) = data.get::<CompressionSample>() {
        return Ok(sample);
    }
    let sample = build_sample(data, opts, compress_ctx, exec_ctx)?;
    Ok(data.get_or_insert_with(|| sample))
}

/// Builds the compression sample of `data` with stats generated using `opts`.
///
/// Inside sample compression the array is already a sample, so it is used whole, sharing the
/// stats cache when the options match.
fn build_sample(
    data: &ArrayAndStats,
    opts: GenerateStatsOptions,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<CompressionSample> {
    let sample_data = if !compress_ctx.is_sample() {
        // `ArrayAndStats` expects a canonical array (so that it can easily compute lazy stats).
        let canonical: Canonical = sample_rows(data.array()).execute(exec_ctx)?;
        ArrayAndStats::new(canonical.into_array(), opts)
    } else if opts == data.stats_options() {
        data.clone()
    } else {
        ArrayAndStats::new(data.array().clone(), opts)
    };

    // Sampled slices still hold the full array's buffers, so measure what their rows reference.
    let before = exact_uncompressed_size(sample_data.array(), exec_ctx)?;
    let all_invalid = !compress_ctx.is_sample() && sample_data.array().all_invalid(exec_ctx)?;
    Ok(CompressionSample {
        data: sample_data,
        before,
        all_invalid,
    })
}

/// Selects the rows of `array` that sampled schemes compress: about 1% of them.
fn sample_rows(array: &ArrayRef) -> ArrayRef {
    sample(
        array,
        SAMPLE_SIZE,
        sample_count_approx_one_percent(array.len()),
    )
}

/// Estimates a scheme's compression ratio by compressing the array's compression sample.
///
/// The sample is shared by every sampled scheme on the array; see [`compression_sample`].
///
/// # Errors
///
/// Returns an error if sample compression fails.
pub(crate) fn estimate_compression_ratio_with_sampling<S: Scheme + ?Sized>(
    compressor: &CascadingCompressor,
    scheme: &S,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateScore> {
    let sample = compression_sample(data, scheme.stats_options(), &compress_ctx, exec_ctx)?;

    // A sample of only nulls compresses to almost nothing whatever the scheme, so its ratio says
    // nothing about how the scheme handles the array's values.
    if sample.all_invalid {
        return Ok(EstimateScore::ZeroBytes);
    }

    let error_ctx = trace::enabled_error_context(&compress_ctx);
    let sample_ctx = compress_ctx.with_sampling();

    let compressed = match scheme.compress(compressor, &sample.data, sample_ctx, exec_ctx) {
        Ok(compressed) => compressed,
        Err(err) => {
            trace::sample_compress_failed(scheme.id(), error_ctx.as_ref(), &err);
            return Err(err);
        }
    };

    let after = compressed.nbytes();
    let score = EstimateScore::from_sample_sizes(sample.before, after);

    if matches!(score, EstimateScore::ZeroBytes) {
        trace::zero_byte_sample_result(scheme.id(), sample.before);
    }

    Ok(score)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn sample_is_deterministic() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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
    fn aligned_chunk_slices_are_whole_aligned_and_spread() {
        let slices = aligned_chunk_slices(100 * 1024 + 7, 1024, 4).unwrap();
        assert_eq!(slices.len(), 4);
        for (i, &(start, end)) in slices.iter().enumerate() {
            assert_eq!(start % 1024, 0);
            assert_eq!(end - start, 1024);
            // One chunk from each quarter of the 100 full chunks; the partial tail is excluded.
            assert!((i * 25 * 1024..(i + 1) * 25 * 1024).contains(&start));
        }
    }

    #[test]
    fn aligned_chunks_need_more_full_chunks_than_requested() {
        assert!(aligned_chunk_slices(4 * 1024 + 1023, 1024, 4).is_none());
        assert!(aligned_chunk_slices(5 * 1024, 1024, 4).is_some());
    }

    #[test]
    fn sample_aligned_chunks_matches_source_rows() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 64 * 1024;
        let array = PrimitiveArray::new(
            Buffer::from_iter(0..u32::try_from(len)?),
            Validity::NonNullable,
        )
        .into_array();

        let sampled = sample_aligned_chunks(&array, 1024, 4).unwrap();
        assert_eq!(sampled.len(), 4 * 1024);

        let chunks = aligned_chunk_slices(len, 1024, 4)
            .unwrap()
            .into_iter()
            .map(|(start, end)| array.slice(start..end))
            .collect::<VortexResult<Vec<_>>>()?;
        let expected = ChunkedArray::try_new(chunks, array.dtype().clone())?.into_array();
        assert_arrays_eq!(sampled, expected, &mut ctx);
        Ok(())
    }
}
