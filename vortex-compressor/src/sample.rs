// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sampling utilities for compression ratio estimation.

use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexExpect;

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

    let slices = stratified_slices(
        input.len(),
        sample_size,
        sample_count,
        &mut StdRng::seed_from_u64(SAMPLE_SEED),
    );

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

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn sample_is_deterministic() -> VortexResult<()> {
        // Create a deterministic array with linear-with-noise pattern
        let values: Vec<i64> = (0i64..100_000).map(|i| i + (i * 7 + 3) % 11).collect();

        let array =
            PrimitiveArray::new(Buffer::from_iter(values), Validity::NonNullable).into_array();

        let first = sample(&array, SAMPLE_SIZE, SAMPLE_COUNT);
        for _ in 0..10 {
            let again = sample(&array, SAMPLE_SIZE, SAMPLE_COUNT);
            assert_eq!(first.nbytes(), again.nbytes());
            assert_arrays_eq!(&first, &again);
        }
        Ok(())
    }
}
