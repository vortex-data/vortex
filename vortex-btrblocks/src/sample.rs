// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexExpect;

use crate::stats::SAMPLE_COUNT;
use crate::stats::SAMPLE_SIZE;

pub(crate) fn sample(input: &ArrayRef, sample_size: u32, sample_count: u32) -> ArrayRef {
    if input.len() <= (sample_size as usize) * (sample_count as usize) {
        return input.to_array();
    }

    let slices = stratified_slices(
        input.len(),
        sample_size,
        sample_count,
        &mut StdRng::seed_from_u64(1234567890u64),
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
    ChunkedArray::try_new(chunks, input.dtype().clone())
        .vortex_expect("sample slices should form valid chunked array")
        .into_array()
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

/// Split a range of array indices into as-equal-as-possible slices. If the provided `num_partitions` doesn't
/// evenly divide into `length`, then the first `(length % num_partitions)` slices will have an extra element.
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
