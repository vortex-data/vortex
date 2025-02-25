use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::arrays::ChunkedArray;
use vortex_array::compute::slice;
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexExpect;

pub(crate) fn sample<T: Array + Clone>(input: T, sample_size: u16, sample_count: u16) -> ArrayRef {
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
    ChunkedArray::try_new(
        slices
            .into_iter()
            .map(|(start, end)| slice(&input, start, end).vortex_expect("slice"))
            .collect(),
        input.dtype().clone(),
    )
    .vortex_expect("sample")
    .into_array()
}

pub fn stratified_slices(
    length: usize,
    sample_size: u16,
    sample_count: u16,
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
pub fn partition_indices(length: usize, num_partitions: u16) -> Vec<(usize, usize)> {
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
