// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct entry points to the fixed-width filter strategies and their configured thresholds.
//!
//! [`crate::arrays::filter`]'s buffer dispatch chooses between these strategies with density
//! thresholds that were benchmarked per architecture. Normal dispatch runs exactly one strategy
//! per mask, so the crossovers cannot be measured through the public filter API. These wrappers
//! bypass dispatch to let every strategy be timed against the others at any density, and expose
//! the compiled-in thresholds to compare against.
//!
//! `examples/filter_threshold_verification.rs` uses this module to rederive and verify the
//! thresholds on new machines and architectures:
//!
//! ```sh
//! cargo run --release -p vortex-array --features _test-harness \
//!     --example filter_threshold_verification
//! ```

use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_mask::MaskValues;

use crate::arrays::filter::execute::buffer;
use crate::arrays::filter::execute::byte_compress;
use crate::arrays::filter::execute::simd_compress;
use crate::arrays::filter::execute::slice;

/// The maximum density at which dispatch gathers by already-cached mask indices.
pub const CACHED_INDICES_MAX_DENSITY: f64 = buffer::CACHED_INDICES_MAX_DENSITY;

/// The minimum average run length at which dispatch copies already-cached mask slices.
pub const MIN_SLICES_AVERAGE_RUN_LENGTH: usize = buffer::MIN_SLICES_AVERAGE_RUN_LENGTH;

/// The minimum mask length required by the SIMD compress kernels.
pub const SIMD_MIN_LEN: usize = simd_compress::MIN_LEN;

/// Filter with the scalar bitmap walk, dispatch's fallback strategy.
pub fn filter_by_bitmap_walk<T: Copy>(values: &[T], mask: &MaskValues) -> Buffer<T> {
    slice::filter_slice_by_bitmap(values, mask)
}

/// Filter with the byte-compress permutation LUT.
pub fn filter_by_byte_compress<T: Copy>(values: &[T], mask: &MaskValues) -> Buffer<T> {
    byte_compress::filter_buffer(values, mask)
}

/// Filter with this CPU's SIMD compress kernel for `T`, ignoring the configured density band.
///
/// Returns `None` when no kernel exists for this element width on this CPU, or when the mask is
/// shorter than [`SIMD_MIN_LEN`].
pub fn filter_by_simd_compress<T: Copy>(values: &[T], mask: &MaskValues) -> Option<Buffer<T>> {
    simd_compress::filter_slice_by_bitmap_any_density(values, mask)
}

/// Filter by gathering strictly increasing indices, dispatch's cached-indices strategy.
pub fn filter_by_indices<T: Copy>(values: &[T], indices: &[usize]) -> Buffer<T> {
    slice::filter_slice_by_indices(values, indices)
}

/// Filter by copying strictly increasing `(start, end)` ranges, dispatch's cached-slices
/// strategy.
pub fn filter_by_slices<T: Copy>(
    values: &[T],
    slices: &[(usize, usize)],
    true_count: usize,
) -> Buffer<T> {
    slice::filter_slice_by_slices(values, slices, true_count)
}

/// The configured density at or above which dispatch prefers byte compress over the bitmap walk
/// for element type `T` on this target.
pub fn byte_compress_density_threshold<T>() -> f64 {
    buffer::byte_compress_density_threshold::<T>()
}

/// The SIMD kernel name and configured density band for element type `T` on this CPU, or `None`
/// when no kernel exists for the width.
pub fn simd_density_band<T>() -> Option<(&'static str, Range<f64>)> {
    simd_compress::configured_density_band::<T>()
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use super::*;
    use crate::dtype::i256;

    fn strategies_agree<T: Copy + PartialEq + std::fmt::Debug>(
        make: impl Fn(usize) -> T,
    ) -> VortexResult<()> {
        const LEN: usize = 300;

        let values: Vec<T> = (0..LEN).map(make).collect();
        let keep = |index: usize| index.is_multiple_of(3) || index % 7 == 1;

        let mask = Mask::from_buffer(BitBuffer::from_iter((0..LEN).map(keep)));
        let mask = mask
            .values()
            .ok_or_else(|| vortex_err!("mixed mask must have values"))?;

        let expected: Vec<T> = (0..LEN).filter(|&i| keep(i)).map(|i| values[i]).collect();
        let indices: Vec<usize> = (0..LEN).filter(|&i| keep(i)).collect();

        assert_eq!(
            filter_by_bitmap_walk(&values, mask).as_slice(),
            expected.as_slice()
        );
        assert_eq!(
            filter_by_byte_compress(&values, mask).as_slice(),
            expected.as_slice()
        );
        assert_eq!(
            filter_by_indices(&values, &indices).as_slice(),
            expected.as_slice()
        );
        if let Some(filtered) = filter_by_simd_compress(&values, mask) {
            assert_eq!(filtered.as_slice(), expected.as_slice());
        }

        Ok(())
    }

    #[test]
    fn test_strategies_agree() -> VortexResult<()> {
        strategies_agree(|index| index as u8)?;
        strategies_agree(|index| index as u16)?;
        strategies_agree(|index| index as u32)?;
        strategies_agree(|index| index as u64)?;
        strategies_agree(|index| index as u128)?;
        strategies_agree(|index| i256::from_i128(index as i128))?;
        Ok(())
    }

    #[test]
    fn test_filter_by_slices() {
        let values: Vec<u32> = (0..100).collect();
        let slices = [(10usize, 20usize), (40, 45), (90, 100)];
        let expected: Vec<u32> = (10..20).chain(40..45).chain(90..100).collect();

        let filtered = filter_by_slices(&values, &slices, expected.len());
        assert_eq!(filtered.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_configured_thresholds_are_sane() {
        for threshold in [
            byte_compress_density_threshold::<u8>(),
            byte_compress_density_threshold::<u16>(),
            byte_compress_density_threshold::<u32>(),
            byte_compress_density_threshold::<u64>(),
            byte_compress_density_threshold::<u128>(),
            byte_compress_density_threshold::<i256>(),
        ] {
            assert!((0.0..=1.0).contains(&threshold));
        }

        if let Some((name, band)) = simd_density_band::<u32>() {
            assert!(!name.is_empty());
            assert!(band.start < band.end);
        }
    }
}
