//! Force-monomorphization helpers. Each `force_*` call exercises every
//! width 1..T_BITS for the chosen integer type, optionally for the chosen
//! comparison closure. We feed runtime-sourced values through `black_box`
//! so the optimiser cannot collapse the dispatch and drop unused widths.

use std::hint::black_box;

use fastlanes::{BitPacking, BitPackingCompare, FastLanes, FastLanesComparable};

/// Materialise every width of `unpack` for `T`.
pub fn force_unpack<T>(width: usize, input: &[T], output: &mut [T])
where
    T: BitPacking + Copy + Default,
{
    // SAFETY: caller passes buffers sized to the test harness's needs.
    unsafe { T::unchecked_unpack(black_box(width), black_box(input), black_box(output)) }
}

/// Materialise every width of `unpack_cmp` for `T` with the given
/// `comparison` closure (one closure type per call site).
pub fn force_cmp<T, V, F>(
    width: usize,
    input: &[T],
    output: &mut [bool; 1024],
    comparison: F,
    value: V,
) where
    T: BitPackingCompare,
    V: FastLanesComparable<Bitpacked = T> + Copy,
    F: Fn(V, V) -> bool,
{
    // SAFETY: caller passes buffers sized to the test harness's needs.
    unsafe {
        T::unchecked_unpack_cmp(
            black_box(width),
            black_box(input),
            black_box(output),
            comparison,
            black_box(value),
        )
    }
}

/// Number of packed `T` units for a 1024-element block at width `W`.
pub const fn packed_len<T: FastLanes>(w: usize) -> usize {
    1024 * w / T::T
}
