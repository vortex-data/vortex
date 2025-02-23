use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::{Array, ArrayRef, Canonical, IntoArray};

/// Limit array to start...stop range
pub trait SliceFn<A> {
    /// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
    /// If start >= stop, returns an empty array of the same type as `self`.
    /// Assumes that start or stop are out of bounds, may panic otherwise.
    fn slice(&self, array: A, start: usize, stop: usize) -> VortexResult<ArrayRef>;
}

impl<E: Encoding> SliceFn<&dyn Array> for E
where
    E: for<'a> SliceFn<&'a E::Array>,
{
    fn slice(&self, array: &dyn Array, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        let vtable = array.vtable();

        SliceFn::slice(self, array_ref, start, stop)
    }
}

/// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
///
/// # Errors
///
/// Slicing returns an error if you attempt to slice a range that exceeds the bounds of the
/// underlying array.
///
/// Slicing returns an error if the underlying codec's [slice](SliceFn::slice()) implementation
/// returns an error.
pub fn slice(array: &dyn Array, start: usize, stop: usize) -> VortexResult<ArrayRef> {
    if start == 0 && stop == array.len() {
        return Ok(array.to_array());
    }

    if start == stop {
        return Ok(Canonical::empty(array.dtype()).into_array());
    }

    check_slice_bounds(array, start, stop)?;

    // We know that constant array don't need stats propagation, so we can avoid the overhead of
    // computing derived stats and merging them in.
    let derived_stats = (!array.is_constant()).then(|| derive_sliced_stats(array));

    let sliced = array
        .vtable()
        .slice_fn()
        .map(|f| f.slice(array, start, stop))
        .unwrap_or_else(|| {
            Err(vortex_err!(
                NotImplemented: "slice",
                array.encoding()
            ))
        })?;

    if let Some(derived_stats) = derived_stats {
        let mut stats = sliced.statistics().stats_set();
        stats.combine_sets(&derived_stats, array.dtype())?;
        for (stat, val) in stats.into_iter() {
            sliced.statistics().set_stat(stat, val)
        }
    }

    debug_assert_eq!(
        sliced.len(),
        stop - start,
        "Slice length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        sliced.dtype(),
        array.dtype(),
        "Slice dtype mismatch {}",
        array.encoding()
    );

    Ok(sliced)
}

fn derive_sliced_stats(arr: &dyn Array) -> StatsSet {
    let stats = arr.statistics().stats_set();

    // an array that is not constant can become constant after slicing
    let is_constant = stats.get_as::<bool>(Stat::IsConstant);
    let is_sorted = stats.get_as::<bool>(Stat::IsConstant);
    let is_strict_sorted = stats.get_as::<bool>(Stat::IsConstant);

    let mut stats = stats.keep_inexact_stats(&[
        Stat::Max,
        Stat::Min,
        Stat::RunCount,
        Stat::TrueCount,
        Stat::NullCount,
        Stat::UncompressedSizeInBytes,
    ]);

    if is_constant == Some(Precision::Exact(true)) {
        stats.set(Stat::IsConstant, Precision::exact(true));
    }
    if is_sorted == Some(Precision::Exact(true)) {
        stats.set(Stat::IsSorted, Precision::exact(true));
    }
    if is_strict_sorted == Some(Precision::Exact(true)) {
        stats.set(Stat::IsStrictSorted, Precision::exact(true));
    }

    stats
}

fn check_slice_bounds(array: &dyn Array, start: usize, stop: usize) -> VortexResult<()> {
    if start > array.len() {
        vortex_bail!(OutOfBounds: start, 0, array.len());
    }
    if stop > array.len() {
        vortex_bail!(OutOfBounds: stop, 0, array.len());
    }
    if start > stop {
        vortex_bail!("start ({start}) must be <= stop ({stop})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_scalar::Scalar;

    use crate::arrays::{ConstantArray, PrimitiveArray};
    use crate::compute::slice;
    use crate::stats::{Precision, Stat, Statistics, STATS_TO_WRITE};
    use crate::Array;

    #[test]
    fn test_slice_primitive() {
        let c = PrimitiveArray::from_iter(0i32..100);
        c.compute_all(STATS_TO_WRITE).unwrap();

        let c2 = slice(&c, 10, 20).unwrap();

        let result_stats = c2.statistics().stats_set();
        assert_eq!(
            result_stats.get_as::<i32>(Stat::Max),
            Some(Precision::inexact(99))
        );
        assert_eq!(
            result_stats.get_as::<i32>(Stat::Min),
            Some(Precision::inexact(0))
        );
    }

    #[test]
    fn test_slice_const() {
        let c = ConstantArray::new(Scalar::from(10), 100);
        c.compute_all(STATS_TO_WRITE).unwrap();

        let c2 = slice(&c, 10, 20).unwrap();
        let result_stats = c2.statistics().stats_set();

        // Constant always knows its exact stats
        assert_eq!(
            result_stats.get_as::<i32>(Stat::Max),
            Some(Precision::exact(10))
        );
        assert_eq!(
            result_stats.get_as::<i32>(Stat::Min),
            Some(Precision::exact(10))
        );
    }
}
